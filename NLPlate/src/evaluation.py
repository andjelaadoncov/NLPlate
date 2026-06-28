"""
Za svaki upit gradi se isti "candidate pool" (filtriranje + TF-IDF top-N).
Komponente se racunaju JEDNOM po upitu. Zatim se:
  - porede baseline modeli (popularnost, TF-IDF, SBERT, hibrid),
  - pretrazuje mreza (grid) tezina i bira kombinacija sa najboljim NDCG@10.

Zbog kesiranja komponenti, pretraga po stotinama kombinacija tezina je brza
(svaka kombinacija = samo ponderisani zbir vec izracunatih komponenti).
"""

# deo za evaluaciju i kako se odredjuju koeficijenti tezinski
# recept je pouzdan ako se poklapa sa unosom i ako ima dovoljan kvalitet i broj recenzija

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, metrics
from .preferences import Preferences, parse_query_preferences
from .recommender import NLPlateRecommender, normalize_weights
from .test_queries import TEST_QUERIES

# provera poklapanja sa unosom 
def _matches_constraints(row, prefs: Preferences) -> bool:
    ingr_join = " ".join(str(x).lower() for x in row["ingredients_list"])
    tags_set = {str(x).lower() for x in row["tags_list"]}

    if prefs.include_ingredients:
        if not all(i.lower() in ingr_join for i in prefs.include_ingredients):
            return False
    if prefs.exclude_ingredients:
        if any(e.lower() in ingr_join for e in prefs.exclude_ingredients):
            return False
    if prefs.tags:
        if not all(t.lower() in tags_set for t in prefs.tags):
            return False
    if prefs.max_minutes:
        m = row["minutes"]
        if not (pd.notna(m) and m <= prefs.max_minutes):
            return False
    return True

# daje ocenu pouzdanosti tj:
"""
 0 = nije na temu / neproveren (malo recenzija) / na temu ali lose ocenjen
 1 = na temu i solidno ocenjen      (Bayes prosek ~ >= 3.5)
 2 = na temu i dobro ocenjen        (Bayes prosek ~ >= 4.0)
 3 = na temu i odlicno ocenjen      (Bayes prosek ~ >= 4.5)

"""
def _relevance_grade(matches: bool, quality: float, n_reviews: int) -> int:

    if not matches:
        return 0
    if n_reviews < config.MIN_REVIEWS_FOR_QUALITY:
        return 0  # nije relevantno jer ne moze da se utrdi kvalitet premalo ocena 
    if quality >= 0.90:   
        return 3 # odlicno ocenjen i jako relevantan
    if quality >= 0.80:    
        return 2 # relevantan i dobro ocenjen
    if quality >= 0.70:    
        return 1 # delimicno
    return 0  # u skladu se ja zahtevima ali je lose ocenjen        

# struktura za evaluaciju
@dataclass
class QueryEvalData:
    query: str
    candidate_idx: np.ndarray
    components: dict[str, np.ndarray]
    graded: np.ndarray      # gradirana relevantnost po kandidatu
    binary: np.ndarray      # binarna relevantnost po kandidatu
    total_relevant: int     # broj binarno-relevantnih u pool-u


# evaluacija modela i pretraga tezina nad fiksnim skupom upita
class Evaluator:

    def __init__(self, recommender: NLPlateRecommender,
                 queries: list[str] | None = None,
                 ks: tuple[int, ...] = (5, 10)):
        self.rec = recommender
        self.queries = queries or TEST_QUERIES
        self.ks = ks
        self._cache: list[QueryEvalData] | None = None


    def prepare(self) -> None:
        cache: list[QueryEvalData] = []
        for q in self.queries:
            prefs = parse_query_preferences(q)
            idx = self.rec._candidate_indices(q, prefs, config.CANDIDATE_POOL_SIZE)
            content_sims = getattr(self.rec, "_last_content_sims", None)
            comps = self.rec.component_scores(q, idx, content_sims)

            sub = self.rec.recipes.iloc[idx]
            graded = np.zeros(len(idx), dtype=float)
            for j, (_, row) in enumerate(sub.iterrows()):
                matches = _matches_constraints(row, prefs)
                graded[j] = _relevance_grade(
                    matches, float(row["rating_quality"]), int(row["n_reviews"])
                )
            binary = (graded >= 2).astype(int)
            cache.append(QueryEvalData(
                query=q,
                candidate_idx=idx,
                components=comps,
                graded=graded,
                binary=binary,
                total_relevant=int(binary.sum()),
            ))
        self._cache = cache

    def _ensure_cache(self) -> list[QueryEvalData]:
        if self._cache is None:
            self.prepare()
        return self._cache  

    def _eval_scores(self, data: QueryEvalData, scores: np.ndarray) -> dict[str, float]:
        order = np.argsort(-scores)
        graded_ranked = data.graded[order]
        binary_ranked = data.binary[order]
        return metrics.evaluate_at_ks(
            graded_ranked, binary_ranked, data.total_relevant, ks=self.ks
        )

    def _average(self, per_query: list[dict[str, float]]) -> dict[str, float]:
        keys = per_query[0].keys()
        out = {}
        for key in keys:
            vals = np.array([d[key] for d in per_query], dtype=float)
            out[key] = float(np.nanmean(vals))
        return out

    # poredi rangiranje po kvalitetu, tf-idf komponentu, sbert, hibrid sa tezinama iz predloga
    def evaluate_baselines(self) -> pd.DataFrame:
        cache = self._ensure_cache()
        models: dict[str, list[dict[str, float]]] = {
            "popularity": [], "tfidf": [], "hybrid_baseline": [],
        }
        if self.rec.has_semantic:
            models["sbert"] = []

        for data in cache:
            comps = data.components
            models["popularity"].append(self._eval_scores(data, comps["rating_quality"]))
            models["tfidf"].append(self._eval_scores(data, comps["content"]))
            if self.rec.has_semantic:
                models["sbert"].append(self._eval_scores(data, comps["semantic"]))
            hybrid = self.rec.combine(comps, config.BASELINE_WEIGHTS)
            models["hybrid_baseline"].append(self._eval_scores(data, hybrid))

        rows = []
        for name, lst in models.items():
            avg = self._average(lst)
            rows.append({"model": name, **avg})
        return pd.DataFrame(rows).set_index("model")
    
    # uzima neku kombinaciju tezina i racuna metrike za nju
    def evaluate_weights(self, weights: dict[str, float]) -> dict[str, float]:
        cache = self._ensure_cache()
        per_query = []
        for data in cache:
            scores = self.rec.combine(data.components, weights)
            per_query.append(self._eval_scores(data, scores))
        return self._average(per_query)
    
    # sistem pretrazuje razlicite kombinacije tezinskih koeficijenata i bira onu koja daje najbolji score na test uzorku (NDCG@10 score)
    def grid_search(
        self,
        step: float = 0.1,
        primary_metric: str = "ndcg@10",
        include_semantic: bool | None = None,
        min_weight: float = 0.0,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        
        if include_semantic is None:
            include_semantic = self.rec.has_semantic

        components = list(config.COMPONENTS)
        if not include_semantic:
            components = [c for c in components if c != "semantic"]

        if min_weight < 0:
            raise ValueError("min_weight mora biti >= 0.")

        if min_weight * len(components) > 1.0:
            raise ValueError( "min_weight je prevelik za broj komponenti. " "Ukupna minimalna suma prelazi 1.")

        combos = _weight_combinations(components, step)

        if min_weight > 0: 
            combos = [
                combo for combo in combos
                if all(combo.get(c, 0.0) >= min_weight for c in components)
            ]
            
        print(f"[grid_search] Broj kombinacija tezina: {len(combos)} "
              f"(komponente: {components})")

        results = []
        best = None
        best_score = -np.inf
        for combo in combos:
            w = {c: 0.0 for c in config.COMPONENTS}
            w.update(combo)
            avg = self.evaluate_weights(w)
            score = avg.get(primary_metric, float("nan"))
            row = {**{f"w_{k}": w[k] for k in config.COMPONENTS}, **avg}
            results.append(row)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best = dict(w)

        df = pd.DataFrame(results).sort_values(primary_metric, ascending=False)
        if best is None:
            best = dict(config.BASELINE_WEIGHTS)
        return best, df.reset_index(drop=True)

    # cuvaju se te tezine
    def save_tuned_weights(self, weights: dict[str, float],
                           primary_metric: str, score: float) -> None:
        config.ensure_dirs()
        payload = {
            "weights": {k: float(weights[k]) for k in config.COMPONENTS},
            "selected_by": primary_metric,
            "score": float(score),
            "n_queries": len(self.queries),
            "semantic_used": self.rec.has_semantic,
        }
        with open(config.TUNED_WEIGHTS_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[evaluation] Snimljene istrenirane tezine: {config.TUNED_WEIGHTS_JSON}")


# fje za pravljenje svih mogucih kombinacija koje se testiraju grid search-u
def _weight_combinations(components: list[str], step: float) -> list[dict[str, float]]:
    n_steps = int(round(1.0 / step))
    combos = []
    k = len(components)
    for parts in _compositions(n_steps, k):
        w = {comp: parts[i] * step for i, comp in enumerate(components)}
        combos.append(w)
    return combos


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest
