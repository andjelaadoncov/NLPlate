# recommender sistem (main py file)

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, content_based, semantic
from .preferences import Preferences, parse_query_preferences, merge_preferences
from .rating_quality import build_quality_table

# ucitavanje preprocessed fajlova 
def _read_parquet_or_pickle(path) -> pd.DataFrame:
    from pathlib import Path
    p = Path(path)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    alt = Path(str(p).replace(".parquet", ".pkl"))
    if alt.exists():
        return pd.read_pickle(alt)
    raise FileNotFoundError(f"Nije pronadjen ni {p} ni {alt}. Pokreni preprocesiranje.")

# ucitava podesene tezine iz json fajla, ili baseline tezine predlozene u predlogu projekta
def load_tuned_weights() -> dict[str, float]:
    if config.TUNED_WEIGHTS_JSON.exists():
        with open(config.TUNED_WEIGHTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("weights", data)

        if all(k in weights for k in config.COMPONENTS):
            return {k: float(weights[k]) for k in config.COMPONENTS}
    return dict(config.BASELINE_WEIGHTS)

# normalizacija tezina kako bi zbir bio 1
def normalize_weights(weights: dict[str, float], drop_semantic: bool) -> dict[str, float]:
    w = dict(weights)
    if drop_semantic:
        w["semantic"] = 0.0
    total = sum(w.values())
    if total <= 0:
        active = [k for k in config.COMPONENTS if not (drop_semantic and k == "semantic")]
        return {k: (1.0 / len(active) if k in active else 0.0) for k in config.COMPONENTS}
    return {k: v / total for k, v in w.items()}

def _to_list(value) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, str):
        return [value] if value.strip() else []

    return []


@dataclass
class Recommendation:
    recipe_id: int
    name: str
    final_score: float
    components: dict[str, float]
    minutes: float
    n_reviews: int
    rating_mean: float
    ingredients: list[str]
    tags: list[str]
    description: str
    steps: list[str]
    explanation: str


class NLPlateRecommender:

    def __init__(
        self,
        recipes: pd.DataFrame,
        tfidf_vectorizer,
        tfidf_matrix,
        embeddings: np.ndarray | None,
        weights: dict[str, float] | None = None,
    ):
        self.recipes = recipes.reset_index(drop=True)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.embeddings = embeddings
        self.weights = weights or load_tuned_weights()
        self.has_semantic = embeddings is not None and semantic.is_available()

        self._quality = self.recipes["rating_quality"].to_numpy(dtype=np.float32)
        self._agreement = self.recipes["agreement"].to_numpy(dtype=np.float32)
        self._sentiment = self.recipes["sentiment"].to_numpy(dtype=np.float32)
        self._minutes = self.recipes["minutes"].to_numpy(dtype=np.float64)
        self._n_reviews = self.recipes["n_reviews"].to_numpy(dtype=np.float64)

    # ucitavanje preprocessed fajlova i sentimenta tf-idf vektora i matricu i embeddinge 
    @classmethod
    def from_processed(cls, weights: dict[str, float] | None = None) -> "NLPlateRecommender":
        recipes = _read_parquet_or_pickle(config.RECIPES_CLEAN_PARQUET)
        stats = _read_parquet_or_pickle(config.RECIPE_STATS_PARQUET)

        # kvalitet ocena + agreement
        quality = build_quality_table(stats)
        global_mean = float(quality["global_mean"].iloc[0]) if len(quality) else 4.0

        # sentiment
        from pathlib import Path
        sent_path = config.PROCESSED_DIR / "recipe_sentiment.parquet"
        try:
            sentiment_df = _read_parquet_or_pickle(sent_path)
        except FileNotFoundError:
            sentiment_df = pd.DataFrame(columns=["id", "sentiment", "sentiment_pos_ratio"])

        # spajanje na recepte
        merged = recipes.merge(
            quality[["id", "rating_mean", "rating_std", "n_reviews",
                     "rating_quality", "agreement"]],
            on="id", how="left",
        )
        merged = merged.merge(
            sentiment_df[["id", "sentiment", "sentiment_pos_ratio"]]
            if "sentiment" in sentiment_df.columns else sentiment_df.assign(sentiment=np.nan, sentiment_pos_ratio=np.nan),
            on="id", how="left",
        )

        # ako recept nema recenzije deo
        merged["rating_mean"] = merged["rating_mean"].fillna(global_mean)
        merged["rating_std"] = merged["rating_std"].fillna(0.0)
        merged["n_reviews"] = merged["n_reviews"].fillna(0).astype(int)
        merged["rating_quality"] = merged["rating_quality"].fillna(global_mean / 5.0)
        merged["agreement"] = merged["agreement"].fillna(0.5)   # neutralno
        merged["sentiment"] = merged["sentiment"].fillna(0.5)   # neutralno
        merged["sentiment_pos_ratio"] = merged["sentiment_pos_ratio"].fillna(0.5)

        # TF-IDF
        vectorizer, matrix = content_based.load_tfidf()

        # embeddinzi
        embeddings = semantic.load_embeddings() if config.USE_SEMANTIC else None
        if embeddings is not None and len(embeddings) != len(merged):
            print("[recommender] Upozorenje: broj embeddinga != broj recepata; "
                  "iskljucujem semanticku komponentu.")
            embeddings = None

        return cls(merged, vectorizer, matrix, embeddings, weights)

    # ovde izbacujem po principu filteringa recepte koji odmah ne upadaju u korisnicke zelje
    def _hard_filter_mask(self, prefs: Preferences, use_tags: bool, use_include: bool,
                          use_time: bool) -> np.ndarray:
        n = len(self.recipes)
        mask = np.ones(n, dtype=bool)

        ingredients_sets = self.recipes["ingredients_list"]
        tags_sets = self.recipes["tags_list"]

        # izbaceni sastojci mora da se ispostuju
        if prefs.exclude_ingredients:
            ex = [e.lower() for e in prefs.exclude_ingredients]
            def has_excluded(lst):
                joined = " ".join(str(x).lower() for x in lst)
                return any(e in joined for e in ex)
            mask &= ~ingredients_sets.apply(has_excluded).to_numpy()

        # mora da sadrzi ove sastojke
        if use_include and prefs.include_ingredients:
            inc = [i.lower() for i in prefs.include_ingredients]
            def has_all_included(lst):
                joined = " ".join(str(x).lower() for x in lst)
                return all(i in joined for i in inc)
            mask &= ingredients_sets.apply(has_all_included).to_numpy()

        # mora da sadrzi te tagove
        if use_tags and prefs.tags:
            want = [t.lower() for t in prefs.tags]
            def has_all_tags(lst):
                s = {str(x).lower() for x in lst}
                return all(t in s for t in want)
            mask &= tags_sets.apply(has_all_tags).to_numpy()

        # vreme kuvanja
        if use_time and prefs.max_minutes:
            mask &= (self._minutes <= prefs.max_minutes)

        return mask

    # odabir kandidata koji ulaze uu finalno rangiranje
    def _candidate_indices(self, query: str, prefs: Preferences,
                           pool_size: int) -> np.ndarray:
        min_pool = max(pool_size, config.TOP_K_DEFAULT)

        relax_levels = [
            (True,  True,  True),
            (False, True,  True),
            (True,  True,  False),
            (False, True,  False),
            (False, False, True),
            (False, False, False),
        ]
        mask = None
        for use_tags, use_include, use_time in relax_levels:
            mask = self._hard_filter_mask(prefs, use_tags, use_include, use_time)
            if mask.sum() >= min_pool:
                break
        if mask is None or mask.sum() == 0:
            mask = np.ones(len(self.recipes), dtype=bool)

        idx = np.where(mask)[0]

        # rangiranje kandidata TF-IDF
        content_sims = content_based.similarity_for_indices(
            query, idx, self.tfidf_vectorizer, self.tfidf_matrix
        )
        if len(idx) > pool_size:
            top_local = np.argsort(-content_sims)[:pool_size]
            idx = idx[top_local]
            content_sims = content_sims[top_local]

        self._last_content_sims = content_sims
        return idx

    # racunanje score-a kandidata
    def component_scores(self, query: str, candidate_idx: np.ndarray,
                         content_sims: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if content_sims is None:
            content_sims = content_based.similarity_for_indices(
                query, candidate_idx, self.tfidf_vectorizer, self.tfidf_matrix
            )

        # semantika SBERT
        if self.has_semantic:
            q_vec = semantic.embed_query(query)
            sem = semantic.query_similarity(q_vec, self.embeddings, candidate_idx)
            if sem is None:
                sem = np.zeros(len(candidate_idx), dtype=np.float32)
        else:
            sem = np.zeros(len(candidate_idx), dtype=np.float32)
        
        # za svakog se vraca kandidata sentiment, recept, kvalitet, i koliko se slazu kriticari
        return {
            "content": np.asarray(content_sims, dtype=np.float32),
            "semantic": np.asarray(sem, dtype=np.float32),
            "sentiment": self._sentiment[candidate_idx],
            "rating_quality": self._quality[candidate_idx],
            "agreement": self._agreement[candidate_idx],
        }
    
    # finalan score za svaku komponentu 
    # final_score = w_content * content + w_semantic * semantic + w_sentiment * sentiment + w_rating_quality * rating_quality + w_agreement * agreement
    def combine(self, comps: dict[str, np.ndarray],
                weights: dict[str, float] | None = None) -> np.ndarray:
        w = normalize_weights(weights or self.weights, drop_semantic=not self.has_semantic)
        final = np.zeros(len(next(iter(comps.values()))), dtype=np.float32)
        for k in config.COMPONENTS:
            final += w[k] * comps[k]
        return final
     
    # vraca listu najboljih k recepata
    def recommend(
        self,
        query: str,
        prefs: Preferences | None = None,
        top_k: int = config.TOP_K_DEFAULT,
        weights: dict[str, float] | None = None,
        use_query_parsing: bool = True,
    ) -> list[Recommendation]:
        prefs = prefs or Preferences()

        # ucitava preference i parsira ih
        if use_query_parsing:
            prefs = merge_preferences(prefs, parse_query_preferences(query))
        
        # izbor kandidata, tf-idf, i izbacivanje kandidata na osnovu preferenci
        idx = self._candidate_indices(query, prefs, config.CANDIDATE_POOL_SIZE)
        content_sims = getattr(self, "_last_content_sims", None)

        # racunaju se sve komponente - pojedinacne ocene
        comps = self.component_scores(query, idx, content_sims)

        # ocena finalna
        final = self.combine(comps, weights)

        order = np.argsort(-final)[:top_k]
        results: list[Recommendation] = []
        for o in order:
            row_idx = int(idx[o])
            row = self.recipes.iloc[row_idx]
            comp_vals = {k: float(comps[k][o]) for k in config.COMPONENTS}
            results.append(Recommendation(
                recipe_id=int(row["id"]),
                name=str(row["name"]),
                final_score=float(final[o]),
                components=comp_vals,
                minutes=float(row["minutes"]) if not pd.isna(row["minutes"]) else float("nan"),
                n_reviews=int(row["n_reviews"]),
                rating_mean=float(row["rating_mean"]),
                ingredients=list(row["ingredients_list"]),
                tags=list(row["tags_list"]),
                description=str(row["description"]) if "description" in row and not pd.isna(row["description"]) else "",
                steps=_to_list(row["steps_list"]) if "steps_list" in row.index else [],
                explanation=self._explain(row, comp_vals, weights or self.weights),
            ))
        return results
    
    # za prikaz preporuka
    def recommend_df(self, query: str, **kwargs) -> pd.DataFrame:
        recs = self.recommend(query, **kwargs)
        return pd.DataFrame([{
            "recipe_id": r.recipe_id,
            "name": r.name,
            "final_score": round(r.final_score, 4),
            "rating": round(r.rating_mean, 2),
            "n_reviews": r.n_reviews,
            "minutes": r.minutes,
            **{f"c_{k}": round(v, 3) for k, v in r.components.items()},
            "explanation": r.explanation,
        } for r in recs])

    # objasnjenje preporuka
    def _explain(self, row, comp_vals: dict[str, float],
                 weights: dict[str, float]) -> str:
        w = normalize_weights(weights, drop_semantic=not self.has_semantic)
        # doprinos svake komponente finalnom skoru
        contrib = {k: w[k] * comp_vals[k] for k in config.COMPONENTS}
        top = sorted(contrib.items(), key=lambda x: -x[1])[:2]

        labels = {
            "content": "keyword-based match with your query",
            "semantic": "semantic similarity to your query",
            "sentiment": "positive user reviews",
            "rating_quality": "high and reliable rating quality",
            "agreement": "strong agreement among users",
        }
        reasons = [labels[k] for k, _ in top]

        parts = ["Recommended because of: " + " and ".join(reasons) + "."]
        n_rev = int(row["n_reviews"])
        if n_rev > 0:
            parts.append(
                f"Average rating {row['rating_mean']:.2f} based on {n_rev} reviews."
            )
        else:
            parts.append("This recipe does not have user reviews yet.")
        if comp_vals["sentiment"] >= 0.6 and n_rev > 0:
            parts.append("The reviews are mostly positive.")
        return " ".join(parts)
