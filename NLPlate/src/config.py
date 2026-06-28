# ovde cuvam sve putanje, podrazumevane tezine i parametre modela, baseline tezine iz predloga teme

from __future__ import annotations

from pathlib import Path

# putanje projekta i foldera
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            
PROCESSED_DIR: Path = DATA_DIR / "processed"  
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"


RAW_RECIPES_CSV: Path = RAW_DIR / "RAW_recipes.csv"
RAW_INTERACTIONS_CSV: Path = RAW_DIR / "RAW_interactions.csv"

# preprocessed artefakti
RECIPES_CLEAN_PARQUET: Path = PROCESSED_DIR / "recipes_clean.parquet"
RECIPE_STATS_PARQUET: Path = PROCESSED_DIR / "recipe_stats.parquet"
TFIDF_MATRIX_NPZ: Path = PROCESSED_DIR / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PKL: Path = PROCESSED_DIR / "tfidf_vectorizer.pkl"
SBERT_EMBEDDINGS_NPY: Path = PROCESSED_DIR / "sbert_embeddings.npy"
RECIPE_INDEX_PARQUET: Path = PROCESSED_DIR / "recipe_index.parquet" 


# final_score = 0.40*content + 0.25*semantic + 0.20*sentiment + 0.10*rating_quality + 0.05*agreement
# baseline tezine za finalni skor
BASELINE_WEIGHTS: dict[str, float] = {
    "content": 0.40,
    "semantic": 0.25,
    "sentiment": 0.20,
    "rating_quality": 0.10,
    "agreement": 0.05,
}

COMPONENTS: list[str] = [
    "content",
    "semantic",
    "sentiment",
    "rating_quality",
    "agreement",
]

# ovde se cuvaju tezine koje su podesene na osnovu evalucija tj tuninga-a
TUNED_WEIGHTS_JSON: Path = PROCESSED_DIR / "tuned_weights.json"

# parametri za semanticko pretrazivanje (SBERT)
USE_SEMANTIC: bool = True
SBERT_MODEL_NAME: str = "all-MiniLM-L6-v2"  

# TF-IDF parametri
TFIDF_MAX_FEATURES: int = 50_000
TFIDF_MIN_DF: int = 2
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)

# Bayes-ov prosek ocena koji ublazava ocene recepata sa malim brojem recenzija 
# bayesian_rating = (v / (v + m)) * R + (m / (v + m)) * C
# m je minimalni broj recenzija da bi se koristila ocena recepta:
BAYES_PRIOR_COUNT: int = 20

# filtriranje recepata sa premalo recenzija prilikom evaluacije:
MIN_REVIEWS_FOR_QUALITY: int = 3


SAMPLE_N_RECIPES: int | None = None
RANDOM_SEED: int = 42


TOP_K_DEFAULT: int = 10          # po default-u prikazuje 10 recenzija, ali ima podesavanja
CANDIDATE_POOL_SIZE: int = 300   # koliko kandidata ulazi u finalno rangiranje


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
