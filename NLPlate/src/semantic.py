# tf-idf komponenta je merila preklapanje reci, a pomocu sentence-transformers merim slicnost znacenja recenica 

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_ST_AVAILABLE = False
_MODEL = None
try:
    from sentence_transformers import SentenceTransformer  
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False


def is_available() -> bool:
    return _ST_AVAILABLE and config.USE_SEMANTIC


# ucitavanje modela
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.SBERT_MODEL_NAME)
    return _MODEL

# izgradnja embeddinga za sve recepte
def build_embeddings(documents: pd.Series, batch_size: int = 256) -> np.ndarray:
    if not is_available():
        raise RuntimeError(
            "Sentence-transformers nije dostupan."
        )
    
    model = _get_model()
    docs = documents.fillna("").astype(str).tolist()
    emb = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def save_embeddings(emb: np.ndarray) -> None:
    config.ensure_dirs()
    np.save(config.SBERT_EMBEDDINGS_NPY, emb)


def load_embeddings() -> np.ndarray | None:
    path = config.SBERT_EMBEDDINGS_NPY
    if not path.exists():
        return None
    return np.load(path)

# pretvaranje korisnickog unosa u embedding vektor
def embed_query(query: str) -> np.ndarray | None:
    if not is_available():
        return None
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0].astype(np.float32)

#  slicnosti izmedju unosa i recepata pomocu kreiranih embeddinga
def query_similarity(query_vec: np.ndarray | None, embeddings: np.ndarray | None, indices: np.ndarray | None = None,) -> np.ndarray | None:
    if query_vec is None or embeddings is None:
        return None
    mat = embeddings if indices is None else embeddings[indices]
    sims = mat @ query_vec      # mnozenje matrica i vektora
    sims = (sims + 1.0) / 2.0  #  dobija se rezultat u opsegu od -1 do 1, skaliram ga na opseg od 0 do 1
    return np.clip(sims, 0.0, 1.0)
