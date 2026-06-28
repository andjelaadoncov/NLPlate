# content-based komponenta (tf-idf)
# omogucava preporucivanje recepata na osnovu slicnosti izmedju unosa korisnika i sadrzaja recepta

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

# izgradnja tf-idf matrice i vektora
def build_tfidf(documents: pd.Series) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words="english", # nema stop reci
        sublinear_tf=True, # smanjuje se efekat cestih reci
    )
    matrix = vectorizer.fit_transform(documents.fillna("").astype(str))
    return vectorizer, matrix


def save_tfidf(vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix) -> None:
    config.ensure_dirs()
    with open(config.TFIDF_VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(config.TFIDF_MATRIX_NPZ, matrix)


def load_tfidf() -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    with open(config.TFIDF_VECTORIZER_PKL, "rb") as f:
        vectorizer = pickle.load(f)
    matrix = sparse.load_npz(config.TFIDF_MATRIX_NPZ)
    return vectorizer, matrix


# korisnicki unos se transformise u vektor i racuna se koliko je on slica svakom receptu iz vektorizovane matrice
def query_similarity( query: str, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix,) -> np.ndarray:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).ravel()
    return sims


def similarity_for_indices(query: str, indices: np.ndarray, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix,) -> np.ndarray:
    q_vec = vectorizer.transform([query])
    sub = matrix[indices]
    sims = cosine_similarity(q_vec, sub).ravel()
    return sims
