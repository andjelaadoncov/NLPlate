# fajl koji ide posle preprocesiranja, pravi fajlove tfidf_vectorizer.pkl, tfidf_matrix.npz, sbert_embeddings.npy, recipe_sentiment.parquet

from __future__ import annotations

import pandas as pd

from . import config, content_based, semantic, sentiment as sentiment_mod
from .preprocessing import (
    _read_or_pickle_path,
    load_raw_interactions,
    _safe_to_parquet,
)


def build_all() -> None:
    config.ensure_dirs()

    # ucitava ociscene recepte
    recipes = _read_or_pickle_path(config.RECIPES_CLEAN_PARQUET)
    print(f"[build_indexes] Recepata: {len(recipes):,}")

    # TF-IDF
    print("[build_indexes] Gradim TF-IDF ...")
    vectorizer, matrix = content_based.build_tfidf(recipes["document"])
    content_based.save_tfidf(vectorizer, matrix)
    print(f"[build_indexes] TF-IDF matrica: {matrix.shape} (recnik: {len(vectorizer.vocabulary_):,})")

    # Sentence-BERT 
    if semantic.is_available():
        print(f"[build_indexes] Racunam SBERT embeddinge (model: {config.SBERT_MODEL_NAME}) ...")
        emb = semantic.build_embeddings(recipes["document"])
        semantic.save_embeddings(emb)
        print(f"[build_indexes] Embeddinzi: {emb.shape}")
    else:
        print("[build_indexes] sentence-transformers nije dostupan ili USE_SEMANTIC=False "
              "-> preskacem semanticku komponentu.")

    # sentiment po receptu (iz svih recenzija)
    print("[build_indexes] Racunam sentiment recenzija ...")
    interactions = load_raw_interactions()
    sent = sentiment_mod.compute_recipe_sentiment(interactions)
    # zadrzi samo recepte koji postoje u ociscenom skupu
    sent = sent[sent["id"].isin(set(recipes["id"]))].reset_index(drop=True)
    _safe_to_parquet(sent, config.PROCESSED_DIR / "recipe_sentiment.parquet")
    print(f"[build_indexes] Sentiment izracunat za {len(sent):,} recepata "
          f"(backend: {sentiment_mod.backend_name()}).")

    print("[build_indexes] Gotovo. Svi artefakti su u data/processed/.")


if __name__ == "__main__":
    build_all()
