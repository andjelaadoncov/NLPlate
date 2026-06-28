from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .preprocessing import (
    load_raw_recipes, load_raw_interactions, clean_recipes,
    compute_interaction_stats, _read_or_pickle_path,
)
from .rating_quality import build_quality_table
from .text_utils import NUTRITION_FIELDS

sns.set_theme(style="whitegrid")


def run_eda() -> dict:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stats_out: dict = {}

    raw_recipes = load_raw_recipes()
    interactions = load_raw_interactions()
    stats_out["n_recipes_raw"] = int(len(raw_recipes))
    stats_out["n_interactions_raw"] = int(len(interactions))

    recipes = clean_recipes(raw_recipes)
    stats_out["n_recipes_clean"] = int(len(recipes))

    istats = compute_interaction_stats(interactions)
    quality = build_quality_table(istats)
    stats_out["n_recipes_with_reviews"] = int(len(istats))
    stats_out["global_mean_rating"] = round(float(quality["global_mean"].iloc[0]), 4) if len(quality) else None

    rated = interactions.copy()
    rated["rating"] = pd.to_numeric(rated["rating"], errors="coerce")
    rated = rated[rated["rating"] > 0]
    rating_counts = rated["rating"].value_counts().sort_index()
    stats_out["rating_distribution"] = {int(k): int(v) for k, v in rating_counts.items()}
    stats_out["share_5star"] = round(float((rated["rating"] == 5).mean()), 4)
    stats_out["share_ge4"] = round(float((rated["rating"] >= 4).mean()), 4)

    plt.figure(figsize=(7, 4.2))
    sns.barplot(x=rating_counts.index.astype(int), y=rating_counts.values, color="#4C72B0")
    plt.xlabel("Ocena (1-5)"); plt.ylabel("Broj recenzija")
    plt.title("Distribucija korisnickih ocena")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_rating_distribution.png", dpi=130); plt.close()

    nrev = istats["n_reviews"]
    stats_out["median_reviews_per_recipe"] = float(np.median(nrev)) if len(nrev) else None
    stats_out["mean_reviews_per_recipe"] = round(float(np.mean(nrev)), 3) if len(nrev) else None
    stats_out["max_reviews_per_recipe"] = int(np.max(nrev)) if len(nrev) else None

    plt.figure(figsize=(7, 4.2))
    plt.hist(np.clip(nrev, 0, 50), bins=50, color="#55A868")
    plt.yscale("log")
    plt.xlabel("Broj recenzija po receptu")
    plt.ylabel("Broj recepata (log skala)")
    plt.title("Raspodela broja recenzija po receptu")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_reviews_per_recipe.png", dpi=130); plt.close()

    minutes = pd.to_numeric(recipes["minutes"], errors="coerce").dropna()
    stats_out["median_minutes"] = float(np.median(minutes)) if len(minutes) else None
    plt.figure(figsize=(7, 4.2))
    plt.hist(np.clip(minutes, 0, 180), bins=40, color="#C44E52")
    plt.xlabel("Vreme pripreme")
    plt.ylabel("Broj recepata")
    plt.title("Raspodela vremena pripreme")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_prep_time.png", dpi=130); plt.close()

    ning = pd.to_numeric(recipes["n_ingredients"], errors="coerce").dropna()
    stats_out["median_n_ingredients"] = float(np.median(ning)) if len(ning) else None
    plt.figure(figsize=(7, 4.2))
    plt.hist(np.clip(ning, 0, 30), bins=30, color="#8172B3")
    plt.xlabel("Broj sastojaka")
    plt.ylabel("Broj recepata")
    plt.title("Raspodela broja sastojaka po receptu")
    plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_n_ingredients.png", dpi=130); plt.close()

    tag_counter: Counter = Counter()
    for lst in recipes["tags_list"]:
        tag_counter.update(str(t).lower() for t in lst)
    top_tags = tag_counter.most_common(20)
    stats_out["top_tags"] = [[t, int(c)] for t, c in top_tags]
    if top_tags:
        labels, counts = zip(*top_tags)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(counts), y=list(labels), color="#4C72B0")
        plt.xlabel("Broj recepata"); plt.ylabel("")
        plt.title("20 najcescih tagova")
        plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_top_tags.png", dpi=130); plt.close()

    nutri_cols = [c for c in NUTRITION_FIELDS if c in recipes.columns]
    if nutri_cols:
        nutri = recipes[nutri_cols].apply(pd.to_numeric, errors="coerce")
        
        nutri = nutri.clip(upper=nutri.quantile(0.99), axis=1)
        corr = nutri.corr()
        plt.figure(figsize=(7.5, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
        plt.title("Korelaciona matrica nutritivnih karakteristika")
        plt.tight_layout(); plt.savefig(config.FIGURES_DIR / "eda_nutrition_corr.png", dpi=130); plt.close()
        stats_out["nutrition_corr"] = {a: {b: round(float(corr.loc[a, b]), 2) for b in corr.columns} for a in corr.index}

    out_path = config.REPORTS_DIR / "eda_stats.json"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    print("=== EDA zavrsena ===")
    print(json.dumps(stats_out, indent=2, ensure_ascii=False)[:2000])
    print(f"\nStatistike: {out_path}")
    print(f"Figure: {config.FIGURES_DIR}")
    return stats_out


if __name__ == "__main__":
    run_eda()