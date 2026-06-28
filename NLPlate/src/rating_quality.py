# racuna se ovde koliko je pouzdana ocena recepta i takodje koliko se korisnici slazu oko ocene recepta

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# ako je recept manje ocenjivan smanjuje mu se kredibilitet ocene, ako je ocenjivan vise puta, vise se veruje njegovoj stvarnoj oceni
def bayesian_rating(rating_mean: pd.Series, n_reviews: pd.Series, global_mean: float, prior_count: int = config.BAYES_PRIOR_COUNT,) -> pd.Series:
    n = n_reviews.fillna(0).astype(float)
    r = rating_mean.fillna(global_mean).astype(float)
    m = float(prior_count)
    br = (n / (n + m)) * r + (m / (n + m)) * global_mean
    return br

# fja koja meri koliko se korisnici slazu oko ocene recepta
def agreement_score(rating_std: pd.Series, max_rating: float = 5.0) -> pd.Series:
    std = rating_std.fillna(0.0).astype(float)
    std_max = max_rating / 2.0  
    norm = np.minimum(std / std_max, 1.0)
    return 1.0 - norm # ako je odstupanje veliko, score je mali, ako je odstupanje malo, score je veliki

#  glavna fja koja se koristi za preporuke i rangiranje recepata
def build_quality_table(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    if df.empty:
        df["rating_quality"] = []
        df["agreement"] = []
        return df

    total_reviews = df["n_reviews"].sum()
    if total_reviews > 0:
        global_mean = float((df["rating_mean"] * df["n_reviews"]).sum() / total_reviews)
    else:
        global_mean = 4.0 

    br = bayesian_rating(df["rating_mean"], df["n_reviews"], global_mean)
    
    df["rating_quality"] = (br / 5.0).clip(0.0, 1.0)
    df["agreement"] = agreement_score(df["rating_std"]).clip(0.0, 1.0)
    df["global_mean"] = global_mean
    return df
