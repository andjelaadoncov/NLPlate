# preprocesing deo, ciscenje raw podataka
 
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .text_utils import (
    build_recipe_document,
    parse_nutrition,
    safe_parse_list,
    normalize_text,
)


# ucitavanje 
def load_raw_recipes() -> pd.DataFrame:
    path = config.RAW_RECIPES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Nije pronadjen {path}.\n")
    df = pd.read_csv(path)
    return df


def load_raw_interactions() -> pd.DataFrame:
    path = config.RAW_INTERACTIONS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Nije pronadjen {path}.\n")
    df = pd.read_csv(path)
    return df

# fja za obradu recepata
def clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:

    df = recipes.copy()

    expected = {"name", "id", "minutes", "tags", "nutrition",
                "n_steps", "steps", "description", "ingredients", "n_ingredients"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Ne sadrzi ocekivane kolone: {sorted(missing)}.\n"
            f"Pronadjene kolone: {sorted(df.columns)}"
        )

    # parsiranje listi iz dataseta (one za nutrition, tagove i za korake pripreme)
    df["tags_list"] = df["tags"].apply(safe_parse_list)
    df["ingredients_list"] = df["ingredients"].apply(safe_parse_list)
    df["steps_list"] = df["steps"].apply(safe_parse_list)

    nutri = df["nutrition"].apply(parse_nutrition).apply(pd.Series)
    df = pd.concat([df, nutri], axis=1)

    # numericke vrednosti
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["n_steps"] = pd.to_numeric(df["n_steps"], errors="coerce")
    df["n_ingredients"] = pd.to_numeric(df["n_ingredients"], errors="coerce")

    # normalizacija 
    df["name_norm"] = df["name"].apply(normalize_text)

    # tekstualni dokument (za TF-IDF i SBERT)
    df["document"] = df.apply(
        lambda r: build_recipe_document(
            name=r["name"],
            tags=r["tags_list"],
            ingredients=r["ingredients_list"],
            steps=r["steps_list"],
            description=r.get("description"),
        ),
        axis=1,
    )

    # ovo sluzi za brzu pretragu recepata po sastojcima i tagovima
    df["ingredients_set"] = df["ingredients_list"].apply(
        lambda lst: {normalize_text(x) for x in lst}
    )
    df["tags_set"] = df["tags_list"].apply(
        lambda lst: {normalize_text(x).replace(" ", "-") for x in lst}
    )

    # uklananje recepata koji imaju missing vrednosti u bitnim kolonama
    before = len(df)
    df = df[df["name_norm"].str.len() > 0]
    df = df[df["ingredients_list"].apply(len) > 0]
    df = df[df["document"].str.len() > 0]
    df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[clean_recipes] Uklonjeno {removed} neispravnih ili dupliranih recepata.")

    return df

# fja za racunanje statistike interakcija po receptu, koliko je broj ocena, prosecna ocena, devijacija i koliko je pozitivnih i negativnih ocena
def compute_interaction_stats(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()

    expected = {"user_id", "recipe_id", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Ne sadrzi ocekivane kolone: {sorted(missing)}."
        )

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # ako recenzija nema ocenu izbacuje se iz racunanja proseka
    rated = df[df["rating"] > 0]

    grp = rated.groupby("recipe_id")["rating"]
    stats = pd.DataFrame({
        "n_reviews": grp.count(),
        "rating_mean": grp.mean(),
        "rating_std": grp.std().fillna(0.0),
    })
    stats["n_pos"] = rated[rated["rating"] >= 4].groupby("recipe_id")["rating"].count()
    stats["n_neg"] = rated[rated["rating"] <= 2].groupby("recipe_id")["rating"].count()
    stats[["n_pos", "n_neg"]] = stats[["n_pos", "n_neg"]].fillna(0).astype(int)
    stats = stats.reset_index().rename(columns={"recipe_id": "id"})
    return stats



def run_preprocessing(sample_n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    config.ensure_dirs()
    rng = np.random.default_rng(config.RANDOM_SEED)

    print("[preprocessing] Ucitavanje RAW_recipes.csv ...")
    recipes = load_raw_recipes()
    print(f"[preprocessing] Ucitano recepata: {len(recipes):,}")

    print("[preprocessing] Ucitavanje RAW_interactions.csv ...")
    interactions = load_raw_interactions()
    print(f"[preprocessing] Ucitano interakcija: {len(interactions):,}")

    print("[preprocessing] Ciscenje recepata ...")
    recipes_clean = clean_recipes(recipes)
    
    if sample_n is not None and sample_n < len(recipes_clean):
        idx = rng.choice(len(recipes_clean), size=sample_n, replace=False)
        recipes_clean = recipes_clean.iloc[np.sort(idx)].reset_index(drop=True)
        print(f"[preprocessing] Uzorkovano {sample_n:,} recepata za razvoj.")

    print("[preprocessing] Racunanje statistike interakcija ...")
    stats = compute_interaction_stats(interactions)

    # statistika se racuna za recepte i cuvaju se samo oni koji su u recipes_clean
    stats = stats[stats["id"].isin(set(recipes_clean["id"]))].reset_index(drop=True)

    cols_to_save = [
        "id", "name", "name_norm", "minutes", "n_steps", "n_ingredients",
        "tags_list", "ingredients_list", "steps_list", "description",
        "document", *config_nutrition_cols(),
    ]

    cols_to_save = [c for c in cols_to_save if c in recipes_clean.columns]
    recipes_to_save = recipes_clean[cols_to_save].copy()
    
    # cuvanje u parquet fajl jer je on najbolji za ucitavanje i obradu u pandas-u
    # brze ucitavanje u kasnijim koracima
    for c in ["tags_list", "ingredients_list", "steps_list"]:
        recipes_to_save[c] = recipes_clean[c].apply(lambda x: list(x))

    _safe_to_parquet(recipes_to_save, config.RECIPES_CLEAN_PARQUET)
    _safe_to_parquet(stats, config.RECIPE_STATS_PARQUET)

    print(f"[preprocessing] Recepti i statistika snimljeni u: {config.PROCESSED_DIR}")
    print(f"[preprocessing] Recepata: {len(recipes_to_save):,} | "
          f"Recepata sa recenzijama: {len(stats):,}")

    return recipes_clean, stats


def config_nutrition_cols() -> list[str]:
    from .text_utils import NUTRITION_FIELDS
    return list(NUTRITION_FIELDS)


# fja za snimanje u parquet fajl
def _safe_to_parquet(df: pd.DataFrame, path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pyarrow/fastparquet nisu obavezni
        alt = str(path).replace(".parquet", ".pkl")
        df.to_pickle(alt)
        print(f"[preprocessing] (parquet nedostupan: {exc}) -> snimljeno kao {alt}")


def _read_or_pickle_path(path) -> pd.DataFrame:
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


if __name__ == "__main__":
    run_preprocessing(sample_n=config.SAMPLE_N_RECIPES)
