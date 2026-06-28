# streamlit run app/streamlit_app.py


from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.preferences import Preferences
from src.recommender import NLPlateRecommender, load_tuned_weights, normalize_weights

st.set_page_config(page_title="NLPlate", page_icon="🍽️", layout="wide")

COMMON_TAGS = [
    "main-dish", "desserts", "breakfast", "brunch", "side-dishes", "appetizers",
    "snacks", "soups-stews", "salads", "vegetarian", "vegan", "healthy",
    "gluten-free", "low-carb", "low-fat", "30-minutes-or-less", "easy",
]


@st.cache_resource(show_spinner="Loading model NLPlate ...")
def get_recommender():
    """Ucitava preporucivac jednom i kesira ga kroz sesije."""
    return NLPlateRecommender.from_processed()


def artifacts_exist() -> bool:
    rec = config.RECIPES_CLEAN_PARQUET
    rec_pkl = Path(str(rec).replace(".parquet", ".pkl"))
    return rec.exists() or rec_pkl.exists()


def parse_csv_field(text: str) -> list[str]:
    return [t.strip().lower() for t in text.split(",") if t.strip()]


st.title("🍽️ NLPlate — Hybrid Recipe Recommendation System")
st.caption(
    "Describe the meal you're looking for. The system combines recipe content, "
    "semantic similarity, review sentiment, and rating quality."
)

if not artifacts_exist():
    st.error(
        "Processed data not found (data/processed/).\n\n"
        "Run the pipeline before using the application:\n"
        "1. Place RAW_recipes.csv and RAW_interactions.csv into data/raw/\n"
        "2. `python scripts/01_preprocess.py`\n"
        "3. `python scripts/02_build_indexes.py`\n\n"
    )
    st.stop()

rec = get_recommender()

# sidebar
with st.sidebar:
    st.header("⚙️ Preferences & Settings")

    include_txt = st.text_input("Preferred ingredients",
                                placeholder="e.g. chicken, rice, tomato")
    exclude_txt = st.text_input("Ingredients to exclude",
                                placeholder="e.g. mushrooms, pork, peanuts")
    max_minutes = st.slider("Maximum cooking time (minutes)", 0, 240, 0, step=5,
                            help="0 = no limit")
    tags = st.multiselect("Categories / Tags", COMMON_TAGS, default=[])

    st.divider()
    top_k = st.slider("Number of recommendations", 3, 20, config.TOP_K_DEFAULT)

    st.divider()
    st.subheader("Component Weights")
    weight_mode = st.radio(
        "Weight configuration",
        [
            "Tuned (Recommended)",
            "Baseline",
            "Manual"
        ],
        help="Tuned weights were obtained using grid search optimization based on NDCG@10.",
    )
    if weight_mode == "Manual":
        w_content = st.slider("content", 0.0, 1.0, 0.40, 0.05)
        w_semantic = st.slider("semantic", 0.0, 1.0, 0.25, 0.05)
        w_sentiment = st.slider("sentiment", 0.0, 1.0, 0.20, 0.05)
        w_quality = st.slider("rating_quality", 0.0, 1.0, 0.10, 0.05)
        w_agree = st.slider("agreement", 0.0, 1.0, 0.05, 0.05)
        weights = {"content": w_content, "semantic": w_semantic,
                   "sentiment": w_sentiment, "rating_quality": w_quality,
                   "agreement": w_agree}
    elif weight_mode == "Baseline":
        weights = dict(config.BASELINE_WEIGHTS)
    else:
        weights = load_tuned_weights()

    norm_w = normalize_weights(weights, drop_semantic=not rec.has_semantic)
    st.caption("Normalized weights:")
    st.json({k: round(v, 3) for k, v in norm_w.items()})
    if not rec.has_semantic:
        st.info(
           "The semantic (SBERT) component is not available. "
           "Its weight has been redistributed across the remaining components."
        )


query = st.text_input(
    "🔎 Describe your ideal meal",
    value="I want a quick and healthy chicken dinner with vegetables",
)

go = st.button("Recommend Recipes", type="primary")

if go and query.strip():
    prefs = Preferences(
        include_ingredients=parse_csv_field(include_txt),
        exclude_ingredients=parse_csv_field(exclude_txt),
        max_minutes=max_minutes if max_minutes > 0 else None,
        tags=tags,
    )
    with st.spinner("Finding the best recipes..."):
        recs = rec.recommend(query, prefs=prefs, top_k=top_k, weights=weights)

    if not recs:
        st.warning(
            "No recipes match your current preferences. "
            "Try relaxing some of the filters."
        )
    else:
        st.success(f"Found {len(recs)} recommendations.")
        for i, r in enumerate(recs, 1):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {i}. {r.name.title()}")
                    st.write(r.explanation)
                    mins = "—" if pd.isna(r.minutes) else f"{int(r.minutes)} min"
                    st.caption(
                        f"⏱️ {mins}  |  ⭐ {r.rating_mean:.2f} "
                        f"({r.n_reviews} reviews)"
                    )
                    if r.ingredients:
                        st.markdown("**Ingredients:** " + ", ".join(r.ingredients[:15]))
                    if r.tags:
                        st.markdown("**Tags:** " + ", ".join(
                            t for t in r.tags
                            if t in COMMON_TAGS or "-" in t)[:200])
                    if r.description:
                        st.markdown("**Description:**")
                        st.write(r.description)

                    if r.steps:
                        with st.expander("Preparation steps"):
                            for step_no, step in enumerate(r.steps, 1):
                                st.markdown(f"**Step {step_no}.** {step}")
                with c2:
                    st.metric("Final Score", f"{r.final_score:.3f}")
                    st.caption("Component contributions:")
                    comp_df = pd.DataFrame({
                        "Component": list(r.components.keys()),
                        "Score": [round(v, 3) for v in r.components.values()],
                    })
                    st.bar_chart(comp_df.set_index("Component"))

elif go:
    st.warning("Please enter a search query.")

st.divider()
st.caption(
    "NLPlate · Hybrid Recipe Recommendation System"
)
