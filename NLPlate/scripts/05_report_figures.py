# grafici za seminarski

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.recommender import NLPlateRecommender, load_tuned_weights
from src.evaluation import Evaluator

sns.set_theme(style="whitegrid")


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rec = NLPlateRecommender.from_processed()
    ev = Evaluator(rec)
    ev.prepare()

    # 1) Baseline poredjenje
    base = ev.evaluate_baselines()
    tuned_w = load_tuned_weights()
    base.loc["hybrid_tuned"] = ev.evaluate_weights(tuned_w)

    metrics_to_plot = [c for c in ["precision@10", "recall@10", "ndcg@10"] if c in base.columns]
    ax = base[metrics_to_plot].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Poredjenje modela (prosek preko upita)")
    ax.set_ylabel("vrednost metrike")
    ax.set_xlabel("model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "baseline_comparison.png", dpi=130)
    plt.close()

    # 2) Baseline vs tuned tezine
    summary = pd.DataFrame({
        "baseline_weights": ev.evaluate_weights(config.BASELINE_WEIGHTS),
        "tuned_weights": ev.evaluate_weights(tuned_w),
    })
    ax = summary.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Baseline tezine vs. istrenirane tezine")
    ax.set_ylabel("vrednost metrike")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "tuned_vs_baseline.png", dpi=130)
    plt.close()

    # 3) Izabrane tezine
    wser = pd.Series({k: tuned_w[k] for k in config.COMPONENTS})
    ax = wser.plot(kind="bar", figsize=(8, 4.5), color=sns.color_palette("viridis", len(wser)))
    ax.set_title("Istrenirane tezine komponenti (grid search po NDCG@10)")
    ax.set_ylabel("tezina")
    for i, v in enumerate(wser.values):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "tuned_weights.png", dpi=130)
    plt.close()

    # 4) NDCG@10 najboljih kombinacija (ako postoji rezultat grid search-a)
    best, results = ev.grid_search(step=0.1, primary_metric="ndcg@10", min_weight=0.05)
    top = results.head(15).reset_index(drop=True)
    labels = [
        "/".join(f"{r[f'w_{k}']:.2f}" for k in config.COMPONENTS)
        for _, r in top.iterrows()
    ]
    plt.figure(figsize=(11, 5))
    sns.barplot(x=list(range(len(top))), y=top["ndcg@10"], color="#4C72B0")
    plt.xticks(range(len(top)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("NDCG@10")
    plt.xlabel("kombinacija tezina (content/semantic/sentiment/quality/agreement)")
    plt.title("Top 15 kombinacija tezina po NDCG@10")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "ndcg_by_topweights.png", dpi=130)
    plt.close()

    print(f"Figure snimljene u: {config.FIGURES_DIR}")
    for f in sorted(config.FIGURES_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
