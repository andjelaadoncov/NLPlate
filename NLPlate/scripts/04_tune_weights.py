# skripta koja poziva da se izvrsi tune_weights deo kako bi se dobili koeficijenti za score na osnovu metrika

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.recommender import NLPlateRecommender
from src.evaluation import Evaluator

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--metric", type=str, default="ndcg@10")
    ap.add_argument("--min-weight", type=float, default=0.05)
    args = ap.parse_args()

    pd.set_option("display.width", 220)
    rec = NLPlateRecommender.from_processed()
    ev = Evaluator(rec)
    ev.prepare()

    print(f"Pretraga tezina (korak={args.step}, metrika={args.metric}), min_weight={args.min_weight}) ...")
    best, results = ev.grid_search(step=args.step, primary_metric=args.metric, min_weight=args.min_weight)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "weight_search_results.csv", index=False)

    best_score = float(results.iloc[0][args.metric])
    print("\n=== NAJBOLJE TEZINE (po " + args.metric + ") ===")
    for k in config.COMPONENTS:
        print(f"  {k:16s}: {best[k]:.2f}")
    print(f"  -> {args.metric} = {best_score:.4f}")

    base_metrics = ev.evaluate_weights(config.BASELINE_WEIGHTS)
    tuned_metrics = ev.evaluate_weights(best)
    summary = pd.DataFrame({
        "baseline_weights": base_metrics,
        "tuned_weights": tuned_metrics,
    })
    summary.to_csv(config.REPORTS_DIR / "tuning_summary.csv")
    print("\n=== Baseline tezine vs. istrenirane tezine ===")
    print(summary.round(4).to_string())

    ev.save_tuned_weights(best, args.metric, best_score)
    print("\nGotovo. Sistem ce od sada koristiti istrenirane tezine.")
