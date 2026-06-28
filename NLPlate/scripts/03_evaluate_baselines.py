# skripta koja poziva na evaluaciju

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.recommender import NLPlateRecommender
from src.evaluation import Evaluator

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rec = NLPlateRecommender.from_processed()
    ev = Evaluator(rec)
    ev.prepare()
    table = ev.evaluate_baselines()
    print("\n=== Poredjenje baseline modela (prosek preko upita) ===")
    print(table.round(4).to_string())
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "baseline_comparison.csv"
    table.to_csv(out)
    print(f"\nSnimljeno: {out}")
