# pokrecem sve skripte odjednom

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("Preprocesiranje", "scripts/01_preprocess.py"),
    ("Izgradnja indeksa", "scripts/02_build_indexes.py"),
    ("Evaluacija baseline modela", "scripts/03_evaluate_baselines.py"),
    ("Odredjivanje tezina", "scripts/04_tune_weights.py"),
]

if __name__ == "__main__":
    for name, script in STEPS:
        print("\n" + "#" * 80)
        print(f"# KORAK: {name}  ({script})")
        print("#" * 80)
        r = subprocess.run([PY, str(ROOT / script)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\n[run_all] Korak '{name}' nije uspeo (kod {r.returncode}). Prekidam.")
            sys.exit(r.returncode)
    print("\n[run_all] Ceo pipeline je uspesno zavrsen.")
