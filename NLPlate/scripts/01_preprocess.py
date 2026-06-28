# skripta koja poziva na ciscenje podataka, pokrece preprocessing

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.preprocessing import run_preprocessing

if __name__ == "__main__":
    run_preprocessing(sample_n=config.SAMPLE_N_RECIPES)
