# eda analiza 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda import run_eda

if __name__ == "__main__":
    run_eda()