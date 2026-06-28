# izvrsava build_indexes.py skriptu kao 2. korak
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build_indexes import build_all

if __name__ == "__main__":
    build_all()
