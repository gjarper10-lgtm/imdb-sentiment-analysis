"""Run EDA, all four models, then the comparison.

    python run_all.py                       # full experiment
    python run_all.py --subset 2000 --epochs 1   # end-to-end smoke test
"""
import sys

from src import compare, eda, run
from src.config import MODELS

if __name__ == "__main__":
    extra = sys.argv[1:]
    if "--subset" not in extra:
        eda.main()
    for name in MODELS:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        run.main(["--model", name, *extra])
    compare.main()
