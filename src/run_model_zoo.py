from __future__ import annotations

import argparse
import sys

from src.ml.model_zoo import run_model_zoo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 4 model zoo selection.")
    parser.add_argument("--labeled", required=True, help="Path to Step 2 labeled CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--n_splits", type=int, default=5, help="TimeSeriesSplit folds, default=5.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Chronological test size, default=0.2.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold, default=0.5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_model_zoo(
            labeled_csv=args.labeled,
            out_dir=args.out,
            n_splits=args.n_splits,
            test_size=args.test_size,
            threshold=args.threshold,
        )
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    print(f"winner: {result.winner_name}")
    print(f"test_roc_auc: {result.metrics['test_roc_auc']}")
    print(f"test_balanced_acc: {result.metrics['test_balanced_acc']:.6f}")


if __name__ == "__main__":
    main()
