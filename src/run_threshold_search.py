from __future__ import annotations

import argparse
import sys

from src.ml.threshold_search import run_threshold_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 6 threshold search and freeze policy.")
    parser.add_argument("--csv", required=True, help="OHLCV CSV path.")
    parser.add_argument("--model", required=True, help="Best model joblib path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--horizon", type=int, required=True, help="Lookahead horizon.")
    parser.add_argument("--tp", type=float, required=True, help="Take-profit percent.")
    parser.add_argument("--sl", type=float, required=True, help="Stop-loss percent.")
    parser.add_argument("--n_splits", type=int, default=5, help="TimeSeriesSplit folds, default=5.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Chronological test size, default=0.2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_threshold_search(
            csv_path=args.csv,
            model_path=args.model,
            out_dir=args.out,
            horizon=args.horizon,
            tp=args.tp,
            sl=args.sl,
            n_splits=args.n_splits,
            test_size=args.test_size,
        )
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    print(f"best threshold: {result['best_threshold']}")
    print(f"search file: {result['search_path']}")
    print(f"test summary: {result['test_summary_path']}")


if __name__ == "__main__":
    main()
