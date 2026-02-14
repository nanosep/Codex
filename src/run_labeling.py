from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_ohlcv_csv
from src.engine.labeling import OUTPUT_COLUMNS, label_tp_sl_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TP/SL binary labeling on daily OHLCV data.")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument("--out", required=True, help="Output directory for artifacts.")
    parser.add_argument("--horizon", type=int, required=True, help="Lookahead horizon in bars.")
    parser.add_argument("--tp", type=float, required=True, help="Take-profit percent, e.g. 0.01.")
    parser.add_argument("--sl", type=float, required=True, help="Stop-loss percent, e.g. 0.003.")
    parser.add_argument("--start", required=False, help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end", required=False, help="Optional end date YYYY-MM-DD.")
    return parser.parse_args()


def apply_date_filter(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    if start:
        start_dt = pd.to_datetime(start)
        out = out[out["date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end)
        out = out[out["date"] <= end_dt]
    return out.reset_index(drop=True)


def print_summary(labeled_df: pd.DataFrame) -> None:
    total_rows = int(len(labeled_df))
    positives = int(labeled_df["label"].sum()) if total_rows > 0 else 0
    positive_rate = (positives / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"total labeled rows: {total_rows}")
    print(f"number of positives: {positives}")
    print(f"positive rate (%): {positive_rate:.2f}")


def main() -> None:
    args = parse_args()

    df = load_ohlcv_csv(args.csv)
    df = apply_date_filter(df, args.start, args.end)
    if df.empty:
        raise ValueError("No data left after applying date filters.")

    labeled = label_tp_sl_binary(df, horizon=args.horizon, tp_pct=args.tp, sl_pct=args.sl)
    labeled = labeled[OUTPUT_COLUMNS]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_dir / "labeled.csv", index=False)

    print_summary(labeled)


if __name__ == "__main__":
    main()
