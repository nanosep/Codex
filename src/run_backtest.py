from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_ohlcv_csv
from src.engine.backtest import export_results, run_backtest
from src.strategies.strategy_001 import apply_strategy_001
from src.viz.plot import plot_backtest_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal rule-based backtest (strategy_001).")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument("--out", required=True, help="Output directory for artifacts.")
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


def print_summary(summary: dict[str, float]) -> None:
    print(f"number of trades: {int(summary['number_of_trades'])}")
    print(f"total return (%): {summary['total_return_pct']:.2f}")
    print(f"win rate (%): {summary['win_rate_pct']:.2f}")
    print(f"max drawdown (%): {summary['max_drawdown_pct']:.2f}")


def main() -> None:
    args = parse_args()
    df = load_ohlcv_csv(args.csv)
    df = apply_date_filter(df, args.start, args.end)
    if df.empty:
        raise ValueError("No data left after applying date filters.")

    strategy_df = apply_strategy_001(df)
    result = run_backtest(strategy_df)

    out_dir = Path(args.out)
    export_results(result, out_dir)
    plot_backtest_chart(strategy_df[["date", "close"]], result.entries, result.exits, out_dir / "chart.png")
    print_summary(result.summary)


if __name__ == "__main__":
    main()
