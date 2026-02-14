from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.loader import load_ohlcv_csv, validate_ohlcv_dataframe
from src.engine.labeling import label_tp_sl_binary
from src.engine.ml_backtest import export_ml_backtest, run_ml_backtest
from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled
from src.viz.plot import plot_backtest_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 5 ML signal backtest.")
    parser.add_argument("--csv", required=False, help="OHLCV CSV path.")
    parser.add_argument("--labeled", required=False, help="Labeled CSV path (Step 2 format).")
    parser.add_argument("--model", required=True, help="Path to best model joblib.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--horizon", type=int, required=True, help="Lookahead horizon.")
    parser.add_argument("--tp", type=float, required=True, help="TP percent, e.g. 0.01.")
    parser.add_argument("--sl", type=float, required=True, help="SL percent, e.g. 0.003.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Signal threshold, default=0.5.")
    return parser.parse_args()


def _load_labeled_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Labeled CSV missing required columns: {missing}")
    out = validate_ohlcv_dataframe(df)
    out["label"] = pd.to_numeric(df["label"], errors="coerce")
    if out["label"].isna().any():
        raise ValueError("Invalid label values in labeled CSV.")
    return out


def _make_signals(model, features_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    x = features_df[FEATURE_COLUMNS]
    y_prob = model.predict_proba(x)[:, 1]
    signal = (y_prob >= threshold).astype(int)
    return pd.DataFrame({"date": pd.to_datetime(features_df["date"]), "y_prob": y_prob, "signal": signal})


def print_summary(summary: dict) -> None:
    print(f"trades count: {summary['trades_count']}")
    print(f"total return: {summary['total_return']:.2f}")
    print(f"win rate: {summary['win_rate']:.2f}")
    print(f"max drawdown: {summary['max_drawdown']:.2f}")
    print(f"average return: {summary['average_return']:.2f}")
    print(f"exposure: {summary['exposure']:.2f}")


def main() -> None:
    args = parse_args()
    if bool(args.csv) == bool(args.labeled):
        raise SystemExit("Provide exactly one of --csv or --labeled.")

    try:
        from joblib import load
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    model = load(args.model)

    if args.csv:
        market_df = load_ohlcv_csv(args.csv)
        labeled_for_features = label_tp_sl_binary(market_df, horizon=args.horizon, tp_pct=args.tp, sl_pct=args.sl)
    else:
        labeled_for_features = _load_labeled_csv(args.labeled)
        market_df = labeled_for_features[["date", "open", "high", "low", "close", "volume"]].copy()

    features_df = build_features_from_labeled(labeled_for_features)
    if features_df.empty:
        raise ValueError("No feature rows available after NaN drop.")

    signals_df = _make_signals(model, features_df, threshold=args.threshold)
    result = run_ml_backtest(
        market_df=market_df,
        signals_df=signals_df,
        horizon=args.horizon,
        tp_pct=args.tp,
        sl_pct=args.sl,
        threshold=args.threshold,
    )

    out_path = Path(args.out)
    export_ml_backtest(result, out_path)
    plot_backtest_chart(
        price_df=market_df[["date", "close"]],
        entries_df=result.entries,
        exits_df=result.exits,
        out_path=out_path / "ml_chart.png",
    )
    print_summary(result.summary)


if __name__ == "__main__":
    main()
