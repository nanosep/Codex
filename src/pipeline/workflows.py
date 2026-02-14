from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.engine.backtest import export_results, run_backtest
from src.engine.labeling import OUTPUT_COLUMNS, label_tp_sl_binary
from src.engine.ml_backtest import export_ml_backtest, run_ml_backtest
from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled
from src.run_train_model import train_and_export
from src.strategies.strategy_001 import apply_strategy_001
from src.viz.plot import plot_backtest_chart


def run_backtest_from_df(df: pd.DataFrame, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    strategy_df = apply_strategy_001(df)
    result = run_backtest(strategy_df)
    export_results(result, out_path)
    chart_path = out_path / "chart.png"
    plot_backtest_chart(strategy_df[["date", "close"]], result.entries, result.exits, chart_path)
    return result.trades, result.equity, str(chart_path)


def label_from_df(df: pd.DataFrame, horizon: int, tp: float, sl: float, out_dir: str | Path) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    labeled = label_tp_sl_binary(df, horizon=horizon, tp_pct=tp, sl_pct=sl)
    labeled = labeled[OUTPUT_COLUMNS]
    labeled_path = out_path / "labeled.csv"
    labeled.to_csv(labeled_path, index=False)
    return str(labeled_path)


def train_from_labeled_path(
    labeled_csv: str | Path, out_dir: str | Path, test_size: float = 0.2, threshold: float = 0.5
) -> tuple[dict, str, str]:
    summary = train_and_export(labeled_csv=labeled_csv, out_dir=out_dir, test_size=test_size, threshold=threshold)
    metrics_path = Path(summary["metrics_path"])
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics, summary["predictions_path"], summary["model_path"]


def run_ml_strategy_from_df(
    df: pd.DataFrame,
    model,
    out_dir: str | Path,
    horizon: int,
    tp: float,
    sl: float,
    threshold: float,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    market_df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    labeled = label_tp_sl_binary(market_df, horizon=horizon, tp_pct=tp, sl_pct=sl)
    features_df = build_features_from_labeled(labeled)
    if features_df.empty:
        raise ValueError("No feature rows available after NaN drop.")

    y_prob = model.predict_proba(features_df[FEATURE_COLUMNS])[:, 1]
    signals_df = pd.DataFrame(
        {
            "date": pd.to_datetime(features_df["date"]),
            "y_prob": y_prob,
            "signal": (y_prob >= threshold).astype(int),
        }
    )
    result = run_ml_backtest(
        market_df=market_df,
        signals_df=signals_df,
        horizon=horizon,
        tp_pct=tp,
        sl_pct=sl,
        threshold=threshold,
    )
    export_ml_backtest(result, out_path)
    chart_path = out_path / "ml_chart.png"
    plot_backtest_chart(
        price_df=market_df[["date", "close"]],
        entries_df=result.entries,
        exits_df=result.exits,
        out_path=chart_path,
    )

    return {
        "summary": result.summary,
        "signals_path": str(out_path / "ml_signals.csv"),
        "trades_path": str(out_path / "ml_trades.csv"),
        "equity_path": str(out_path / "ml_equity.csv"),
        "chart_path": str(chart_path),
        "summary_path": str(out_path / "ml_summary.json"),
    }
