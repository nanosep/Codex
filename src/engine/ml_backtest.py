from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MLBacktestResult:
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    entries: pd.DataFrame
    exits: pd.DataFrame
    summary: dict


def run_ml_backtest(
    market_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    threshold: float,
) -> MLBacktestResult:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if tp_pct <= 0 or sl_pct <= 0:
        raise ValueError("tp and sl must be > 0")

    mkt = market_df.copy().sort_values("date").reset_index(drop=True)
    mkt["date"] = pd.to_datetime(mkt["date"])
    sig = signals_df.copy().sort_values("date").reset_index(drop=True)
    sig["date"] = pd.to_datetime(sig["date"])

    sig = sig[["date", "y_prob", "signal"]]
    merged = mkt.merge(sig, on="date", how="left")
    merged["signal"] = merged["signal"].fillna(0).astype(int)

    trades: list[dict] = []
    entries: list[dict] = []
    exits: list[dict] = []

    i = 0
    n = len(merged)
    while i < n:
        row = merged.iloc[i]
        if int(row["signal"]) != 1 or i + horizon >= n:
            i += 1
            continue

        entry_date = row["date"]
        entry_price = float(row["close"])
        y_prob_at_entry = float(row["y_prob"]) if pd.notna(row["y_prob"]) else float("nan")
        tp_level = entry_price * (1.0 + tp_pct)
        sl_level = entry_price * (1.0 - sl_pct)

        exit_idx = None
        exit_date = None
        exit_price = None
        exit_reason = None

        for k in range(i + 1, i + horizon + 1):
            bar = merged.iloc[k]
            low_k = float(bar["low"])
            high_k = float(bar["high"])
            sl_hit = low_k <= sl_level
            tp_hit = high_k >= tp_level

            if sl_hit and tp_hit:
                exit_idx = k
                exit_date = bar["date"]
                exit_price = sl_level
                exit_reason = "sl"
                break
            if sl_hit:
                exit_idx = k
                exit_date = bar["date"]
                exit_price = sl_level
                exit_reason = "sl"
                break
            if tp_hit:
                exit_idx = k
                exit_date = bar["date"]
                exit_price = tp_level
                exit_reason = "tp"
                break

        if exit_idx is None:
            exit_idx = i + horizon
            bar = merged.iloc[exit_idx]
            exit_date = bar["date"]
            exit_price = float(bar["close"])
            exit_reason = "time"

        bars_held = int(exit_idx - i)
        return_pct = float((exit_price / entry_price) - 1.0)

        trades.append(
            {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "return_pct": return_pct,
                "bars_held": bars_held,
                "y_prob_at_entry": y_prob_at_entry,
                "threshold": float(threshold),
            }
        )
        entries.append({"date": entry_date, "price": entry_price})
        exits.append({"date": exit_date, "price": exit_price})

        # One position at a time: skip all bars while trade is open.
        i = exit_idx + 1

    trades_df = pd.DataFrame(
        trades,
        columns=[
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "exit_reason",
            "return_pct",
            "bars_held",
            "y_prob_at_entry",
            "threshold",
        ],
    )
    entries_df = pd.DataFrame(entries, columns=["date", "price"])
    exits_df = pd.DataFrame(exits, columns=["date", "price"])

    equity_df = _build_equity(merged, trades_df)
    summary = _build_summary(merged, trades_df, equity_df, horizon, tp_pct, sl_pct, threshold)
    signals_out = sig[["date", "y_prob", "signal"]].copy()
    return MLBacktestResult(
        signals=signals_out,
        trades=trades_df,
        equity=equity_df,
        entries=entries_df,
        exits=exits_df,
        summary=summary,
    )


def _build_equity(merged: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    out = merged[["date"]].copy()
    out["equity"] = 1.0
    if trades_df.empty:
        return out

    returns_by_date: dict[pd.Timestamp, float] = {}
    for _, t in trades_df.iterrows():
        d = pd.to_datetime(t["exit_date"])
        returns_by_date[d] = returns_by_date.get(d, 0.0) + float(t["return_pct"])

    eq = 1.0
    values = []
    for d in out["date"]:
        if d in returns_by_date:
            eq *= 1.0 + returns_by_date[d]
        values.append(eq)
    out["equity"] = values
    return out


def _build_summary(
    merged: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    threshold: float,
) -> dict:
    n_trades = int(len(trades_df))
    total_return = float((equity_df["equity"].iloc[-1] - 1.0) * 100.0) if len(equity_df) else 0.0
    win_rate = float((trades_df["return_pct"] > 0).mean() * 100.0) if n_trades else 0.0
    avg_return = float(trades_df["return_pct"].mean() * 100.0) if n_trades else 0.0

    if len(equity_df):
        peak = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] / peak) - 1.0
        max_dd = float(abs(drawdown.min()) * 100.0)
    else:
        max_dd = 0.0

    exposure = float(trades_df["bars_held"].sum() / len(merged) * 100.0) if len(merged) else 0.0
    return {
        "trades_count": n_trades,
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "average_return": avg_return,
        "exposure": exposure,
        "params": {
            "horizon": int(horizon),
            "tp": float(tp_pct),
            "sl": float(sl_pct),
            "threshold": float(threshold),
            "ambiguity_rule": "If both TP and SL are hit in the same bar, SL is assumed first (conservative).",
        },
    }


def export_ml_backtest(result: MLBacktestResult, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result.signals.to_csv(out_path / "ml_signals.csv", index=False)
    result.trades.to_csv(out_path / "ml_trades.csv", index=False)
    result.equity.to_csv(out_path / "ml_equity.csv", index=False)
    with (out_path / "ml_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2)
