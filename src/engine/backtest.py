from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    entries: pd.DataFrame
    exits: pd.DataFrame
    summary: dict[str, float]


def run_backtest(df: pd.DataFrame) -> BacktestResult:
    in_position = False
    entry_idx = -1
    entry_date = None
    entry_price = 0.0
    equity_value = 1.0

    trades: list[dict] = []
    entries: list[dict] = []
    exits: list[dict] = []
    equity_rows: list[dict] = []

    for i, row in df.iterrows():
        date = row["date"]
        close = float(row["close"])
        if not in_position and bool(row["entry_signal"]):
            in_position = True
            entry_idx = i
            entry_date = date
            entry_price = close
            entries.append({"date": date, "price": close})
        elif in_position and bool(row["exit_signal"]):
            exit_price = close
            return_pct = (exit_price / entry_price) - 1.0
            bars_held = i - entry_idx
            trades.append(
                {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": date,
                    "exit_price": exit_price,
                    "return_pct": return_pct,
                    "bars_held": bars_held,
                }
            )
            exits.append({"date": date, "price": close})
            equity_value *= 1.0 + return_pct
            in_position = False

        equity_rows.append({"date": date, "equity": equity_value})

    trades_df = pd.DataFrame(
        trades,
        columns=["entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "bars_held"],
    )
    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"])
    entries_df = pd.DataFrame(entries, columns=["date", "price"])
    exits_df = pd.DataFrame(exits, columns=["date", "price"])
    summary = compute_summary(trades_df, equity_df)
    return BacktestResult(trades=trades_df, equity=equity_df, entries=entries_df, exits=exits_df, summary=summary)


def compute_summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict[str, float]:
    n_trades = int(len(trades_df))
    total_return_pct = float((equity_df["equity"].iloc[-1] - 1.0) * 100.0) if len(equity_df) else 0.0
    if n_trades == 0:
        win_rate_pct = 0.0
    else:
        win_rate_pct = float((trades_df["return_pct"] > 0).mean() * 100.0)

    if len(equity_df) == 0:
        max_drawdown_pct = 0.0
    else:
        peak = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] / peak) - 1.0
        max_drawdown_pct = float(abs(drawdown.min()) * 100.0)

    return {
        "number_of_trades": float(n_trades),
        "total_return_pct": total_return_pct,
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }


def export_results(result: BacktestResult, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out_path / "trades.csv", index=False)
    result.equity.to_csv(out_path / "equity.csv", index=False)
