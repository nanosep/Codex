from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_backtest_chart(
    price_df: pd.DataFrame, entries_df: pd.DataFrame, exits_df: pd.DataFrame, out_path: str | Path
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_df["date"], price_df["close"], label="Close", color="steelblue", linewidth=1.5)

    if not entries_df.empty:
        ax.scatter(entries_df["date"], entries_df["price"], marker="^", color="green", label="Entry", s=60)
    if not exits_df.empty:
        ax.scatter(exits_df["date"], exits_df["price"], marker="v", color="red", label="Exit", s=60)

    ax.set_title("Backtest Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
