from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_daily_csv(output_path: str | Path = "data/sample_daily.csv", rows: int = 220, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")

    # Mild upward drift with clipped downside so 1% TP can occur while 0.3% SL is not constantly touched.
    returns = rng.normal(loc=0.0012, scale=0.0007, size=rows)
    returns = np.clip(returns, -0.0006, 0.0030)
    # Deterministic pullback events to avoid all-positive labels and keep training feasible.
    for idx in (55, 120, 175):
        if idx < rows:
            returns[idx] = -0.0040
    close = 100.0 * np.cumprod(1.0 + returns)

    open_ = np.empty(rows)
    open_[0] = close[0] * (1 + rng.normal(0, 0.0002))
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.0002, size=rows - 1))

    high = np.maximum(open_, close) * (1 + rng.uniform(0.0010, 0.0060, size=rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.0010, size=rows))
    volume = rng.integers(120_000, 950_000, size=rows)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": volume,
        }
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    make_sample_daily_csv()
