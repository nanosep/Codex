from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.labeling import label_tp_sl_binary


def _build_realistic_df(rows: int, seed: int, preset: tuple[float, float, float, float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=rows, freq="D")

    sigma = np.empty(rows)
    sigma[0] = 0.006
    z = rng.standard_t(df=4, size=rows)
    for t in range(1, rows):
        sigma[t] = 0.0007 + 0.18 * abs(z[t - 1]) * sigma[t - 1] + 0.75 * sigma[t - 1]
        sigma[t] = np.clip(sigma[t], 0.0025, 0.03)

    returns = 0.0001 + sigma * z * 0.12
    returns = np.clip(returns, -0.05, 0.05)

    # Deterministic trend bursts to ensure non-zero positives with strict SL.
    for start in [280, 620, 980, 1360, 1720, 2140]:
        if start + 20 < rows:
            returns[start : start + 15] += 0.0022
            returns[start + 15 : start + 20] -= 0.0013

    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(rows)
    open_[0] = close[0] * (1.0 + rng.normal(0.0, 0.0015))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.0018, size=rows - 1))

    h_low, h_high, l_low, l_high = preset
    high = np.maximum(open_, close) * (1.0 + rng.uniform(h_low, h_high, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(l_low, l_high, size=rows))

    for i in range(120, rows, 170):
        low[i] *= 1.0 - rng.uniform(0.01, 0.03)

    volume = (rng.integers(150_000, 1_200_000, size=rows) * (1.0 + np.clip(sigma * 30.0, 0.0, 4.0))).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": volume,
        }
    )


def make_sample_data_realistic(
    output_path: str | Path = "data/sample_daily_realistic.csv", rows: int = 2600, seed: int = 29
) -> pd.DataFrame:
    presets = [
        (0.0010, 0.0150, 0.0002, 0.0025),
        (0.0010, 0.0180, 0.0003, 0.0030),
        (0.0010, 0.0200, 0.0004, 0.0040),
    ]

    chosen = None
    chosen_stats = None
    for preset in presets:
        df = _build_realistic_df(rows=rows, seed=seed, preset=preset)
        labeled = label_tp_sl_binary(df, horizon=40, tp_pct=0.01, sl_pct=0.003)
        positives = int(labeled["label"].sum())
        total = len(labeled)
        positive_rate = (positives / total * 100.0) if total else 0.0
        if positives >= 5 and 0.5 <= positive_rate <= 20.0:
            chosen = df
            chosen_stats = (positives, total, positive_rate)
            break

    if chosen is None:
        raise RuntimeError("No deterministic realistic preset matched target label band.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(out_path, index=False)

    positives, total, positive_rate = chosen_stats
    print(f"rows: {len(chosen)}")
    print(f"date range: {chosen['date'].iloc[0].date()} -> {chosen['date'].iloc[-1].date()}")
    print(f"labels@h40,tp1%,sl0.3%: positives={positives}, total={total}, positive_rate={positive_rate:.2f}%")
    return chosen


if __name__ == "__main__":
    make_sample_data_realistic()
