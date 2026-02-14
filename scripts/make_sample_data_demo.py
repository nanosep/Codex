from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.labeling import label_tp_sl_binary


def _build_demo_df(rows: int, seed: int, preset: tuple[float, float, float, float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")

    regimes = [
        (0.0010, 0.0015),
        (0.0002, 0.0020),
        (-0.0005, 0.0018),
        (0.0012, 0.0014),
        (0.0000, 0.0022),
    ]
    rets = []
    for i in range(rows):
        mu, vol = regimes[(i // 80) % len(regimes)]
        rets.append(np.clip(rng.normal(mu, vol), -0.006, 0.006))
    returns = np.array(rets)

    for i in range(50, rows, 90):
        returns[i] -= 0.0040

    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(rows)
    open_[0] = close[0] * (1.0 + rng.normal(0.0, 0.0008))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.0009, size=rows - 1))

    h_low, h_high, l_low, l_high = preset
    high = np.maximum(open_, close) * (1.0 + rng.uniform(h_low, h_high, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(l_low, l_high, size=rows))
    volume = rng.integers(100_000, 900_000, size=rows)

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


def make_sample_data_demo(output_path: str | Path = "data/sample_daily_demo.csv", rows: int = 600, seed: int = 11) -> pd.DataFrame:
    presets = [
        (0.0020, 0.0100, 0.0003, 0.0035),
        (0.0015, 0.0080, 0.0002, 0.0028),
        (0.0025, 0.0120, 0.0004, 0.0042),
    ]

    chosen = None
    chosen_stats = None
    for preset in presets:
        df = _build_demo_df(rows=rows, seed=seed, preset=preset)
        labeled = label_tp_sl_binary(df, horizon=40, tp_pct=0.01, sl_pct=0.003)
        positives = int(labeled["label"].sum())
        total = len(labeled)
        positive_rate = (positives / total * 100.0) if total else 0.0
        if positives >= 20 and 10.0 <= positive_rate <= 60.0:
            chosen = df
            chosen_stats = (positives, total, positive_rate)
            break

    if chosen is None:
        raise RuntimeError("No deterministic demo preset matched target label band.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(out_path, index=False)

    positives, total, positive_rate = chosen_stats
    print(f"rows: {len(chosen)}")
    print(f"date range: {chosen['date'].iloc[0].date()} -> {chosen['date'].iloc[-1].date()}")
    print(f"labels@h40,tp1%,sl0.3%: positives={positives}, total={total}, positive_rate={positive_rate:.2f}%")
    return chosen


if __name__ == "__main__":
    make_sample_data_demo()
