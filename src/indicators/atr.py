from __future__ import annotations

import numpy as np
import pandas as pd


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 21) -> pd.Series:
    h = high.astype(float).to_numpy()
    l = low.astype(float).to_numpy()
    c = close.astype(float).to_numpy()
    n = len(c)
    tr = np.full(n, np.nan, dtype=float)
    atr = np.full(n, np.nan, dtype=float)

    if n == 0:
        return pd.Series(atr, index=close.index)

    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    if n <= period:
        return pd.Series(atr, index=close.index)

    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period

    return pd.Series(atr, index=close.index)
