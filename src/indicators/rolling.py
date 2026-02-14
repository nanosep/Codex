from __future__ import annotations

import pandas as pd


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=window).max()
