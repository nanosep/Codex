from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(window=period, min_periods=period).mean()
