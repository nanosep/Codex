import numpy as np
import pandas as pd

from src.indicators.atr import atr_wilder
from src.indicators.ema import ema
from src.indicators.rolling import rolling_max
from src.indicators.rsi import rsi_wilder


def test_ema_matches_pandas_ewm():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7], dtype=float)
    got = ema(s, 3)
    expected = s.ewm(span=3, adjust=False, min_periods=3).mean()
    pd.testing.assert_series_equal(got, expected)


def test_rsi_wilder_bounds_and_flat_series_case():
    s = pd.Series(np.linspace(10, 20, 40))
    out = rsi_wilder(s, 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()

    flat = pd.Series([100.0] * 40)
    flat_rsi = rsi_wilder(flat, 14).dropna()
    assert (flat_rsi == 50.0).all()


def test_atr_non_negative_and_known_case():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0])
    close = pd.Series([9.5, 10.5, 11.5, 12.5])
    out = atr_wilder(high, low, close, period=3)
    assert (out.dropna() >= 0).all()
    assert round(out.iloc[2], 6) == round((1.0 + 1.5 + 1.5) / 3.0, 6)
    expected_next = ((out.iloc[2] * 2) + 1.5) / 3.0
    assert round(out.iloc[3], 6) == round(expected_next, 6)


def test_atr_stop_formula_hh21_minus_4atr21():
    idx = pd.date_range("2025-01-01", periods=30, freq="D")
    high = pd.Series(np.arange(30, dtype=float) + 100.0, index=idx)
    low = high - 2.0
    close = high - 1.0
    atr21 = atr_wilder(high, low, close, period=21)
    hh21 = rolling_max(high, window=21)
    atr_stop = hh21 - 4.0 * atr21
    last = atr_stop.dropna().index[-1]
    assert atr_stop.loc[last] == hh21.loc[last] - 4.0 * atr21.loc[last]
