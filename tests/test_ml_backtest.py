from __future__ import annotations

import pandas as pd

from src.engine.ml_backtest import run_ml_backtest


def _market_df(highs, lows, closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n,
        }
    )


def _signals_df(n, on_indices):
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "y_prob": [0.9 if i in on_indices else 0.1 for i in range(n)],
            "signal": [1 if i in on_indices else 0 for i in range(n)],
        }
    )


def test_tp_hit_first():
    market = _market_df(
        highs=[100.1, 101.2, 100.8, 100.7],
        lows=[99.9, 99.8, 99.9, 99.85],
        closes=[100.0, 100.3, 100.2, 100.1],
    )
    signals = _signals_df(4, {0})
    result = run_ml_backtest(market, signals, horizon=3, tp_pct=0.01, sl_pct=0.003, threshold=0.5)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "tp"


def test_sl_hit_first():
    market = _market_df(
        highs=[100.1, 100.5, 100.6, 100.4],
        lows=[99.9, 99.6, 99.7, 99.8],
        closes=[100.0, 100.1, 100.2, 100.3],
    )
    signals = _signals_df(4, {0})
    result = run_ml_backtest(market, signals, horizon=3, tp_pct=0.01, sl_pct=0.003, threshold=0.5)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "sl"


def test_both_hit_same_bar_exits_sl():
    market = _market_df(
        highs=[100.1, 101.3, 100.6, 100.4],
        lows=[99.9, 99.6, 99.7, 99.8],
        closes=[100.0, 100.1, 100.2, 100.3],
    )
    signals = _signals_df(4, {0})
    result = run_ml_backtest(market, signals, horizon=3, tp_pct=0.01, sl_pct=0.003, threshold=0.5)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "sl"


def test_time_exit_when_neither_tp_nor_sl():
    market = _market_df(
        highs=[100.1, 100.9, 100.8, 100.95],
        lows=[99.9, 99.8, 99.75, 99.8],
        closes=[100.0, 100.1, 100.2, 100.4],
    )
    signals = _signals_df(4, {0})
    result = run_ml_backtest(market, signals, horizon=3, tp_pct=0.01, sl_pct=0.003, threshold=0.5)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time"
    assert float(trade["exit_price"]) == 100.4


def test_no_overlapping_positions():
    market = _market_df(
        highs=[100.2, 100.4, 101.2, 100.3, 101.5, 100.2, 101.3],
        lows=[99.9, 99.8, 99.9, 99.9, 99.8, 99.9, 99.8],
        closes=[100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
    )
    signals = _signals_df(7, {0, 1, 2, 3, 4, 5})
    result = run_ml_backtest(market, signals, horizon=2, tp_pct=0.01, sl_pct=0.003, threshold=0.5)
    assert len(result.trades) == 2
    first_exit = pd.to_datetime(result.trades.iloc[0]["exit_date"])
    second_entry = pd.to_datetime(result.trades.iloc[1]["entry_date"])
    assert second_entry > first_exit
