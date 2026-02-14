import pandas as pd

from src.engine.labeling import OUTPUT_COLUMNS, label_tp_sl_binary


def _base_df(closes: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
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


def test_l1_clean_win():
    df = _base_df(
        closes=[100, 100.3, 100.4, 100.5],
        highs=[100.0, 101.2, 100.8, 100.7],
        lows=[100.0, 99.8, 99.9, 99.95],
    )
    labeled = label_tp_sl_binary(df, horizon=2, tp_pct=0.01, sl_pct=0.003)
    first = labeled.iloc[0]
    assert int(first["tp_hit"]) == 1
    assert int(first["sl_hit"]) == 0
    assert int(first["label"]) == 1


def test_l2_sl_touched_sets_label_zero_even_if_tp_hits():
    df = _base_df(
        closes=[100, 100.2, 100.3, 100.4],
        highs=[100.0, 101.3, 100.9, 100.8],
        lows=[100.0, 99.6, 99.7, 99.8],
    )
    labeled = label_tp_sl_binary(df, horizon=2, tp_pct=0.01, sl_pct=0.003)
    first = labeled.iloc[0]
    assert int(first["tp_hit"]) == 1
    assert int(first["sl_hit"]) == 1
    assert int(first["label"]) == 0


def test_l3_tail_drop_n_minus_horizon_rows():
    df = _base_df(
        closes=[100, 101, 102, 103, 104],
        highs=[100, 101, 102, 103, 104],
        lows=[99, 100, 101, 102, 103],
    )
    labeled = label_tp_sl_binary(df, horizon=2, tp_pct=0.01, sl_pct=0.003)
    assert len(labeled) == 3
    assert list(labeled.columns) == OUTPUT_COLUMNS
