from __future__ import annotations

from pathlib import Path

from src.data.loader import load_ohlcv_csv
from src.engine.labeling import label_tp_sl_binary


DEMO_PATH = Path("data/sample_daily_demo.csv")
REALISTIC_PATH = Path("data/sample_daily_realistic.csv")


def test_c1_both_files_exist_and_load():
    assert DEMO_PATH.exists()
    assert REALISTIC_PATH.exists()
    demo_df = load_ohlcv_csv(DEMO_PATH)
    realistic_df = load_ohlcv_csv(REALISTIC_PATH)
    assert len(demo_df) > 0
    assert len(realistic_df) > 0


def test_c2_demo_dataset_label_distribution_target_band():
    demo_df = load_ohlcv_csv(DEMO_PATH)
    labeled = label_tp_sl_binary(demo_df, horizon=40, tp_pct=0.01, sl_pct=0.003)
    positives = int(labeled["label"].sum())
    total = len(labeled)
    positive_rate = (positives / total * 100.0) if total else 0.0
    assert positives >= 20
    assert 10.0 <= positive_rate <= 60.0


def test_c3_realistic_dataset_has_nonzero_positives_and_reasonable_rate():
    realistic_df = load_ohlcv_csv(REALISTIC_PATH)
    labeled = label_tp_sl_binary(realistic_df, horizon=40, tp_pct=0.01, sl_pct=0.003)
    positives = int(labeled["label"].sum())
    total = len(labeled)
    positive_rate = (positives / total * 100.0) if total else 0.0
    assert positives >= 5
    assert positive_rate <= 20.0
