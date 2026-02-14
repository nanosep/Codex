from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled
from src.run_train_model import chronological_split, train_and_export


def _make_labeled_df(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0.05, 0.6, size=n))
    open_ = close + rng.normal(0, 0.15, size=n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.8, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.8, size=n)
    volume = rng.integers(100_000, 900_000, size=n)
    label = np.array([0, 1] * (n // 2) + ([0] if n % 2 else []))

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "entry_price": close,
            "tp_level": close * 1.01,
            "sl_level": close * 0.997,
            "tp_hit": label,
            "sl_hit": 1 - label,
            "label": label,
        }
    )


def test_t31_feature_shape_and_numeric_types():
    labeled = _make_labeled_df()
    features = build_features_from_labeled(labeled)
    for col in FEATURE_COLUMNS:
        assert col in features.columns
        assert pd.api.types.is_numeric_dtype(features[col])
    assert "label" in features.columns
    assert len(features) > 0


def test_t32_chronological_split_order():
    labeled = _make_labeled_df()
    features = build_features_from_labeled(labeled)
    train_df, test_df = chronological_split(features, test_size=0.2)
    assert train_df["date"].max() < test_df["date"].min()


def test_t33_smoke_run_exports_artifacts():
    labeled = _make_labeled_df()
    run_id = uuid.uuid4().hex[:8]
    out_dir = Path("tests/_artifacts") / f"train_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled_fixture.csv"
    labeled.to_csv(labeled_path, index=False)

    summary = train_and_export(labeled_csv=labeled_path, out_dir=out_dir, test_size=0.2, threshold=0.5)
    assert summary["train_rows"] > 0
    assert summary["test_rows"] > 0
    assert (out_dir / "features.csv").exists()
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "metrics.json").exists()
