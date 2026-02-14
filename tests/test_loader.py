import pytest

from src.data.loader import load_ohlcv_csv


def test_loader_fails_when_required_columns_missing():
    path = "tests/fixtures/bad_missing_volume.csv"
    with pytest.raises(ValueError, match="missing required columns"):
        load_ohlcv_csv(path)
