from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from src.data.loader import load_ohlcv_csv
from src.engine.labeling import label_tp_sl_binary
from src.ml.model_zoo import run_model_zoo
from src.ml.threshold_search import run_threshold_search


def test_threshold_search_outputs_and_freeze_usage():
    out_dir = Path("tests/_artifacts") / f"threshold_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    market_df = load_ohlcv_csv("data/sample_daily_demo.csv")
    labeled = label_tp_sl_binary(market_df, horizon=40, tp_pct=0.01, sl_pct=0.003)
    labeled_path = out_dir / "labeled.csv"
    labeled.to_csv(labeled_path, index=False)

    run_model_zoo(labeled_csv=labeled_path, out_dir=out_dir, n_splits=5, test_size=0.2, threshold=0.5)
    run_threshold_search(
        csv_path="data/sample_daily_demo.csv",
        model_path=out_dir / "best_model.joblib",
        out_dir=out_dir,
        horizon=40,
        tp=0.01,
        sl=0.003,
        n_splits=5,
        test_size=0.2,
    )

    assert (out_dir / "threshold_search.csv").exists()
    assert (out_dir / "best_threshold.json").exists()
    assert (out_dir / "test_backtest_summary.json").exists()
    assert (out_dir / "test_equity.csv").exists()
    assert (out_dir / "test_trades.csv").exists()
    assert (out_dir / "test_chart.png").exists()

    with (out_dir / "best_threshold.json").open("r", encoding="utf-8") as f:
        best = json.load(f)
    with (out_dir / "test_backtest_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    chosen = float(best["chosen_threshold"])
    used = float(summary["params"]["threshold"])
    assert chosen == used
