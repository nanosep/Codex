from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from src.data.loader import load_ohlcv_csv
from src.engine.labeling import label_tp_sl_binary
from src.ml.model_zoo import run_model_zoo


def test_model_zoo_smoke_outputs_and_card_headings():
    demo_df = load_ohlcv_csv("data/sample_daily_demo.csv")
    labeled_df = label_tp_sl_binary(demo_df, horizon=40, tp_pct=0.01, sl_pct=0.003)

    run_id = uuid.uuid4().hex[:8]
    out_dir = Path("tests/_artifacts") / f"model_zoo_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)

    run_model_zoo(labeled_csv=labeled_path, out_dir=out_dir, n_splits=5, test_size=0.2, threshold=0.5)

    assert (out_dir / "model_candidates.json").exists()
    assert (out_dir / "best_model.joblib").exists()
    assert (out_dir / "best_model_predictions.csv").exists()
    assert (out_dir / "best_model_metrics.json").exists()
    card_path = out_dir / "best_model_card.md"
    assert card_path.exists()

    card_text = card_path.read_text(encoding="utf-8")
    required_headings = [
        "## Label definition",
        "## Feature set",
        "## Models compared",
        "## Cross-validation results",
        "## Winner and rationale",
        "## Test set results",
        "## Failure modes",
        "## Operational notes",
    ]
    for heading in required_headings:
        assert heading in card_text
