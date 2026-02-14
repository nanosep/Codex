from __future__ import annotations

import argparse
import json
import platform
import sys
from math import floor
from pathlib import Path

import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled


class OneClassLabelError(Exception):
    def __init__(self, class_counts: dict[int, int]):
        super().__init__("Training labels contain only one class.")
        self.class_counts = class_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step 3 baseline model.")
    parser.add_argument("--labeled", required=True, help="Path to Step 2 labeled CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction. Default 0.2.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold. Default 0.5.")
    return parser.parse_args()


def chronological_split(features_df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1)")
    n = len(features_df)
    if n < 2:
        raise ValueError("Need at least 2 rows after feature cleaning.")

    split_idx = floor((1.0 - test_size) * n)
    split_idx = max(1, min(split_idx, n - 1))
    train_df = features_df.iloc[:split_idx].reset_index(drop=True)
    test_df = features_df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def build_model_pipeline():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn is required for Step 3 training. Install scikit-learn.") from exc

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def compute_metrics(y_test, y_prob, y_pred) -> dict:
    try:
        from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for Step 3 metrics. Install scikit-learn.") from exc

    metrics: dict = {}
    if len(set(y_test)) < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics


def train_and_export(labeled_csv: str | Path, out_dir: str | Path, test_size: float = 0.2, threshold: float = 0.5) -> dict:
    labeled_df = pd.read_csv(labeled_csv)
    features_df = build_features_from_labeled(labeled_df)
    if len(features_df) < 2:
        raise ValueError("Not enough rows after feature NaN drop.")

    train_df, test_df = chronological_split(features_df, test_size=test_size)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)
    class_counts = y_train.value_counts().to_dict()
    count_0 = int(class_counts.get(0, 0))
    count_1 = int(class_counts.get(1, 0))
    if count_0 == 0 or count_1 == 0:
        raise OneClassLabelError(class_counts={0: count_0, 1: count_1})

    try:
        from joblib import dump
    except ImportError as exc:
        raise ImportError("joblib is required for Step 3 model export. Install joblib.") from exc
    try:
        import sklearn
    except ImportError as exc:
        raise ImportError("scikit-learn is required for Step 3 training. Install scikit-learn.") from exc

    pipeline = build_model_pipeline()
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_prob, y_pred)
    metrics["python_version"] = platform.python_version()
    metrics["sklearn_version"] = sklearn.__version__
    metrics["pandas_version"] = pd.__version__

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(out_path / "features.csv", index=False)
    dump(pipeline, out_path / "model.joblib")

    predictions = pd.DataFrame(
        {
            "date": test_df["date"],
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )
    predictions.to_csv(out_path / "predictions.csv", index=False)

    with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    train_pos_rate = float(y_train.mean() * 100.0)
    test_pos_rate = float(y_test.mean() * 100.0)
    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_pos_rate_pct": train_pos_rate,
        "test_pos_rate_pct": test_pos_rate,
        "roc_auc": metrics["roc_auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "metrics": metrics,
        "features_path": str(out_path / "features.csv"),
        "model_path": str(out_path / "model.joblib"),
        "predictions_path": str(out_path / "predictions.csv"),
        "metrics_path": str(out_path / "metrics.json"),
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"train rows: {summary['train_rows']}")
    print(f"test rows: {summary['test_rows']}")
    print(f"positive rate train (%): {summary['train_pos_rate_pct']:.2f}")
    print(f"positive rate test (%): {summary['test_pos_rate_pct']:.2f}")
    print(f"roc_auc: {summary['roc_auc']}")
    print(f"balanced_accuracy: {summary['balanced_accuracy']:.6f}")


def main() -> None:
    args = parse_args()
    try:
        summary = train_and_export(
            labeled_csv=args.labeled,
            out_dir=args.out,
            test_size=args.test_size,
            threshold=args.threshold,
        )
        print_summary(summary)
    except OneClassLabelError as exc:
        print(f"class counts in train: #0={exc.class_counts.get(0, 0)}, #1={exc.class_counts.get(1, 0)}", file=sys.stderr)
        print(
            "Try increasing sl, decreasing tp, decreasing horizon, or regenerate sample data tuned to produce positives.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
