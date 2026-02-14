from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled
from src.run_train_model import chronological_split


@dataclass
class ModelZooResult:
    candidates: list[dict]
    winner_name: str
    winner_pipeline: object
    predictions: pd.DataFrame
    metrics: dict
    model_card: str


def _build_model_factories():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return [
        {
            "model_name": "LogisticRegression",
            "params": {"class_weight": "balanced", "max_iter": 1000, "random_state": 42},
            "builder": lambda: Pipeline(
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
            ),
        },
        {
            "model_name": "KNeighborsClassifier",
            "params": {"n_neighbors": 15},
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=15)),
                ]
            ),
        },
        {
            "model_name": "RandomForestClassifier",
            "params": {"n_estimators": 300, "min_samples_leaf": 2, "random_state": 42},
            "builder": lambda: RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
            ),
        },
        {
            "model_name": "GaussianNB",
            "params": {},
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", GaussianNB()),
                ]
            ),
        },
    ]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def run_model_zoo(
    labeled_csv: str | Path,
    out_dir: str | Path,
    n_splits: int = 5,
    test_size: float = 0.2,
    threshold: float = 0.5,
) -> ModelZooResult:
    from joblib import dump
    import sklearn
    from sklearn.metrics import classification_report
    from sklearn.model_selection import TimeSeriesSplit

    labeled_df = pd.read_csv(labeled_csv)
    features_df = build_features_from_labeled(labeled_df)
    if len(features_df) < 20:
        raise ValueError("Not enough rows after feature building for model zoo.")

    train_df, test_df = chronological_split(features_df, test_size=test_size)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int).to_numpy()
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int).to_numpy()
    dates_test = pd.to_datetime(test_df["date"])

    train_pos_rate = float(np.mean(y_train) * 100.0)
    test_pos_rate = float(np.mean(y_test) * 100.0)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates: list[dict] = []
    model_defs = _build_model_factories()

    for model_def in model_defs:
        fold_roc = []
        fold_bal = []
        for fold_train_idx, fold_val_idx in tscv.split(x_train):
            model = model_def["builder"]()
            x_fold_train = x_train.iloc[fold_train_idx]
            y_fold_train = y_train[fold_train_idx]
            x_fold_val = x_train.iloc[fold_val_idx]
            y_fold_val = y_train[fold_val_idx]

            model.fit(x_fold_train, y_fold_train)
            y_prob_val = model.predict_proba(x_fold_val)[:, 1]
            y_pred_val = (y_prob_val >= threshold).astype(int)

            fold_roc.append(_safe_roc_auc(y_fold_val, y_prob_val))
            fold_bal.append(_safe_balanced_acc(y_fold_val, y_pred_val))

        cv_mean_roc_auc = float(np.nanmean(fold_roc)) if np.any(~np.isnan(fold_roc)) else float("-inf")
        cv_std_roc_auc = float(np.nanstd(fold_roc)) if np.any(~np.isnan(fold_roc)) else float("nan")
        cv_mean_bal_acc = float(np.nanmean(fold_bal)) if np.any(~np.isnan(fold_bal)) else float("-inf")
        cv_std_bal_acc = float(np.nanstd(fold_bal)) if np.any(~np.isnan(fold_bal)) else float("nan")

        candidates.append(
            {
                "model_name": model_def["model_name"],
                "params": model_def["params"],
                "cv_mean_roc_auc": cv_mean_roc_auc,
                "cv_std_roc_auc": cv_std_roc_auc,
                "cv_mean_bal_acc": cv_mean_bal_acc,
                "cv_std_bal_acc": cv_std_bal_acc,
                "train_positive_rate": train_pos_rate,
                "test_positive_rate": test_pos_rate,
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
            }
        )

    def sort_key(candidate: dict) -> tuple[float, float]:
        return (candidate["cv_mean_roc_auc"], candidate["cv_mean_bal_acc"])

    winner_summary = max(candidates, key=sort_key)
    winner_name = winner_summary["model_name"]
    winner_builder = next(m["builder"] for m in model_defs if m["model_name"] == winner_name)
    winner_pipeline = winner_builder()
    winner_pipeline.fit(x_train, y_train)

    y_prob = winner_pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    test_roc_auc = _safe_roc_auc(y_test, y_prob)
    test_bal_acc = _safe_balanced_acc(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    predictions = pd.DataFrame(
        {
            "date": dates_test,
            "y_true": y_test,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    metrics = {
        "test_roc_auc": None if np.isnan(test_roc_auc) else test_roc_auc,
        "test_balanced_acc": test_bal_acc,
        "classification_report": report,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "winner_model_name": winner_name,
    }

    model_card = _build_model_card(
        candidates=candidates,
        winner_name=winner_name,
        metrics=metrics,
        threshold=threshold,
        n_splits=n_splits,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "model_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    dump(winner_pipeline, out_path / "best_model.joblib")
    predictions.to_csv(out_path / "best_model_predictions.csv", index=False)
    with (out_path / "best_model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_path / "best_model_card.md").open("w", encoding="utf-8") as f:
        f.write(model_card)

    return ModelZooResult(
        candidates=candidates,
        winner_name=winner_name,
        winner_pipeline=winner_pipeline,
        predictions=predictions,
        metrics=metrics,
        model_card=model_card,
    )


def _build_model_card(candidates: list[dict], winner_name: str, metrics: dict, threshold: float, n_splits: int) -> str:
    lines: list[str] = []
    lines.append("## Label definition")
    lines.append(
        "Step 2 binary label is reused exactly: for bar t, TP hit if future high >= tp_level and SL hit if future low <= sl_level over horizon bars excluding t; label=1 only when TP hit and SL not hit."
    )
    lines.append("")
    lines.append("## Feature set")
    lines.append("Step 3 feature set is reused exactly (16 trailing-only features from price/volume/indicators).")
    lines.append("")
    lines.append("## Models compared")
    lines.append("- LogisticRegression")
    lines.append("- KNeighborsClassifier")
    lines.append("- RandomForestClassifier")
    lines.append("- GaussianNB")
    lines.append("")
    lines.append("## Cross-validation results")
    lines.append(f"TimeSeriesSplit with n_splits={n_splits} on train-only chronology.")
    lines.append("")
    lines.append("| Model | Mean ROC AUC | Std ROC AUC | Mean Bal Acc | Std Bal Acc |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in candidates:
        lines.append(
            f"| {c['model_name']} | {c['cv_mean_roc_auc']:.6f} | {c['cv_std_roc_auc']:.6f} | {c['cv_mean_bal_acc']:.6f} | {c['cv_std_bal_acc']:.6f} |"
        )
    lines.append("")
    lines.append("## Winner and rationale")
    lines.append(
        f"{winner_name} selected by highest mean CV ROC AUC; balanced accuracy used as tie-breaker. Threshold fixed at {threshold:.2f}."
    )
    lines.append("")
    lines.append("## Test set results")
    lines.append(f"- test_roc_auc: {metrics['test_roc_auc']}")
    lines.append(f"- test_balanced_acc: {metrics['test_balanced_acc']:.6f}")
    lines.append("")
    lines.append("## Failure modes")
    lines.append("- label strictness sensitivity (tp/sl/horizon)")
    lines.append("- class imbalance changes")
    lines.append("- non-stationarity / regime shifts")
    lines.append("- leakage risk if features ever use future info")
    lines.append("- threshold sensitivity (0.5 may be wrong)")
    lines.append("")
    lines.append("## Operational notes")
    lines.append("- Demo synthetic dataset is for pipeline validation and stable end-to-end runs.")
    lines.append("- Realistic synthetic dataset is market-like and can produce lower positive rates.")
    return "\n".join(lines) + "\n"
