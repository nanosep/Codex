from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_ohlcv_csv
from src.engine.labeling import label_tp_sl_binary
from src.engine.ml_backtest import run_ml_backtest
from src.features.build_features import FEATURE_COLUMNS, build_features_from_labeled
from src.run_train_model import chronological_split
from src.viz.plot import plot_backtest_chart


THRESHOLDS = [round(x, 2) for x in [i * 0.05 for i in range(1, 20)]]


def run_threshold_search(
    csv_path: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    horizon: int,
    tp: float,
    sl: float,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> dict:
    from joblib import load
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit

    market_df = load_ohlcv_csv(csv_path)
    labeled_df = label_tp_sl_binary(market_df, horizon=horizon, tp_pct=tp, sl_pct=sl)
    features_df = build_features_from_labeled(labeled_df)
    if len(features_df) < 50:
        raise ValueError("Not enough feature rows for threshold search.")

    train_features, test_features = chronological_split(features_df, test_size=test_size)
    if len(train_features) <= n_splits:
        raise ValueError("Not enough train rows for requested n_splits.")

    base_model = load(model_path)
    x_train = train_features[FEATURE_COLUMNS]
    y_train = train_features["label"].astype(int)
    x_test = test_features[FEATURE_COLUMNS]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    for th in THRESHOLDS:
        fold_total_return = []
        fold_dd = []
        fold_trades = []
        fold_winrate = []
        fold_exposure = []

        for fold_train_idx, fold_val_idx in tscv.split(x_train):
            model = clone(base_model)
            x_fold_train = x_train.iloc[fold_train_idx]
            y_fold_train = y_train.iloc[fold_train_idx]
            x_fold_val = x_train.iloc[fold_val_idx]
            dates_fold_val = pd.to_datetime(train_features.iloc[fold_val_idx]["date"])

            model.fit(x_fold_train, y_fold_train)
            y_prob_val = model.predict_proba(x_fold_val)[:, 1]
            signals_val = pd.DataFrame(
                {
                    "date": dates_fold_val,
                    "y_prob": y_prob_val,
                    "signal": (y_prob_val >= th).astype(int),
                }
            )
            market_val = market_df[market_df["date"].isin(dates_fold_val)].copy()
            result = run_ml_backtest(
                market_df=market_val,
                signals_df=signals_val,
                horizon=horizon,
                tp_pct=tp,
                sl_pct=sl,
                threshold=th,
            )
            fold_total_return.append(float(result.summary["total_return"]))
            fold_dd.append(float(result.summary["max_drawdown"]))
            fold_trades.append(float(result.summary["trades_count"]))
            fold_winrate.append(float(result.summary["win_rate"]))
            fold_exposure.append(float(result.summary["exposure"]))

        rows.append(
            {
                "threshold": th,
                "cv_mean_total_return": float(pd.Series(fold_total_return).mean()),
                "cv_std_total_return": float(pd.Series(fold_total_return).std(ddof=0)),
                "cv_mean_dd": float(pd.Series(fold_dd).mean()),
                "cv_mean_trades": float(pd.Series(fold_trades).mean()),
                "cv_mean_winrate": float(pd.Series(fold_winrate).mean()),
                "cv_mean_exposure": float(pd.Series(fold_exposure).mean()),
            }
        )

    search_df = pd.DataFrame(rows)
    best_row = search_df.sort_values(
        by=["cv_mean_total_return", "cv_std_total_return"], ascending=[False, True]
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    winner_model = clone(base_model)
    winner_model.fit(x_train, y_train)
    y_prob_test = winner_model.predict_proba(x_test)[:, 1]
    signals_test = pd.DataFrame(
        {
            "date": pd.to_datetime(test_features["date"]),
            "y_prob": y_prob_test,
            "signal": (y_prob_test >= best_threshold).astype(int),
        }
    )
    market_test = market_df[market_df["date"].isin(pd.to_datetime(test_features["date"]))].copy()
    test_result = run_ml_backtest(
        market_df=market_test,
        signals_df=signals_test,
        horizon=horizon,
        tp_pct=tp,
        sl_pct=sl,
        threshold=best_threshold,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(out_path / "threshold_search.csv", index=False)

    best_threshold_payload = {
        "chosen_threshold": best_threshold,
        "selection_stats": {
            "cv_mean_total_return": float(best_row["cv_mean_total_return"]),
            "cv_std_total_return": float(best_row["cv_std_total_return"]),
            "cv_mean_dd": float(best_row["cv_mean_dd"]),
            "cv_mean_trades": float(best_row["cv_mean_trades"]),
            "cv_mean_winrate": float(best_row["cv_mean_winrate"]),
            "cv_mean_exposure": float(best_row["cv_mean_exposure"]),
        },
        "params": {
            "tp": float(tp),
            "sl": float(sl),
            "horizon": int(horizon),
            "n_splits": int(n_splits),
            "test_size": float(test_size),
            "ambiguity_rule": "If both TP and SL are hit in the same bar, SL is assumed first (conservative).",
        },
        "primary_metric": "total_return",
    }
    with (out_path / "best_threshold.json").open("w", encoding="utf-8") as f:
        json.dump(best_threshold_payload, f, indent=2)

    test_summary = dict(test_result.summary)
    test_summary["dataset_split"] = "test"
    with (out_path / "test_backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    test_result.equity.to_csv(out_path / "test_equity.csv", index=False)
    test_result.trades.to_csv(out_path / "test_trades.csv", index=False)
    plot_backtest_chart(
        price_df=market_test[["date", "close"]],
        entries_df=test_result.entries,
        exits_df=test_result.exits,
        out_path=out_path / "test_chart.png",
    )

    return {
        "best_threshold": best_threshold,
        "search_path": str(out_path / "threshold_search.csv"),
        "best_threshold_path": str(out_path / "best_threshold.json"),
        "test_summary_path": str(out_path / "test_backtest_summary.json"),
    }
