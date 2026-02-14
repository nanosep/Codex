from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data.loader import load_ohlcv_csv, validate_ohlcv_dataframe
from src.features.build_features import build_features_from_labeled
from src.pipeline.workflows import label_from_df, run_backtest_from_df, run_ml_strategy_from_df, train_from_labeled_path

SYSTEM_PROMPT = """You are a quantitative trading QA analyst reviewing an ML-driven backtest UI.
You must be precise, skeptical, and operational.

Rules:
- You are NOT giving financial advice. You are auditing a prototype.
- Base every claim only on the data provided in the payload.
- If something is missing, say “unknown from payload.”
- Be concise and concrete.

Output format (must follow):
1) What we’re seeing (5 bullets max)
2) Red flags / failure modes (5 bullets max)
3) What to try next (5 bullets max) — must include specific parameter or experiment suggestions
4) If you only do one thing: (one sentence)"""


@st.cache_data
def cached_load_csv(path: str) -> pd.DataFrame:
    return load_ohlcv_csv(path)


@st.cache_data
def cached_load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    return validate_ohlcv_dataframe(raw_df)


@st.cache_data
def cached_build_features(labeled_df: pd.DataFrame) -> pd.DataFrame:
    return build_features_from_labeled(labeled_df)


@st.cache_resource
def cached_load_model_from_path(path: str):
    from joblib import load

    return load(path)


@st.cache_resource
def cached_load_model_from_bytes(model_bytes: bytes):
    from joblib import load

    return load(io.BytesIO(model_bytes))


@st.cache_resource
def cached_openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


@st.cache_data
def cached_payload(
    dataset_mode: str,
    date_min: str,
    date_max: str,
    horizon: int,
    tp: float,
    sl: float,
    labeled_path: str,
    metrics_path: str,
    model_candidates_path: str,
    best_model_metrics_path: str,
    ml_summary_path: str,
    ml_trades_path: str,
    best_threshold_path: str,
    threshold_search_path: str,
    threshold: float,
) -> dict:
    payload: dict = {
        "context": {
            "dataset_mode": dataset_mode,
            "date_range": {"start": date_min, "end": date_max},
            "timeframe": "daily",
        },
        "label_definition": {
            "horizon": int(horizon),
            "tp_pct": float(tp),
            "sl_pct": float(sl),
            "ambiguity_rule": "sl_first_if_both_hit_same_bar",
        },
        "model": {"threshold": float(threshold)},
    }

    labeled_file = Path(labeled_path)
    if labeled_file.exists():
        labeled_df = pd.read_csv(labeled_file)
        payload["label_definition"]["n_rows_labeled"] = int(len(labeled_df))
        payload["label_definition"]["positive_rate"] = (
            float(labeled_df["label"].mean() * 100.0) if len(labeled_df) and "label" in labeled_df.columns else None
        )
    else:
        payload["label_definition"]["n_rows_labeled"] = None
        payload["label_definition"]["positive_rate"] = None

    metrics_file = Path(metrics_path)
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        payload["model"]["test_metrics"] = {
            "roc_auc": m.get("roc_auc"),
            "balanced_accuracy": m.get("balanced_accuracy"),
        }

    candidates_file = Path(model_candidates_path)
    if candidates_file.exists():
        candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
        payload["model"]["cv_results_summary"] = [
            {
                "model_name": c.get("model_name"),
                "cv_mean_roc_auc": c.get("cv_mean_roc_auc"),
                "cv_std_roc_auc": c.get("cv_std_roc_auc"),
                "cv_mean_bal_acc": c.get("cv_mean_bal_acc"),
                "cv_std_bal_acc": c.get("cv_std_bal_acc"),
            }
            for c in candidates
        ]

    best_metrics_file = Path(best_model_metrics_path)
    if best_metrics_file.exists():
        bm = json.loads(best_metrics_file.read_text(encoding="utf-8"))
        payload["model"]["model_name"] = bm.get("winner_model_name")
        payload["model"]["test_metrics"] = {
            "roc_auc": bm.get("test_roc_auc"),
            "balanced_accuracy": bm.get("test_balanced_acc"),
        }

    ml_summary_file = Path(ml_summary_path)
    if ml_summary_file.exists():
        ms = json.loads(ml_summary_file.read_text(encoding="utf-8"))
        payload["ml_backtest"] = {
            "trades_count": ms.get("trades_count"),
            "total_return_pct": ms.get("total_return"),
            "win_rate_pct": ms.get("win_rate"),
            "max_drawdown_pct": ms.get("max_drawdown"),
            "exposure_pct": ms.get("exposure"),
            "avg_return_pct": ms.get("average_return"),
        }

    ml_trades_file = Path(ml_trades_path)
    if ml_trades_file.exists():
        td = pd.read_csv(ml_trades_file)
        if len(td):
            payload.setdefault("ml_backtest", {})
            payload["ml_backtest"]["median_return_pct"] = float(td["return_pct"].median() * 100.0)
            payload["ml_backtest"]["first_trade_date"] = str(td["entry_date"].iloc[0])
            payload["ml_backtest"]["last_trade_date"] = str(td["exit_date"].iloc[-1])
            sample_cols = ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "return_pct"]
            payload["ml_backtest"]["sample_trades"] = {
                "first_5": td[sample_cols].head(5).to_dict(orient="records"),
                "worst_5_by_return": td.sort_values("return_pct").head(5)[sample_cols].to_dict(orient="records"),
            }

    best_threshold_file = Path(best_threshold_path)
    threshold_search_file = Path(threshold_search_path)
    if best_threshold_file.exists():
        bt = json.loads(best_threshold_file.read_text(encoding="utf-8"))
        payload["step6_threshold_policy"] = {"chosen_threshold": bt.get("chosen_threshold")}
    if threshold_search_file.exists():
        ts = pd.read_csv(threshold_search_file)
        top5 = ts.sort_values("cv_mean_total_return", ascending=False).head(5)
        payload.setdefault("step6_threshold_policy", {})
        payload["step6_threshold_policy"]["top_5_thresholds_by_cv_return"] = top5[
            ["threshold", "cv_mean_total_return", "cv_mean_dd", "cv_mean_trades"]
        ].to_dict(orient="records")
    return payload


def _download_file(label: str, path: str, mime: str) -> None:
    file_path = Path(path)
    if file_path.exists() and file_path.stat().st_size > 0:
        st.download_button(label=label, data=file_path.read_bytes(), file_name=file_path.name, mime=mime)


def _existing_path(primary: Path, fallback: Path) -> Path | None:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def _deterministic_insights(payload: dict, tab_name: str) -> str:
    label_info = payload.get("label_definition", {})
    pos_rate = label_info.get("positive_rate")
    n_rows = label_info.get("n_rows_labeled")
    warnings = []
    if pos_rate is not None and pos_rate < 2:
        warnings.append("Positive rate < 2%: signal learning may be unstable.")
    if n_rows is not None and n_rows < 200:
        warnings.append("Labeled rows < 200: statistical confidence is limited.")
    ml = payload.get("ml_backtest", {})
    if ml.get("trades_count") is not None and ml["trades_count"] < 10:
        warnings.append("Trades < 10: backtest is likely not meaningful.")
    if ml.get("max_drawdown_pct") is not None and ml["max_drawdown_pct"] > 20:
        warnings.append("Max drawdown > 20%: risk is high.")
    if not warnings:
        warnings.append("No critical rule-based warning triggered by current payload.")

    lines = [
        f"Dataset mode: `{payload.get('context', {}).get('dataset_mode', 'unknown')}`",
        f"Label params: horizon={label_info.get('horizon')}, tp={label_info.get('tp_pct')}, sl={label_info.get('sl_pct')}",
        f"Class balance: positive_rate={pos_rate if pos_rate is not None else 'unknown'}%, n_rows={n_rows if n_rows is not None else 'unknown'}",
        f"Tab: {tab_name}",
        "Warnings:",
    ]
    lines.extend([f"- {w}" for w in warnings[:5]])
    return "\n".join(lines)


def _get_api_key() -> str | None:
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def _render_ai_insights(tab_key: str, payload: dict) -> None:
    with st.expander("AI Insights (optional)"):
        if st.button("Generate AI analysis for this tab", key=f"ai_btn_{tab_key}"):
            api_key = _get_api_key()
            if not api_key:
                st.info("Set OPENAI_API_KEY as an env var or Streamlit Community Cloud secret.")
            else:
                try:
                    client = cached_openai_client(api_key)
                    user_prompt = (
                        "Here is the current screen payload as JSON. Analyze it using the required output format.\n\n"
                        "PAYLOAD_JSON:\n"
                        f"{json.dumps(payload, ensure_ascii=True)}"
                    )
                    resp = client.responses.create(
                        model="gpt-4.1-mini",
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    st.session_state[f"ai_text_{tab_key}"] = resp.output_text
                except Exception as exc:
                    st.error(f"AI generation failed: {exc}")
        if st.session_state.get(f"ai_text_{tab_key}"):
            st.markdown(st.session_state[f"ai_text_{tab_key}"])


def _plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    matrix = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Backtest/Label/Train", layout="wide")
    st.title("Minimal Trading Pipeline")
    repo_root = Path(__file__).resolve().parent
    with st.expander("About this app"):
        overview_path = repo_root / "docs" / "overview.md"
        if overview_path.exists():
            st.markdown(overview_path.read_text(encoding="utf-8"))
        else:
            st.info("Overview document not found. Expected path: docs/overview.md")

    st.sidebar.header("Input")
    data_mode = st.sidebar.radio(
        "Data source",
        ["Demo synthetic (balanced-ish)", "Realistic synthetic (market-like)", "Upload CSV"],
        index=0,
    )
    uploaded = st.sidebar.file_uploader("OHLCV CSV", type=["csv"])
    if uploaded is not None:
        st.session_state["uploaded_csv_bytes"] = uploaded.getvalue()

    horizon = st.sidebar.number_input("Horizon", min_value=1, value=40, step=1)
    tp = st.sidebar.number_input("TP", min_value=0.0001, value=0.01, format="%.4f")
    sl = st.sidebar.number_input("SL", min_value=0.0001, value=0.003, format="%.4f")
    threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, format="%.2f")
    test_size = st.sidebar.number_input("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05, format="%.2f")

    st.sidebar.subheader("ML Strategy params")
    ml_threshold = st.sidebar.slider("ML threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    model_source = st.sidebar.radio("Model source", ["Use best_model.joblib from out/", "Upload model.joblib"], index=0)
    model_upload = None
    if model_source == "Upload model.joblib":
        model_upload = st.sidebar.file_uploader("Upload model.joblib", type=["joblib"])
        if model_upload is not None:
            st.session_state["uploaded_model_bytes"] = model_upload.getvalue()

    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = st.sidebar.text_input("Run ID", value=default_run_id).strip() or default_run_id
    run_out_dir = repo_root / "out" / run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if data_mode == "Demo synthetic (balanced-ish)":
        csv_path = repo_root / "data" / "sample_daily_demo.csv"
        df = cached_load_csv(str(csv_path))
        dataset_mode = "demo"
    elif data_mode == "Realistic synthetic (market-like)":
        csv_path = repo_root / "data" / "sample_daily_realistic.csv"
        df = cached_load_csv(str(csv_path))
        dataset_mode = "realistic"
    else:
        file_bytes = st.session_state.get("uploaded_csv_bytes")
        if not file_bytes:
            st.warning("Upload a CSV or select a built-in dataset.")
            st.stop()
        df = cached_load_uploaded_csv(file_bytes)
        dataset_mode = "upload"

    if "last_status" not in st.session_state:
        st.session_state["last_status"] = "No runs yet."

    st.caption(
        f"Loaded rows: {len(df)} | Date range: {pd.to_datetime(df['date']).min().date()} -> {pd.to_datetime(df['date']).max().date()}"
    )

    tab_backtest, tab_labeling, tab_training, tab_model_zoo, tab_ml_strategy, tab_threshold_policy = st.tabs(
        [
            "Backtest (Step 1)",
            "Labeling (Step 2)",
            "Training (Step 3)",
            "Model Zoo (Step 4)",
            "ML Strategy (Step 5)",
            "Threshold Policy (Step 6)",
        ]
    )

    labeled_path = str(run_out_dir / "labeled.csv")
    payload = cached_payload(
        dataset_mode=dataset_mode,
        date_min=str(pd.to_datetime(df["date"]).min().date()),
        date_max=str(pd.to_datetime(df["date"]).max().date()),
        horizon=int(horizon),
        tp=float(tp),
        sl=float(sl),
        labeled_path=labeled_path,
        metrics_path=str(run_out_dir / "metrics.json"),
        model_candidates_path=str(_existing_path(run_out_dir / "model_candidates.json", repo_root / "out" / "model_candidates.json") or ""),
        best_model_metrics_path=str(
            _existing_path(run_out_dir / "best_model_metrics.json", repo_root / "out" / "best_model_metrics.json") or ""
        ),
        ml_summary_path=str(run_out_dir / "ml_summary.json"),
        ml_trades_path=str(run_out_dir / "ml_trades.csv"),
        best_threshold_path=str(_existing_path(run_out_dir / "best_threshold.json", repo_root / "out" / "best_threshold.json") or ""),
        threshold_search_path=str(_existing_path(run_out_dir / "threshold_search.csv", repo_root / "out" / "threshold_search.csv") or ""),
        threshold=float(ml_threshold),
    )

    with tab_backtest:
        st.subheader("Step 1 Backtest")
        st.write(f"Last run status: {st.session_state['last_status']}")
        if st.button("Run Step 1", use_container_width=True):
            trades_df, equity_df, chart_path = run_backtest_from_df(df, run_out_dir)
            st.session_state["trades_path"] = str(run_out_dir / "trades.csv")
            st.session_state["equity_path"] = str(run_out_dir / "equity.csv")
            st.session_state["chart_path"] = chart_path
            st.session_state["last_status"] = "Step 1 completed."
            st.success("Backtest completed.")

        if st.session_state.get("trades_path") and Path(st.session_state["trades_path"]).exists():
            trades_df = pd.read_csv(st.session_state["trades_path"])
            equity_df = pd.read_csv(st.session_state["equity_path"])
            st.dataframe(trades_df.head(200), use_container_width=True)
            if not equity_df.empty:
                eq = equity_df.copy()
                eq["date"] = pd.to_datetime(eq["date"])
                st.line_chart(eq.set_index("date")["equity"])
            st.image(st.session_state["chart_path"])
            _download_file("Download trades.csv", st.session_state["trades_path"], "text/csv")
            _download_file("Download equity.csv", st.session_state["equity_path"], "text/csv")
            _download_file("Download chart.png", st.session_state["chart_path"], "image/png")

    with tab_labeling:
        st.subheader("Step 2 Labeling")
        if st.button("Run Step 2", use_container_width=True):
            out = label_from_df(df, int(horizon), float(tp), float(sl), run_out_dir)
            st.session_state["labeled_path"] = out
            st.session_state["last_status"] = "Step 2 completed."
            st.success("Labeling completed.")

        lp = st.session_state.get("labeled_path", labeled_path)
        if Path(lp).exists():
            labeled_df = pd.read_csv(lp)
            st.dataframe(labeled_df.head(200), use_container_width=True)
            _download_file("Download labeled.csv", lp, "text/csv")
            positives = int(labeled_df["label"].sum()) if len(labeled_df) else 0
            rate = positives / len(labeled_df) * 100.0 if len(labeled_df) else 0.0
            st.write(f"total labeled rows: {len(labeled_df)}")
            st.write(f"number of positives: {positives}")
            st.write(f"positive rate (%): {rate:.2f}")
        else:
            st.info("Run Step 2 to create labeled.csv.")

    with tab_training:
        st.subheader("Step 3 Training")
        lp = st.session_state.get("labeled_path", labeled_path)
        can_train = False
        if not Path(lp).exists():
            st.info("Run Step 2 first to create labeled.csv.")
        else:
            ldf = pd.read_csv(lp)
            c = ldf["label"].value_counts().to_dict()
            if int(c.get(0, 0)) == 0 or int(c.get(1, 0)) == 0:
                st.warning(
                    "Try increasing sl, decreasing tp, decreasing horizon, or regenerate sample data tuned to produce positives."
                )
            else:
                can_train = True
        if st.button("Run Step 3", use_container_width=True, disabled=not can_train):
            metrics, predictions_path, model_path = train_from_labeled_path(
                labeled_csv=lp,
                out_dir=run_out_dir,
                test_size=float(test_size),
                threshold=float(threshold),
            )
            st.session_state["metrics_path"] = str(run_out_dir / "metrics.json")
            st.session_state["predictions_path"] = predictions_path
            st.session_state["model_path"] = model_path
            st.session_state["features_path"] = str(run_out_dir / "features.csv")
            st.session_state["metrics_dict"] = metrics
            st.session_state["last_status"] = "Step 3 completed."
            st.success("Training completed.")

        mp = st.session_state.get("metrics_path", str(run_out_dir / "metrics.json"))
        pp = st.session_state.get("predictions_path", str(run_out_dir / "predictions.csv"))
        if Path(mp).exists():
            metrics_dict = json.loads(Path(mp).read_text(encoding="utf-8"))
            st.json(metrics_dict)
            if Path(pp).exists():
                preds = pd.read_csv(pp)
                st.dataframe(preds.head(200), use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(preds["y_prob"], bins=20, color="steelblue", alpha=0.8)
                ax.set_title("Probability Histogram (Test)")
                ax.set_xlabel("y_prob")
                ax.set_ylabel("count")
                fig.tight_layout()
                st.pyplot(fig)

                y_pred_now = (preds["y_prob"] >= float(threshold)).astype(int)
                cm_fig = _plot_confusion_matrix(preds["y_true"], y_pred_now)
                st.pyplot(cm_fig)

            st.markdown("**Deterministic Insights**")
            st.code(_deterministic_insights(payload, "Step 3"), language="text")
            _render_ai_insights("step3", payload)

            _download_file("Download metrics.json", mp, "application/json")
            _download_file("Download predictions.csv", pp, "text/csv")
            _download_file("Download model.joblib", st.session_state.get("model_path", str(run_out_dir / "model.joblib")), "application/octet-stream")
            _download_file("Download features.csv", st.session_state.get("features_path", str(run_out_dir / "features.csv")), "text/csv")

    with tab_model_zoo:
        st.subheader("Step 4 Model Zoo")
        cpath = _existing_path(run_out_dir / "model_candidates.json", repo_root / "out" / "model_candidates.json")
        if cpath is None:
            st.info("No model_candidates.json found. Run Step 4 Model Zoo first.")
        else:
            candidates = json.loads(cpath.read_text(encoding="utf-8"))
            cdf = pd.DataFrame(candidates)
            st.dataframe(cdf, use_container_width=True)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(
                cdf["model_name"],
                cdf["cv_mean_roc_auc"],
                yerr=cdf["cv_std_roc_auc"],
                color="teal",
                alpha=0.85,
                capsize=4,
            )
            ax.set_title("Model Comparison: CV Mean ROC AUC")
            ax.set_ylabel("ROC AUC")
            ax.set_xlabel("Model")
            fig.tight_layout()
            st.pyplot(fig)
            st.markdown("**Deterministic Insights**")
            st.code(_deterministic_insights(payload, "Step 4"), language="text")
            _render_ai_insights("step4", payload)

    with tab_ml_strategy:
        st.subheader("Step 5 ML Strategy")
        model = None
        model_ready = False
        if model_source == "Use best_model.joblib from out/":
            best_model_path = _existing_path(run_out_dir / "best_model.joblib", repo_root / "out" / "best_model.joblib")
            if best_model_path is None:
                st.info("No best_model.joblib found. Run Step 4 Model Zoo first or upload a model.")
            else:
                try:
                    model = cached_load_model_from_path(str(best_model_path))
                    model_ready = True
                except Exception as exc:
                    st.error(f"Failed to load model: {exc}")
        else:
            mb = st.session_state.get("uploaded_model_bytes")
            if not mb:
                st.info("Upload a .joblib model to run ML backtest.")
            else:
                try:
                    model = cached_load_model_from_bytes(mb)
                    model_ready = True
                except Exception as exc:
                    st.error(f"Failed to load uploaded model: {exc}")

        if st.button("Run ML Backtest", use_container_width=True, disabled=not model_ready):
            out = run_ml_strategy_from_df(
                df=df,
                model=model,
                out_dir=run_out_dir,
                horizon=int(horizon),
                tp=float(tp),
                sl=float(sl),
                threshold=float(ml_threshold),
            )
            st.session_state["ml_summary"] = out["summary"]
            st.session_state["ml_signals_path"] = out["signals_path"]
            st.session_state["ml_trades_path"] = out["trades_path"]
            st.session_state["ml_equity_path"] = out["equity_path"]
            st.session_state["ml_chart_path"] = out["chart_path"]
            st.session_state["ml_summary_path"] = out["summary_path"]
            st.success("ML backtest completed.")

        sp = st.session_state.get("ml_summary_path", str(run_out_dir / "ml_summary.json"))
        tp_path = st.session_state.get("ml_trades_path", str(run_out_dir / "ml_trades.csv"))
        ep = st.session_state.get("ml_equity_path", str(run_out_dir / "ml_equity.csv"))
        cp = st.session_state.get("ml_chart_path", str(run_out_dir / "ml_chart.png"))
        if Path(sp).exists():
            summary = json.loads(Path(sp).read_text(encoding="utf-8"))
            st.write(f"trades: {summary['trades_count']}")
            st.write(f"total return: {summary['total_return']:.2f}")
            st.write(f"win rate: {summary['win_rate']:.2f}")
            st.write(f"max drawdown: {summary['max_drawdown']:.2f}")
            st.write(f"exposure: {summary['exposure']:.2f}")
            st.image(cp)

            trades = pd.read_csv(tp_path) if Path(tp_path).exists() else pd.DataFrame()
            equity = pd.read_csv(ep) if Path(ep).exists() else pd.DataFrame()
            st.dataframe(trades.head(500), use_container_width=True)
            if not equity.empty:
                eq = equity.copy()
                eq["date"] = pd.to_datetime(eq["date"])
                curve = eq.set_index("date")["equity"]
                st.line_chart(curve)
                dd = (curve / curve.cummax()) - 1.0
                dd_fig, dd_ax = plt.subplots(figsize=(7, 2.5))
                dd_ax.plot(dd.index, dd.values, color="crimson")
                dd_ax.set_title("Drawdown Curve")
                dd_ax.set_ylabel("drawdown")
                dd_fig.tight_layout()
                st.pyplot(dd_fig)

            if not trades.empty and "return_pct" in trades.columns:
                tr_fig, tr_ax = plt.subplots(figsize=(6, 3))
                tr_ax.hist(trades["return_pct"] * 100.0, bins=20, color="darkorange", alpha=0.85)
                tr_ax.set_title("Trade Returns Distribution")
                tr_ax.set_xlabel("return %")
                tr_ax.set_ylabel("count")
                tr_fig.tight_layout()
                st.pyplot(tr_fig)

            st.markdown("**Deterministic Insights**")
            st.code(_deterministic_insights(payload, "Step 5"), language="text")
            _render_ai_insights("step5", payload)

            _download_file("Download ml_signals.csv", st.session_state.get("ml_signals_path", str(run_out_dir / "ml_signals.csv")), "text/csv")
            _download_file("Download ml_trades.csv", tp_path, "text/csv")
            _download_file("Download ml_equity.csv", ep, "text/csv")
            _download_file("Download ml_chart.png", cp, "image/png")
            _download_file("Download ml_summary.json", sp, "application/json")

    with tab_threshold_policy:
        st.subheader("Step 6 Threshold Policy")
        tsp = _existing_path(run_out_dir / "threshold_search.csv", repo_root / "out" / "threshold_search.csv")
        btp = _existing_path(run_out_dir / "best_threshold.json", repo_root / "out" / "best_threshold.json")
        if tsp is None:
            st.info("No threshold_search.csv found. Run Step 6 Threshold Policy first.")
        else:
            ts = pd.read_csv(tsp)
            st.dataframe(ts, use_container_width=True)
            fig1, ax1 = plt.subplots(figsize=(7, 3))
            ax1.plot(ts["threshold"], ts["cv_mean_total_return"], marker="o", color="navy")
            ax1.set_title("Threshold vs CV Mean Total Return")
            ax1.set_xlabel("threshold")
            ax1.set_ylabel("cv mean total return")
            fig1.tight_layout()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.plot(ts["threshold"], ts["cv_mean_dd"], marker="o", color="purple")
            ax2.set_title("Threshold vs CV Mean Drawdown")
            ax2.set_xlabel("threshold")
            ax2.set_ylabel("cv mean drawdown")
            fig2.tight_layout()
            st.pyplot(fig2)

            if btp is not None:
                st.json(json.loads(btp.read_text(encoding="utf-8")))
            st.markdown("**Deterministic Insights**")
            st.code(_deterministic_insights(payload, "Step 6"), language="text")
            _render_ai_insights("step6", payload)


if __name__ == "__main__":
    main()
