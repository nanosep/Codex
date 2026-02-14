# Minimal Rule-Based Backtester (Step 1)

## About this artifact

For a plain-English walkthrough of datasets, the Step 1–6 user flow, assumptions, and glossary, see `docs/overview.md`.
The Streamlit app also includes an in-app About expander that displays this same overview.

## Install & Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
python -m src.run_backtest --csv data/sample_daily.csv --out out
python -m src.run_labeling --csv data/sample_daily.csv --out out --horizon 40 --tp 0.01 --sl 0.003
python -m src.run_train_model --labeled out/labeled.csv --out out
streamlit run streamlit_app.py
```

## Generate sample data (optional)

```bash
python scripts/make_sample_data.py
python scripts/make_sample_data_demo.py
python scripts/make_sample_data_realistic.py
```

This writes `data/sample_daily.csv` deterministically (fixed seed).

## Datasets

- `data/sample_daily_demo.csv`: pipeline validation dataset (balanced-ish labels for Step 2/3).
- `data/sample_daily_realistic.csv`: market-like synthetic dataset (volatility clustering/fatter tails; positives may be lower).

Streamlit selector options:

- `Demo synthetic (balanced-ish)`
- `Realistic synthetic (market-like)`
- `Upload CSV`

## Run backtest

```bash
python -m src.run_backtest --csv data/sample_daily.csv --out out
```

Optional date filters:

```bash
python -m src.run_backtest --csv data/sample_daily.csv --out out --start 2024-03-01 --end 2024-10-31
```

## Outputs

The run writes:

- `out/trades.csv`
- `out/equity.csv`
- `out/chart.png`

Console summary prints:

- number of trades
- total return (%)
- win rate (%)
- max drawdown (%)

## Strategy (`strategy_001`) rules

Entry (all true on bar `t`):

- `xtrend[t] > xtrend[t-1]`, where `xtrend = EMA(close,20) - EMA(close,50)`
- `rsi[t] > rsi_sma[t]`, where `rsi = Wilder RSI(close,14)` and `rsi_sma = SMA(rsi,14)`
- `rsi_sma[t] > rsi_sma[t-1]`

Exit (any true on bar `t`):

- `xtrend[t] < xtrend[t-1]`
- `xtrend[t] < 0`
- `close[t] < atr_stop_21_4[t]`

ATR stop definition:

- `ATR21` uses Wilder smoothing over `ATR(high, low, close, 21)`
- `HH21 = rolling_max(high, 21)` including bar `t`
- `atr_stop_21_4 = HH21 - 4 * ATR21`

## Step 2 Labeling

Run TP/SL labeling:

```bash
python -m src.run_labeling --csv data/sample_daily.csv --out out --horizon 40 --tp 0.01 --sl 0.003
```

Optional date filters:

```bash
python -m src.run_labeling --csv data/sample_daily.csv --out out --horizon 40 --tp 0.01 --sl 0.003 --start 2024-03-01 --end 2024-10-31
```

Expected artifact:

- `out/labeled.csv`

`out/labeled.csv` contains (in order):

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `entry_price`
- `tp_level`
- `sl_level`
- `tp_hit`
- `sl_hit`
- `label`

Console summary prints:

- total labeled rows
- number of positives
- positive rate (%)

## Step 3 Training

Train baseline model:

```bash
python -m src.run_train_model --labeled out/labeled.csv --out out
```

Optional flags:

```bash
python -m src.run_train_model --labeled out/labeled.csv --out out --test_size 0.2 --threshold 0.5
```

Expected artifacts:

- `out/features.csv`
- `out/model.joblib`
- `out/predictions.csv`
- `out/metrics.json`

## Step 4 Model Zoo

Run model comparison and best-model selection:

```bash
python -m src.run_model_zoo --labeled out/labeled.csv --out out
```

Optional flags:

```bash
python -m src.run_model_zoo --labeled out/labeled.csv --out out --n_splits 5 --test_size 0.2 --threshold 0.5
```

Expected artifacts:

- `out/model_candidates.json`
- `out/best_model.joblib`
- `out/best_model_predictions.csv`
- `out/best_model_metrics.json`
- `out/best_model_card.md`

## Step 5 ML Backtest

Run ML-driven signal backtest with fixed TP/SL/horizon exits:

```bash
python -m src.run_backtest_ml --csv data/sample_daily_demo.csv --model out/best_model.joblib --out out --horizon 40 --tp 0.01 --sl 0.003 --threshold 0.5
```

Alternative input:

```bash
python -m src.run_backtest_ml --labeled out/labeled.csv --model out/best_model.joblib --out out --horizon 40 --tp 0.01 --sl 0.003 --threshold 0.5
```

Expected artifacts:

- `out/ml_signals.csv`
- `out/ml_trades.csv`
- `out/ml_equity.csv`
- `out/ml_chart.png`
- `out/ml_summary.json`

## Step 6 Threshold Policy

Search thresholds on train-only CV, freeze best threshold, and evaluate once on test:

```bash
python -m src.run_threshold_search --csv data/sample_daily_realistic.csv --model out/best_model.joblib --out out --horizon 40 --tp 0.01 --sl 0.003 --n_splits 5 --test_size 0.2
```

Expected artifacts:

- `out/threshold_search.csv`
- `out/best_threshold.json`
- `out/test_backtest_summary.json`
- `out/test_equity.csv`
- `out/test_trades.csv`
- `out/test_chart.png`

## Deploy to Streamlit Community Cloud

- Ensure `requirements.txt` exists and includes all dependencies.
- Set Python version in Streamlit app settings if needed.
- App entrypoint: `streamlit_app.py`
