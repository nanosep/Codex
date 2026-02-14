from __future__ import annotations

import pandas as pd

from src.engine.rules import evaluate_entry_exit_rules
from src.indicators.atr import atr_wilder
from src.indicators.ema import ema
from src.indicators.rolling import rolling_max
from src.indicators.rsi import rsi_wilder
from src.indicators.sma import sma


def apply_strategy_001(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ema20 = ema(out["close"], 20)
    ema50 = ema(out["close"], 50)
    out["xtrend"] = ema20 - ema50

    out["rsi"] = rsi_wilder(out["close"], 14)
    out["rsi_sma"] = sma(out["rsi"], 14)

    out["atr21"] = atr_wilder(out["high"], out["low"], out["close"], 21)
    out["hh21"] = rolling_max(out["high"], 21)
    out["atr_stop_21_4"] = out["hh21"] - 4.0 * out["atr21"]

    out["e1_xtrend_increased"] = out["xtrend"] > out["xtrend"].shift(1)
    out["e2_rsi_above_sma"] = out["rsi"] > out["rsi_sma"]
    out["e3_rsi_sma_rising"] = out["rsi_sma"] > out["rsi_sma"].shift(1)

    out["x1_xtrend_decreased"] = out["xtrend"] < out["xtrend"].shift(1)
    out["x2_xtrend_below_zero"] = out["xtrend"] < 0
    out["x3_close_below_atr_stop"] = out["close"] < out["atr_stop_21_4"]

    required_cols = ["xtrend", "rsi", "rsi_sma", "atr21", "hh21", "atr_stop_21_4"]
    out["signals_ready"] = out[required_cols].notna().all(axis=1)

    entry_signal, exit_signal = evaluate_entry_exit_rules(out)
    out["entry_signal"] = out["signals_ready"] & entry_signal
    out["exit_signal"] = out["signals_ready"] & exit_signal
    return out
