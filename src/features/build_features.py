from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.atr import atr_wilder
from src.indicators.ema import ema
from src.indicators.rolling import rolling_max
from src.indicators.rsi import rsi_wilder
from src.indicators.sma import sma


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "hl_range",
    "close_vs_ema20",
    "close_vs_ema50",
    "vol_chg_1",
    "ema20",
    "ema50",
    "xtrend",
    "xtrend_slope_1",
    "rsi14",
    "rsi_sma14",
    "rsi_diff",
    "atr21",
    "atr_stop_21_4",
    "dist_to_atr_stop",
]


def build_features_from_labeled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["xtrend"] = out["ema20"] - out["ema50"]

    out["rsi14"] = rsi_wilder(out["close"], 14)
    out["rsi_sma14"] = sma(out["rsi14"], 14)
    out["rsi_diff"] = out["rsi14"] - out["rsi_sma14"]

    out["atr21"] = atr_wilder(out["high"], out["low"], out["close"], 21)
    hh21 = rolling_max(out["high"], 21)
    out["atr_stop_21_4"] = hh21 - 4.0 * out["atr21"]

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_5"] = out["close"].pct_change(5)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["close_vs_ema20"] = (out["close"] - out["ema20"]) / out["close"]
    out["close_vs_ema50"] = (out["close"] - out["ema50"]) / out["close"]
    out["vol_chg_1"] = out["volume"].replace(0, np.nan).pct_change(1)
    out["xtrend_slope_1"] = out["xtrend"].diff(1)
    out["dist_to_atr_stop"] = (out["close"] - out["atr_stop_21_4"]) / out["close"]

    features = out[["date"] + FEATURE_COLUMNS + ["label"]].copy()
    features["label"] = pd.to_numeric(features["label"], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    return features
