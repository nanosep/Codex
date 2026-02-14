from __future__ import annotations

import pandas as pd


def evaluate_entry_exit_rules(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    entry = (df["e1_xtrend_increased"] & df["e2_rsi_above_sma"] & df["e3_rsi_sma_rising"]).fillna(False)
    exit_ = (df["x1_xtrend_decreased"] | df["x2_xtrend_below_zero"] | df["x3_close_below_atr_stop"]).fillna(False)
    return entry.astype(bool), exit_.astype(bool)
