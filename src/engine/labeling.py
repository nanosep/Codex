from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "entry_price",
    "tp_level",
    "sl_level",
    "tp_hit",
    "sl_hit",
    "label",
]


def label_tp_sl_binary(df: pd.DataFrame, horizon: int, tp_pct: float, sl_pct: float) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if tp_pct <= 0:
        raise ValueError("tp_pct must be > 0")
    if sl_pct <= 0:
        raise ValueError("sl_pct must be > 0")

    if len(df) <= horizon:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = df.copy().reset_index(drop=True)
    out["entry_price"] = out["close"]
    out["tp_level"] = out["entry_price"] * (1.0 + tp_pct)
    out["sl_level"] = out["entry_price"] * (1.0 - sl_pct)

    tp_hits = []
    sl_hits = []
    labels = []
    last_labelable_index = len(out) - horizon

    for i in range(last_labelable_index):
        future = out.iloc[i + 1 : i + 1 + horizon]
        tp_hit = int((future["high"] >= out.at[i, "tp_level"]).any())
        sl_hit = int((future["low"] <= out.at[i, "sl_level"]).any())
        label = int(tp_hit == 1 and sl_hit == 0)
        tp_hits.append(tp_hit)
        sl_hits.append(sl_hit)
        labels.append(label)

    labeled = out.iloc[:last_labelable_index].copy()
    labeled["tp_hit"] = tp_hits
    labeled["sl_hit"] = sl_hits
    labeled["label"] = labels
    return labeled[OUTPUT_COLUMNS]
