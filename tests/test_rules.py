import pandas as pd

from src.engine.rules import evaluate_entry_exit_rules


def test_rule_eval_triggers_one_entry_and_one_exit():
    df = pd.DataFrame(
        {
            "e1_xtrend_increased": [False, True, True, False, False],
            "e2_rsi_above_sma": [False, True, True, True, False],
            "e3_rsi_sma_rising": [False, True, False, False, False],
            "x1_xtrend_decreased": [False, False, False, True, False],
            "x2_xtrend_below_zero": [False, False, False, False, False],
            "x3_close_below_atr_stop": [False, False, False, False, False],
        }
    )

    entry, exit_ = evaluate_entry_exit_rules(df)
    assert entry.sum() == 1
    assert exit_.sum() == 1
    assert entry[1]
    assert exit_[3]
