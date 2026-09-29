"""
tests/unit/test_trading_strategy.py
------------------------------------
Unit tests for 5-state trading signal engine in utils/trading_strategy.py.
"""

import pytest
import pandas as pd
from utils.trading_strategy import generate_signal, generate_signal_v2, SIGNAL_THRESHOLDS


def test_generate_signal_v2_confidence_gating():
    """Signals should revert to 'Hold' if confidence is below min_confidence threshold."""
    # Strong positive return but confidence too low
    signal = generate_signal_v2(
        expected_return_pct=0.08,
        confidence_score=0.40,
        min_confidence=0.65
    )
    assert signal == "Hold"


def test_generate_signal_v2_strong_buy():
    """Expected return exceeding strong_buy threshold yields 'Strong Buy'."""
    signal = generate_signal_v2(
        expected_return_pct=SIGNAL_THRESHOLDS["strong_buy"] + 0.01,
        confidence_score=0.85,
        min_confidence=0.65
    )
    assert signal == "Strong Buy"


def test_generate_signal_v2_buy():
    """Expected return between buy and strong_buy thresholds yields 'Buy'."""
    mid_return = (SIGNAL_THRESHOLDS["buy"] + SIGNAL_THRESHOLDS["strong_buy"]) / 2
    signal = generate_signal_v2(
        expected_return_pct=mid_return,
        confidence_score=0.85,
        min_confidence=0.65
    )
    assert signal == "Buy"


def test_generate_signal_v2_strong_sell():
    """Expected return below strong_sell threshold yields 'Strong Sell'."""
    signal = generate_signal_v2(
        expected_return_pct=SIGNAL_THRESHOLDS["strong_sell"] - 0.01,
        confidence_score=0.90,
        min_confidence=0.65
    )
    assert signal == "Strong Sell"


def test_generate_signal_v2_sell():
    """Expected return between sell and strong_sell thresholds yields 'Sell'."""
    mid_return = (SIGNAL_THRESHOLDS["sell"] + SIGNAL_THRESHOLDS["strong_sell"]) / 2
    signal = generate_signal_v2(
        expected_return_pct=mid_return,
        confidence_score=0.80,
        min_confidence=0.65
    )
    assert signal == "Sell"


def test_generate_signal_v2_neutral_hold():
    """Small return within neutral deadband yields 'Hold'."""
    signal = generate_signal_v2(
        expected_return_pct=0.001,
        confidence_score=0.95,
        min_confidence=0.65
    )
    assert signal == "Hold"


def test_generate_signal_v1_empty_df():
    """Empty dataframe produces 0 (flat/hold)."""
    assert generate_signal(pd.DataFrame(), c0=100.0) == 0


def test_generate_signal_v1_buy_and_sell():
    """Tests v1 integer signals (+1, -1, 0)."""
    # Buy scenario
    pred_buy = pd.DataFrame({"point": [100.0, 105.0], "p10": [98.0, 103.0], "p90": [102.0, 107.0]})
    sig_buy = generate_signal(pred_buy, c0=100.0, threshold=0.02, min_confidence=0.40)
    assert sig_buy == 1

    # Sell scenario
    pred_sell = pd.DataFrame({"point": [100.0, 95.0], "p10": [93.0, 93.0], "p90": [102.0, 97.0]})
    sig_sell = generate_signal(pred_sell, c0=100.0, threshold=0.02, min_confidence=0.40)
    assert sig_sell == -1
