"""
tests/unit/test_explainability.py
----------------------------------
Unit tests for rule-based explainability in utils/explainability.py.
"""

import pytest
from utils.explainability import generate_reasoning


def test_rsi_oversold_and_overbought():
    """RSI levels generate appropriate explainability strings."""
    # Oversold
    reasons_low = generate_reasoning({"rsi": 22.5}, {})
    assert any("RSI Oversold (22.5)" in r for r in reasons_low)

    # Overbought
    reasons_high = generate_reasoning({"rsi": 78.1}, {})
    assert any("RSI Overbought (78.1)" in r for r in reasons_high)


def test_macd_bullish_and_bearish_crossovers():
    """MACD crossing signal line triggers Bullish/Bearish Crossover reasoning."""
    # Bullish crossover: prior MACD <= Sig, current MACD > Sig
    curr = {"macd": 1.5, "macd_sig": 1.2}
    prior = {"macd": 1.0, "macd_sig": 1.2}
    reasons_bull = generate_reasoning(curr, {}, prior_indicators=prior)
    assert "MACD Bullish Crossover" in reasons_bull

    # Bearish crossover: prior MACD >= Sig, current MACD < Sig
    curr_bear = {"macd": 0.8, "macd_sig": 1.1}
    prior_bear = {"macd": 1.2, "macd_sig": 1.1}
    reasons_bear = generate_reasoning(curr_bear, {}, prior_indicators=prior_bear)
    assert "MACD Bearish Crossover" in reasons_bear


def test_kronos_trend_confirmation_and_conflict():
    """Kronos pattern matching forecast direction triggers confirmation; opposite triggers conflict."""
    # Confirmation: TimesFM > 0 and Kronos Bullish
    forecast_bull = {
        "model_used": "TimesFM",
        "expected_return_pct": 0.05,
        "confidence_score": 0.85,
        "kronos_pattern": {"label": "Bullish Engulfing", "confidence": 0.90}
    }
    reasons_conf = generate_reasoning({}, forecast_bull)
    assert any("Kronos trend confirmation" in r for r in reasons_conf)

    # Conflict: TimesFM > 0 and Kronos Bearish
    forecast_conflict = {
        "model_used": "TimesFM",
        "expected_return_pct": 0.05,
        "confidence_score": 0.80,
        "kronos_pattern": {"label": "Bearish Engulfing", "confidence": 0.85}
    }
    reasons_conf = generate_reasoning({}, forecast_conflict)
    assert any("Kronos pattern conflicts" in r for r in reasons_conf)
