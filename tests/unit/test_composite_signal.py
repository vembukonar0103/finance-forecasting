"""
tests/unit/test_composite_signal.py
------------------------------------
Unit tests for composite AI signal engine in utils/composite_signal.py.
"""

import pytest
from utils.composite_signal import compute_composite_rating


def test_bullish_confirmation():
    """Kronos Bullish Engulfing + TimesFM upward trend -> Strong Bullish confirmation."""
    kronos_res = {
        "pattern": "CDLENGULFING",
        "label": "Bullish",
        "confidence": 0.88,
        "forecast_step": 3
    }
    timesfm_res = {
        "horizons": {
            "horizon_30": {
                "point_forecast": [100.0, 102.0, 105.0],
                "probability_plus_5_pct": 0.75
            }
        }
    }
    result = compute_composite_rating(kronos_res, timesfm_res)
    assert result["ai_rating"] == "Strong Bullish"
    assert result["signal_status"] == "confirmation"
    assert result["timesfm_direction"] == "Bullish"
    assert result["kronos_confidence"] == 0.88


def test_bearish_confirmation():
    """Kronos Bearish Engulfing + TimesFM downward trend -> Strong Bearish confirmation."""
    kronos_res = {
        "pattern": "CDLENGULFING",
        "label": "Bearish",
        "confidence": 0.91,
        "forecast_step": 2
    }
    timesfm_res = {
        "horizons": {
            "horizon_30": {
                "point_forecast": [150.0, 145.0, 140.0],
                "probability_plus_5_pct": 0.12
            }
        }
    }
    result = compute_composite_rating(kronos_res, timesfm_res)
    assert result["ai_rating"] == "Strong Bearish"
    assert result["signal_status"] == "confirmation"
    assert result["timesfm_direction"] == "Bearish"


def test_conflicted_signal_bullish_kronos_bearish_timesfm():
    """Kronos Bullish + TimesFM Bearish -> Conflicted / Neutral."""
    kronos_res = {
        "pattern": "CDLENGULFING",
        "label": "Bullish",
        "confidence": 0.82
    }
    timesfm_res = {
        "horizons": {
            "horizon_30": {
                "point_forecast": [200.0, 195.0, 190.0],
                "probability_plus_5_pct": 0.15
            }
        }
    }
    result = compute_composite_rating(kronos_res, timesfm_res)
    assert result["ai_rating"] == "Conflicted / Neutral"
    assert result["signal_status"] == "conflict"


def test_no_pattern_falls_back_to_timesfm_direction():
    """Kronos No Pattern -> rating equals TimesFM direction."""
    kronos_res = {
        "pattern": "None",
        "label": "No Pattern",
        "confidence": 0.0
    }
    timesfm_res = {
        "horizons": {
            "horizon_30": {
                "point_forecast": [100.0, 104.0, 108.0],
                "probability_plus_5_pct": 0.65
            }
        }
    }
    result = compute_composite_rating(kronos_res, timesfm_res)
    assert result["ai_rating"] == "Bullish"
    assert result["signal_status"] == "neutral"


def test_empty_payloads_graceful_handling():
    """Empty or malformed dictionaries should not crash the engine."""
    result = compute_composite_rating({}, {})
    assert "ai_rating" in result
    assert result["signal_status"] == "neutral"
    assert result["timesfm_direction"] == "Neutral"
