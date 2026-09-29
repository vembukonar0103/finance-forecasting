"""
tests/integration/test_api.py
------------------------------
Integration tests for FastAPI endpoints:
- GET /health
- GET /metrics
- GET /features/{ticker}
- GET /api/v1/screeners/ai-filter
- POST /predict/batch
- Mocked GET /predict/{ticker}, /predict/timesfm/{ticker}, /predict/kronos/{ticker}
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Verifies dependency health check endpoint returns 200 and expected status keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "dependencies" in data
    assert "redis" in data["dependencies"]
    assert "timesfm" in data["dependencies"]
    assert "kronos" in data["dependencies"]
    assert "yahoo_finance" in data["dependencies"]


def test_metrics_endpoint():
    """Verifies Prometheus exposition metrics endpoint returns 200 with metrics text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "inference_requests_total" in text
    assert "inference_latency_seconds" in text


def test_features_endpoint_404_for_unknown_ticker():
    """Non-existent ticker in feature store returns 404."""
    with patch("api.app.get_online_features", return_value={}):
        response = client.get("/features/NONEXISTENT999")
        assert response.status_code == 404


def test_features_endpoint_success():
    """Retrieved online features return 200 with valid schema."""
    fake_features = {"rsi": 55.4, "macd": 1.25, "bb_width": 0.08}
    with patch("api.app.get_online_features", return_value=fake_features):
        response = client.get("/features/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["features_count"] == 3
        assert data["features"]["rsi"] == 55.4


def test_screener_ai_filter_endpoint_precomputed():
    """Screener returns matching assets from pre-computed cache."""
    cached_mock = [
        {
            "ticker": "AAPL",
            "kronos": {
                "pattern": "CDLENGULFING",
                "label": "Bullish",
                "confidence": 0.85,
                "forecast_step": 3
            },
            "timesfm_summary": {
                "direction": "Bullish",
                "prob_plus_5": 0.72
            },
            "composite_rating": {
                "ai_rating": "Strong Bullish"
            }
        },
        {
            "ticker": "MSFT",
            "kronos": {
                "pattern": "CDLENGULFING",
                "label": "Bearish",
                "confidence": 0.40,
                "forecast_step": 1
            },
            "timesfm_summary": {
                "direction": "Bearish",
                "prob_plus_5": 0.10
            },
            "composite_rating": {
                "ai_rating": "Neutral"
            }
        }
    ]
    with patch("api.app.get_screener_cache", return_value=cached_mock):
        response = client.get("/api/v1/screeners/ai-filter?pattern=bullish_engulfing&min_confidence=0.75")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "precomputed_redis_cache"
        assert data["total_scanned"] == 2
        assert data["total_matched"] == 1
        assert data["matches"][0]["ticker"] == "AAPL"


def test_batch_prediction_validation_empty_fails():
    """Batch prediction requires at least 1 ticker; empty list triggers 422 Unprocessable Entity."""
    response = client.post("/predict/batch", json={"tickers": []})
    assert response.status_code == 422


def test_predict_kronos_endpoint_cached():
    """Dedicated Kronos endpoint retrieves cached response without model recomputation."""
    mock_kronos_data = {
        "ticker": "AAPL",
        "timeframe": "1d",
        "pattern": "CDLENGULFING",
        "label": "Bullish",
        "confidence": 0.88,
        "forecast_step": 3,
        "tokenization_ms": 12.5,
        "cache_hit": False,
        "attention_heatmap": {"matrix": [[0.1, 0.2], [0.3, 0.4]]}
    }
    with patch("api.app.kronos_model", MagicMock()), patch("api.app.get_cached", return_value=mock_kronos_data):
        response = client.get("/predict/kronos/AAPL?timeframe=1d")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["confidence"] == 0.88
        assert data["cache_hit"] is True
        assert data["attention_heatmap"] == {"matrix": [[0.1, 0.2], [0.3, 0.4]]}


def test_predict_timesfm_endpoint_cached():
    """Dedicated TimesFM endpoint retrieves multi-horizon fan charts from cache."""
    mock_horizon_30 = {
        "timestamps": ["2026-08-01", "2026-08-02"],
        "point_forecast": [100.0, 102.0],
        "p10": [98.0, 99.0],
        "p25": [99.0, 100.5],
        "p50": [100.0, 102.0],
        "p75": [101.0, 103.5],
        "p90": [102.0, 105.0],
        "probability_plus_5_pct": 0.65,
        "inference_ms": 22.0,
        "cache_hit": False,
        "fallback": False,
        "provider": "timesfm"
    }
    with patch("api.app.timesfm_model", MagicMock()), patch("api.app.get_cached", return_value=mock_horizon_30):
        response = client.get("/predict/timesfm/AAPL?horizons=30")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "horizon_30" in data["horizons"]
        assert data["horizons"]["horizon_30"]["cache_hit"] is True
        assert data["horizons"]["horizon_30"]["probability_plus_5_pct"] == 0.65

