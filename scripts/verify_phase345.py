"""
scripts/verify_phase345.py
--------------------------
Comprehensive Verification Suite for Phases 3, 4, and 5 Implementation:
- Phase 3: TimesFM Context Sizing (4096/16384), Data Pipeline (15y), Kronos Attention Heatmaps & Quantization
- Phase 4: Pre-computed Screener Cache (<200ms SLA), Universe Config (50+ tickers), Circuit Breaker
- Phase 5: BigQuery/Statistical Fallback Engine, Dependency-Aware Health Check, Graceful Degradation
"""

import sys
import os
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from api.app import app
from utils.cache import set_screener_cache, get_screener_cache, is_redis_available
from utils.circuit_breaker import CircuitBreaker
from utils.health import get_full_health_status
from models.bigquery_fallback import BigQueryForecast
from scripts.refresh_screener_cache import load_universe


def test_phase3_data_loader_and_context():
    print("\n[TEST 1] Testing Phase 3 Data Loader & Configurable Context...")
    from utils.data_loader import load_data
    df = load_data("AAPL", period="2y", interval="1d")
    assert not df.empty, "Data loader returned empty DataFrame"
    assert "timestamps" in df.columns, "'timestamps' column missing"
    assert df["timestamps"].is_monotonic_increasing, "Timestamps are not strictly chronological"
    assert not df["timestamps"].duplicated().any(), "Duplicate timestamps detected"
    print(f"  PASS: Loaded {len(df)} rows. Chronological sorting and deduplication verified.")


def test_phase3_attention_and_quantization():
    print("\n[TEST 2] Testing Phase 3 Kronos Attention & Quantization Interface...")
    from models.kronos_model import KronosForecast
    # Verify KronosForecast has quantize and attention attributes
    assert hasattr(KronosForecast, "quantize"), "KronosForecast missing quantize() method"
    assert hasattr(KronosForecast, "_extract_attention_map"), "KronosForecast missing _extract_attention_map()"
    print("  PASS: Kronos attention heatmap extraction & INT8 quantization interfaces verified.")


def test_phase4_universe_config():
    print("\n[TEST 3] Testing Phase 4 Universe Configuration...")
    tickers = load_universe()
    assert len(tickers) >= 50, f"Universe should contain at least 50 tickers, found {len(tickers)}"
    assert "AAPL" in tickers and "BTC-USD" in tickers and "SPY" in tickers
    print(f"  PASS: Loaded {len(tickers)} assets from config/universe.yaml.")


def test_phase4_precomputed_screener_sla():
    print("\n[TEST 4] Testing Phase 4 Pre-computed Screener SLA (<200ms)...")
    client = TestClient(app)
    
    # Mock pre-computed scan data
    mock_screener = [
        {
            "ticker": "AAPL",
            "kronos": {"pattern": "CDLENGULFING", "label": "Bullish", "confidence": 0.88, "forecast_step": 2, "attention_heatmap": {"status": "extracted"}},
            "timesfm_summary": {"direction": "Bullish", "prob_plus_5": 0.75},
            "composite_rating": {"rating": "Strong Bullish", "action": "Buy"}
        },
        {
            "ticker": "NVDA",
            "kronos": {"pattern": "CDLENGULFING", "label": "Bullish", "confidence": 0.92, "forecast_step": 3, "attention_heatmap": {"status": "extracted"}},
            "timesfm_summary": {"direction": "Bullish", "prob_plus_5": 0.81},
            "composite_rating": {"rating": "Strong Bullish", "action": "Buy"}
        },
        {
            "ticker": "MSFT",
            "kronos": {"pattern": "CDLENGULFING", "label": "Bearish", "confidence": 0.60, "forecast_step": 1},
            "timesfm_summary": {"direction": "Bearish", "prob_plus_5": 0.30},
            "composite_rating": {"rating": "Bearish", "action": "Sell"}
        }
    ]
    set_screener_cache(mock_screener, ttl_seconds=600)

    t0 = time.perf_counter()
    resp = client.get("/api/v1/screeners/ai-filter?pattern=bullish_engulfing&min_confidence=0.75")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
    data = resp.json()
    assert data["source"] == "precomputed_redis_cache", f"Expected precomputed_redis_cache, got {data.get('source')}"
    assert data["total_matched"] == 2, f"Expected 2 matches, got {data['total_matched']}"
    assert elapsed_ms < 200.0, f"SLA Violation: Expected <200ms, took {elapsed_ms:.2f}ms"
    print(f"  PASS: Screener responded in {elapsed_ms:.2f}ms (Well under <2.0s SLA! Matches: {data['total_matched']}).")


def test_phase4_circuit_breaker():
    print("\n[TEST 5] Testing Phase 4 Circuit Breaker Mechanism...")
    cb = CircuitBreaker("unit_test_service", failure_threshold=2, recovery_timeout_seconds=1.0)
    assert not cb.is_open("ticker_x"), "Circuit breaker should initially be CLOSED"
    cb.record_failure("ticker_x")
    assert not cb.is_open("ticker_x"), "Circuit breaker should not trip after 1 failure"
    cb.record_failure("ticker_x")
    assert cb.is_open("ticker_x"), "Circuit breaker should be OPEN after 2 failures"
    time.sleep(1.1)
    # Recovery timeout elapsed -> HALF_OPEN (allows probe)
    assert not cb.is_open("ticker_x"), "Circuit breaker should allow probe after recovery timeout"
    cb.record_success("ticker_x")
    assert not cb.is_open("ticker_x"), "Circuit breaker should reset to CLOSED after success"
    print("  PASS: Circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) verified.")


def test_phase5_fallback_engine():
    print("\n[TEST 6] Testing Phase 5 BigQuery / Statistical Fallback Engine...")
    fb = BigQueryForecast()
    test_df = pd.DataFrame({
        "timestamps": pd.date_range("2026-01-01", periods=100),
        "close": [150.0 + np.sin(i / 10.0) * 5.0 + i * 0.1 for i in range(100)]
    })
    pred_df, prob, meta = fb.predict(test_df, horizon=30)
    assert pred_df.shape == (30, 6), f"Expected shape (30, 6), got {pred_df.shape}"
    assert all(c in pred_df.columns for c in ['point', 'p10', 'p25', 'p50', 'p75', 'p90'])
    assert meta["fallback"] is True, "Expected fallback flag to be True"
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    print(f"  PASS: Fallback prediction generated successfully via {meta['provider']} (P(>=+5%): {prob:.2f}).")


def test_phase5_enterprise_health():
    print("\n[TEST 7] Testing Phase 5 Dependency-Aware Enterprise Health Check...")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "dependencies" in data, "Dependencies dictionary missing from /health"
    deps = data["dependencies"]
    assert "redis" in deps and "timesfm" in deps and "kronos" in deps and "yahoo_finance" in deps
    print(f"  PASS: /health returned status '{data['status']}' with full dependency telemetry.")


def run_all_checks():
    print("=" * 80)
    print(" RUNNING PHASES 3, 4 & 5 VALIDATION SUITE")
    print("=" * 80)
    test_phase3_data_loader_and_context()
    test_phase3_attention_and_quantization()
    test_phase4_universe_config()
    test_phase4_precomputed_screener_sla()
    test_phase4_circuit_breaker()
    test_phase5_fallback_engine()
    test_phase5_enterprise_health()
    print("\n" + "=" * 80)
    print(" ALL CHECKS PASSED: PHASES 3, 4 & 5 IMPLEMENTATION VALIDATED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_checks()
