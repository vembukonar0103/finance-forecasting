"""
utils/health.py
---------------
Dependency-Aware Health & Readiness Checker.

Phase 5 (Task 6B & 6C) Deliverable:
- Checks operational status and latency for Redis, TimesFM, Kronos, and Yahoo Finance.
- Detects degraded states (e.g., Redis down, circuit breakers open, model initialization failures).
- Provides graceful degradation metadata for the /health endpoint.
"""

import time
from typing import Dict, Any, Optional
from utils.cache import r as redis_client
from utils.circuit_breaker import timesfm_breaker, kronos_breaker, yfinance_breaker


def check_redis_dependency() -> Dict[str, Any]:
    """Tests Redis connectivity and ping latency."""
    if not redis_client:
        return {"status": "down", "error": "Redis client not initialized"}
    
    t0 = time.perf_counter()
    try:
        ok = redis_client.ping()
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)
        return {
            "status": "up" if ok else "degraded",
            "latency_ms": latency_ms,
            "connected": bool(ok)
        }
    except Exception as e:
        return {
            "status": "down",
            "latency_ms": None,
            "error": str(e)
        }


def check_timesfm_dependency(timesfm_model: Any) -> Dict[str, Any]:
    """Inspects TimesFM model state, device, context, and circuit breaker."""
    if timesfm_model is None:
        return {"status": "down", "loaded": False, "error": "Model not loaded"}

    is_open = timesfm_breaker.is_open("global")
    return {
        "status": "degraded" if is_open else "up",
        "loaded": True,
        "device": str(getattr(timesfm_model, "device", "unknown")),
        "max_context": getattr(timesfm_model, "default_max_context", 4096),
        "fp16": getattr(timesfm_model, "fp16", False),
        "circuit_breaker": "OPEN" if is_open else "CLOSED"
    }


def check_kronos_dependency(kronos_model: Any) -> Dict[str, Any]:
    """Inspects Kronos model state, device, quantization, and circuit breaker."""
    if kronos_model is None:
        return {"status": "down", "loaded": False, "error": "Model not loaded"}

    is_open = kronos_breaker.is_open("global")
    return {
        "status": "degraded" if is_open else "up",
        "loaded": True,
        "device": str(getattr(kronos_model, "device", "unknown")),
        "quantized": getattr(kronos_model, "is_quantized", False),
        "circuit_breaker": "OPEN" if is_open else "CLOSED"
    }


def check_yfinance_dependency() -> Dict[str, Any]:
    """Checks Yahoo Finance upstream availability status."""
    is_open = yfinance_breaker.is_open("global")
    return {
        "status": "degraded" if is_open else "up",
        "circuit_breaker": "OPEN" if is_open else "CLOSED",
        "provider": "yfinance"
    }


def get_full_health_status(timesfm_model: Any, kronos_model: Any) -> Dict[str, Any]:
    """Aggregates all system dependencies into an enterprise health response."""
    redis_info = check_redis_dependency()
    tfm_info = check_timesfm_dependency(timesfm_model)
    kronos_info = check_kronos_dependency(kronos_model)
    yf_info = check_yfinance_dependency()

    # Determine overall system health
    if tfm_info["status"] == "down" or kronos_info["status"] == "down":
        overall = "unhealthy"
    elif redis_info["status"] == "down" or tfm_info["status"] == "degraded" or kronos_info["status"] == "degraded":
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "dependencies": {
            "redis": redis_info,
            "timesfm": tfm_info,
            "kronos": kronos_info,
            "yahoo_finance": yf_info
        }
    }
