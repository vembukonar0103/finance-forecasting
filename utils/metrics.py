"""
utils/metrics.py
----------------
Prometheus Metrics Telemetry for Unified Finance Forecasting Platform.

Phase 6 Deliverable:
- Tracks inference latency histograms categorized by foundation model (TimesFM, Kronos, Fallback).
- Tracks request counters by endpoint and HTTP status.
- Tracks Redis cache hit ratios.
- Tracks live screener match counts.
- Generates standard Prometheus exposition format via /metrics.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from fastapi import Response

# 1. Inference Latency Histogram
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# 2. Total Requests Counter
REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total HTTP requests received",
    ["endpoint", "status"]
)

# 3. Cache Hits Counter
CACHE_ACCESS_TOTAL = Counter(
    "cache_access_total",
    "Total Redis cache queries",
    ["service", "hit"]
)

# 4. Screener Match Count Gauge
SCREENER_MATCH_COUNT = Gauge(
    "screener_match_count",
    "Number of asset matches in the latest screener scan"
)


def metrics_endpoint_response() -> Response:
    """Generates the standard Prometheus scrape response."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
