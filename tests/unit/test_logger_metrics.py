"""
tests/unit/test_logger_metrics.py
----------------------------------
Unit tests for structured JSON logger and Prometheus metrics definitions.
"""

import json
import logging
import pytest
from utils.logger import JsonFormatter, setup_logger
from utils.metrics import (
    REQUESTS_TOTAL,
    INFERENCE_LATENCY,
    CACHE_ACCESS_TOTAL,
    SCREENER_MATCH_COUNT,
    metrics_endpoint_response
)


def test_json_formatter_valid_json():
    """Verifies JsonFormatter outputs parseable JSON containing standard fields."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=20,
        msg="Test log message",
        args=(),
        exc_info=None
    )
    record.endpoint = "/predict/AAPL"
    record.status_code = 200
    record.latency_ms = 45.2

    output_str = formatter.format(record)
    log_obj = json.loads(output_str)

    assert log_obj["level"] == "INFO"
    assert log_obj["message"] == "Test log message"
    assert log_obj["endpoint"] == "/predict/AAPL"
    assert log_obj["status_code"] == 200
    assert log_obj["latency_ms"] == 45.2
    assert "timestamp" in log_obj


def test_prometheus_metrics_instrumentation():
    """Verifies that Prometheus metrics can be recorded and exported."""
    # Increment counter
    REQUESTS_TOTAL.labels(endpoint="/health", status="200").inc()

    # Time latency histogram
    with INFERENCE_LATENCY.labels(model="timesfm").time():
        _ = 1 + 1

    # Cache hit counter
    CACHE_ACCESS_TOTAL.labels(service="unified", hit="true").inc()

    # Screener match gauge
    SCREENER_MATCH_COUNT.set(5)

    # Scrape metrics response
    resp = metrics_endpoint_response()
    content = resp.body.decode("utf-8")

    assert "# HELP inference_requests_total" in content
    assert "# HELP inference_latency_seconds" in content
    assert "screener_match_count 5.0" in content
