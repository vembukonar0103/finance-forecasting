"""
tests/unit/test_cache.py
-------------------------
Unit tests for Redis cache layer in utils/cache.py.
Tests serialization, cache retrieval, screener caching, and graceful offline fallback.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from utils.cache import (
    get_cached,
    set_cached,
    get_screener_cache,
    set_screener_cache,
    is_redis_available
)


def test_set_and_get_cached_mocked():
    """Verifies get_cached and set_cached serialization and retrieval."""
    test_key = "test:unit:ticker"
    payload = {"ticker": "NVDA", "forecast": [120.5, 122.0], "prob": 0.82}

    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps(payload)
    mock_redis.set.return_value = True

    with patch("utils.cache.r", mock_redis), patch("utils.cache.is_redis_available", return_value=True):
        success = set_cached(test_key, payload, ttl_seconds=60)
        assert success is True
        mock_redis.set.assert_called_once_with(name=test_key, value=json.dumps(payload), ex=60)

        data = get_cached(test_key)
        assert data == payload
        mock_redis.get.assert_called_once_with(test_key)


def test_get_cached_miss_returns_none():
    """Cache miss returns None without exception."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    with patch("utils.cache.r", mock_redis), patch("utils.cache.is_redis_available", return_value=True):
        result = get_cached("nonexistent:key")
        assert result is None


def test_screener_cache_roundtrip():
    """Tests get_screener_cache and set_screener_cache serialization."""
    screener_data = [
        {"ticker": "AAPL", "composite_rating": {"ai_rating": "Strong Bullish"}},
        {"ticker": "MSFT", "composite_rating": {"ai_rating": "Bullish"}}
    ]
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps(screener_data)
    mock_redis.set.return_value = True

    with patch("utils.cache.r", mock_redis), patch("utils.cache.is_redis_available", return_value=True):
        set_ok = set_screener_cache(screener_data, ttl_seconds=300)
        assert set_ok is True

        cached = get_screener_cache()
        assert cached == screener_data
        assert len(cached) == 2


def test_redis_offline_graceful_degradation():
    """When Redis is unavailable, get and set operations return None/False without raising exceptions."""
    with patch("utils.cache.is_redis_available", return_value=False):
        assert get_cached("any_key") is None
        assert set_cached("any_key", {"data": 1}) is False
        assert get_screener_cache() is None
        assert set_screener_cache([1, 2]) is False


def test_corrupted_json_in_cache():
    """If Redis returns corrupted/malformed JSON, get_cached returns None safely."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = "{malformed json:"

    with patch("utils.cache.r", mock_redis), patch("utils.cache.is_redis_available", return_value=True):
        val = get_cached("corrupted:key")
        assert val is None


def test_is_redis_available_logic():
    """Tests is_redis_available ping behavior."""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch("utils.cache.r", mock_redis):
        assert is_redis_available() is True

    # When ping fails
    mock_fail = MagicMock()
    mock_fail.ping.side_effect = Exception("Connection refused")
    with patch("utils.cache.r", mock_fail):
        # Reset check time
        with patch("utils.cache._redis_last_check", 0.0):
            assert is_redis_available() is False

