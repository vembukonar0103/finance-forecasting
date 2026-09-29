"""
utils/cache.py
--------------
Redis caching wrapper using redis-py.

Key-Naming Conventions:
----------------------
- Kronos Inference:   kronos:{ticker}:{timeframe}      (e.g., kronos:AAPL:1d)
- TimesFM Inference:  timesfm:{ticker}:{horizon_days}  (e.g., timesfm:MSFT:30)
- Technical Indicators: indicators:{ticker}:{date}    (e.g., indicators:NVDA:2026-07-28)
"""

import os
import json
import time
from typing import Optional, Dict, Any
import redis

# Global Redis client instance (127.0.0.1 prevents Windows 2,000ms IPv6 loopback delay)
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=0, 
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5
)

_redis_healthy = True
_redis_last_check = 0.0

def is_redis_available() -> bool:
    """
    Checks if Redis is reachable. If unreachable, imposes a 10s cooldown
    to avoid stalling application threads on connection timeouts.
    """
    global _redis_healthy, _redis_last_check
    now = time.time()
    if not _redis_healthy and (now - _redis_last_check < 10.0):
        return False
    _redis_last_check = now
    try:
        r.ping()
        _redis_healthy = True
        return True
    except Exception:
        _redis_healthy = False
        return False


def get_cached(key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a JSON-decoded dictionary from Redis by key.
    Returns None if key does not exist, expired, or Redis offline.
    """
    if not is_redis_available():
        return None
    try:
        val = r.get(key)
        if val is None:
            return None
        return json.loads(val)
    except Exception as e:
        print(f"[Redis Cache Warning] get_cached failed for key '{key}': {e}")
        return None


def set_cached(key: str, value: Dict[str, Any], ttl_seconds: int = 300) -> bool:
    """
    JSON-encodes and stores a dictionary in Redis with specified TTL in seconds (default 300s).
    """
    if not is_redis_available():
        return False
    try:
        serialized = json.dumps(value)
        return bool(r.set(name=key, value=serialized, ex=ttl_seconds))
    except Exception as e:
        print(f"[Redis Cache Warning] set_cached failed for key '{key}': {e}")
        return False


SCREENER_CACHE_KEY = "screener:latest_results"


def get_screener_cache() -> Optional[list]:
    """
    Retrieves pre-computed screener results list from Redis.
    """
    if not is_redis_available():
        return None
    try:
        val = r.get(SCREENER_CACHE_KEY)
        if val is None:
            return None
        return json.loads(val)
    except Exception as e:
        print(f"[Redis Cache Warning] get_screener_cache failed: {e}")
        return None


def set_screener_cache(results: list, ttl_seconds: int = 600) -> bool:
    """
    Saves pre-computed screener results list to Redis with TTL (default 10 minutes).
    """
    if not is_redis_available():
        return False
    try:
        serialized = json.dumps(results)
        return bool(r.set(name=SCREENER_CACHE_KEY, value=serialized, ex=ttl_seconds))
    except Exception as e:
        print(f"[Redis Cache Warning] set_screener_cache failed: {e}")
        return False


def run_standalone_test():
    """
    Standalone test to verify setting, getting, and TTL expiration in Redis.
    """
    test_key = "test:kronos:AAPL:1d"
    test_data = {
        "ticker": "AAPL",
        "timeframe": "1d",
        "pattern": "CDLENGULFING",
        "label": "Bullish",
        "confidence": 0.85
    }

    print("=== Running Redis Cache Standalone Test ===")
    
    # 1. Set key with 2s TTL
    print(f"Setting key '{test_key}' with 2-second TTL...")
    set_success = set_cached(test_key, test_data, ttl_seconds=2)
    print(f"Set operation success: {set_success}")

    # 2. Immediate get
    cached_val = get_cached(test_key)
    print(f"Immediate get_cached result: {cached_val}")
    assert cached_val == test_data, "Error: Cached value does not match original data!"
    print("PASS: Immediate get returned matching data.")

    # 3. Wait 3 seconds to confirm TTL expiration
    print("Sleeping for 3 seconds to wait for TTL expiry...")
    time.sleep(3)

    # 4. Get after expiration
    expired_val = get_cached(test_key)
    print(f"Post-expiration get_cached result: {expired_val}")
    assert expired_val is None, "Error: Key did not expire as expected!"
    print("PASS: Key successfully expired after TTL.")
    print("=== All Cache Tests Passed Successfully ===")


if __name__ == "__main__":
    run_standalone_test()
