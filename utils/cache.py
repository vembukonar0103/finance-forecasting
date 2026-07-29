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

import json
import time
from typing import Optional, Dict, Any
import redis

# Global Redis client instance
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def get_cached(key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a JSON-decoded dictionary from Redis by key.
    Returns None if key does not exist or is expired.
    """
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
    try:
        serialized = json.dumps(value)
        return bool(r.set(name=key, value=serialized, ex=ttl_seconds))
    except Exception as e:
        print(f"[Redis Cache Warning] set_cached failed for key '{key}': {e}")
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
