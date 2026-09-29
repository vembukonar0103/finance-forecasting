"""
utils/circuit_breaker.py
------------------------
Resilience and Circuit Breaker Pattern for ML Inference and External APIs.

Phase 4 & 5 Deliverable:
- Tracks consecutive failures for critical dependencies (TimesFM, Kronos, Yahoo Finance).
- Automatically trips OPEN when threshold (e.g., 3 failures) is exceeded.
- Routes requests to fallbacks (e.g., BigQueryForecast or stale cache) instead of crashing.
- Attempts half-open recovery probes after a cooldown period.
"""

import time
import json
from typing import Optional, Callable, Any
from utils.cache import r as redis_client, is_redis_available


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 60.0
    ):
        self.service = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        
        # Local fallback state if Redis is offline
        self._local_failures = {}
        self._local_last_failure = {}
        self._local_state = {}

    def _get_keys(self, key_id: str):
        safe_id = key_id.replace(":", "_")
        prefix = f"cb:{self.service}:{safe_id}"
        return (
            f"{prefix}:failures",
            f"{prefix}:state",
            f"{prefix}:last_fail"
        )

    def is_open(self, key_id: str = "global") -> bool:
        """
        Checks if the circuit breaker is currently OPEN (tripped).
        If recovery timeout has expired, allows a probe (HALF_OPEN).
        """
        fail_key, state_key, last_fail_key = self._get_keys(key_id)
        now = time.time()

        if is_redis_available():
            try:
                state = redis_client.get(state_key)
                if state == "OPEN":
                    last_fail = float(redis_client.get(last_fail_key) or 0)
                    if now - last_fail >= self.recovery_timeout:
                        # Cooldown elapsed, enter HALF_OPEN
                        redis_client.set(state_key, "HALF_OPEN", ex=int(self.recovery_timeout))
                        return False
                    return True
                return False
            except Exception:
                pass

        # Fallback to local memory
        state = self._local_state.get(key_id, "CLOSED")
        if state == "OPEN":
            last_fail = self._local_last_failure.get(key_id, 0)
            if now - last_fail >= self.recovery_timeout:
                self._local_state[key_id] = "HALF_OPEN"
                return False
            return True
        return False

    def record_success(self, key_id: str = "global"):
        """Resets failure count on success."""
        fail_key, state_key, last_fail_key = self._get_keys(key_id)
        if is_redis_available():
            try:
                redis_client.delete(fail_key, state_key, last_fail_key)
            except Exception:
                pass

        self._local_failures[key_id] = 0
        self._local_state[key_id] = "CLOSED"

    def record_failure(self, key_id: str = "global") -> bool:
        """
        Records a failure. Trips circuit to OPEN if threshold reached.
        Returns True if breaker just tripped or is open.
        """
        fail_key, state_key, last_fail_key = self._get_keys(key_id)
        now = time.time()
        failures = 1

        if is_redis_available():
            try:
                failures = redis_client.incr(fail_key)
                redis_client.set(last_fail_key, str(now), ex=int(self.recovery_timeout * 2))
                if failures >= self.failure_threshold:
                    redis_client.set(state_key, "OPEN", ex=int(self.recovery_timeout * 2))
                    print(f"[Circuit Breaker TRIP] Service '{self.service}:{key_id}' tripped OPEN ({failures} failures).")
                    return True
                return False
            except Exception:
                pass

        failures = self._local_failures.get(key_id, 0) + 1
        self._local_failures[key_id] = failures
        self._local_last_failure[key_id] = now
        if failures >= self.failure_threshold:
            self._local_state[key_id] = "OPEN"
            print(f"[Circuit Breaker TRIP (local)] Service '{self.service}:{key_id}' tripped OPEN ({failures} failures).")
            return True
        return False


# Shared instances for core models and external data providers
timesfm_breaker = CircuitBreaker("timesfm", failure_threshold=3, recovery_timeout_seconds=60)
kronos_breaker = CircuitBreaker("kronos", failure_threshold=3, recovery_timeout_seconds=60)
yfinance_breaker = CircuitBreaker("yfinance", failure_threshold=4, recovery_timeout_seconds=90)
