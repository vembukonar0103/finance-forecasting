"""
tests/unit/test_circuit_breaker.py
-----------------------------------
Unit tests for circuit breaker resilience state machine in utils/circuit_breaker.py.
"""

import time
import pytest
from unittest.mock import patch
from utils.circuit_breaker import CircuitBreaker


def test_circuit_breaker_initial_closed_state():
    """Circuit breaker should initialize in CLOSED state (not open)."""
    cb = CircuitBreaker("test_service", failure_threshold=3, recovery_timeout_seconds=5)
    with patch("utils.circuit_breaker.is_redis_available", return_value=False):
        assert cb.is_open("item_1") is False


def test_circuit_breaker_trips_to_open_after_threshold():
    """Breaker should trip OPEN after reaching failure threshold."""
    cb = CircuitBreaker("test_service", failure_threshold=3, recovery_timeout_seconds=5)
    with patch("utils.circuit_breaker.is_redis_available", return_value=False):
        # 1st failure
        tripped = cb.record_failure("item_2")
        assert tripped is False
        assert cb.is_open("item_2") is False

        # 2nd failure
        tripped = cb.record_failure("item_2")
        assert tripped is False
        assert cb.is_open("item_2") is False

        # 3rd failure -> trips OPEN
        tripped = cb.record_failure("item_2")
        assert tripped is True
        assert cb.is_open("item_2") is True


def test_circuit_breaker_recovery_half_open():
    """After recovery timeout expires, circuit breaker enters HALF_OPEN, allowing a probe."""
    cb = CircuitBreaker("test_service", failure_threshold=2, recovery_timeout_seconds=0.1)
    with patch("utils.circuit_breaker.is_redis_available", return_value=False):
        cb.record_failure("item_3")
        cb.record_failure("item_3")
        assert cb.is_open("item_3") is True

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should now allow a probe (returns False)
        assert cb.is_open("item_3") is False

        # If probe succeeds, breaker resets to CLOSED
        cb.record_success("item_3")
        assert cb.is_open("item_3") is False


def test_circuit_breaker_reset_on_success():
    """record_success resets failures and closes the circuit."""
    cb = CircuitBreaker("test_service", failure_threshold=2, recovery_timeout_seconds=10)
    with patch("utils.circuit_breaker.is_redis_available", return_value=False):
        cb.record_failure("item_4")
        cb.record_success("item_4")
        # 1 more failure should not trip because count was reset
        tripped = cb.record_failure("item_4")
        assert tripped is False
        assert cb.is_open("item_4") is False
