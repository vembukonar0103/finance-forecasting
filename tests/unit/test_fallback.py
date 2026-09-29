"""
tests/unit/test_fallback.py
----------------------------
Unit tests for BigQuery and statistical drift fallback forecasting engine.
"""

import pytest
import numpy as np
import pandas as pd
from models.bigquery_fallback import BigQueryForecast


@pytest.fixture
def sample_history():
    """Generates 100 days of price history."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = 150.0 + np.cumsum(np.random.normal(0.1, 1.0, 100))
    return pd.DataFrame({
        "timestamps": dates,
        "close": prices
    })


def test_statistical_fallback_forecast_schema(sample_history):
    """Verifies that fallback produces all 6 probability bands and valid probability."""
    fallback = BigQueryForecast()
    horizon = 30
    pred_df, prob_plus_5, meta = fallback.predict(sample_history, target_col="close", horizon=horizon)

    # 1. Output length matches horizon
    assert len(pred_df) == horizon

    # 2. Required quantile columns
    expected_cols = ["point", "p10", "p25", "p50", "p75", "p90"]
    for col in expected_cols:
        assert col in pred_df.columns, f"Missing quantile column: {col}"

    # 3. Monotonic ordering of quantile bands
    assert (pred_df["p10"] <= pred_df["p25"]).all(), "p10 must be <= p25"
    assert (pred_df["p25"] <= pred_df["p50"]).all(), "p25 must be <= p50"
    assert (pred_df["p50"] <= pred_df["p75"]).all(), "p50 must be <= p75"
    assert (pred_df["p75"] <= pred_df["p90"]).all(), "p75 must be <= p90"

    # 4. Probability bounds
    assert 0.0 <= prob_plus_5 <= 1.0

    # 5. Metadata verification
    assert meta["fallback"] is True
    assert "provider" in meta


def test_fallback_datetime_index(sample_history):
    """Verifies fallback works when DataFrame uses DatetimeIndex instead of timestamps column."""
    df_indexed = sample_history.set_index("timestamps")
    fallback = BigQueryForecast()
    pred_df, prob_plus_5, meta = fallback.predict(df_indexed, target_col="close", horizon=14)

    assert len(pred_df) == 14
    assert not pred_df.empty
    assert meta["fallback"] is True


def test_fallback_constant_prices():
    """Fallback should handle flat price series without numerical collapse or ZeroDivision."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    flat_df = pd.DataFrame({"timestamps": dates, "close": [100.0] * 50})
    fallback = BigQueryForecast()
    pred_df, prob_plus_5, meta = fallback.predict(flat_df, target_col="close", horizon=10)

    assert len(pred_df) == 10
    assert not np.isnan(pred_df["point"]).any()
    assert 0.0 <= prob_plus_5 <= 1.0
