"""
tests/unit/test_indicators.py
------------------------------
Unit tests for 25 TA-Lib technical indicators and volatility regime features
implemented in features/indicators.py.
"""

import pytest
import numpy as np
import pandas as pd
from features.indicators import build_features


@pytest.fixture
def sample_ohlcv_df():
    """Generates 250 bars of synthetic OHLCV market data."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    # Random walk close price
    returns = np.random.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0.005, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0.005, 0.005, n)))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, n)
    volume = np.random.uniform(100000, 5000000, n)

    df = pd.DataFrame({
        "timestamps": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return df


def test_build_features_all_columns_present(sample_ohlcv_df):
    """Verifies that all 25 TA-Lib indicators, 3 volatility features, and timestamps are generated."""
    feats = build_features(sample_ohlcv_df)

    expected_indicators = [
        'sma10', 'sma20', 'sma50', 'sma200',
        'ema10', 'ema20', 'ema50', 'ema200',
        'rsi', 'macd', 'macd_sig', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'adx', 'cci', 'slowk', 'slowd', 'willr',
        'obv', 'adosc', 'volume_sma20', 'relative_volume'
    ]
    volatility_features = ['bb_width', 'atr_percentile_20d', 'rsi_slope_5d']
    timestamp_cols = ['timestamp', 'event_timestamp', 'created_timestamp']

    for col in expected_indicators + volatility_features + timestamp_cols:
        assert col in feats.columns, f"Missing expected feature column: {col}"
    
    assert len(feats) == len(sample_ohlcv_df)


def test_indicator_mathematical_bounds(sample_ohlcv_df):
    """Verifies physical mathematical bounds on indicators."""
    feats = build_features(sample_ohlcv_df)

    # Valid tail values (post warm-up period)
    valid_tail = feats.iloc[200:]

    # RSI bounds [0, 100]
    assert (valid_tail['rsi'].dropna() >= 0).all()
    assert (valid_tail['rsi'].dropna() <= 100).all()

    # Bollinger Bands ordering
    assert (valid_tail['bb_upper'] >= valid_tail['bb_middle']).all()
    assert (valid_tail['bb_middle'] >= valid_tail['bb_lower']).all()
    assert (valid_tail['bb_width'] >= 0).all()

    # ATR strictly positive
    assert (valid_tail['atr'].dropna() > 0).all()

    # ATR percentile rank in [0, 1]
    assert (feats['atr_percentile_20d'] >= 0.0).all()
    assert (feats['atr_percentile_20d'] <= 1.0).all()


def test_datetime_index_support(sample_ohlcv_df):
    """Verifies build_features works when timestamps are set as the DataFrame index."""
    df_indexed = sample_ohlcv_df.set_index("timestamps")
    feats = build_features(df_indexed)
    assert "timestamp" in feats.columns
    assert "event_timestamp" in feats.columns
    assert len(feats) == len(sample_ohlcv_df)


def test_case_insensitivity(sample_ohlcv_df):
    """Verifies that column names with mixed casing are handled gracefully."""
    df_caps = sample_ohlcv_df.rename(columns={
        "timestamps": "Timestamps",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    feats = build_features(df_caps)
    assert "rsi" in feats.columns
    assert "macd" in feats.columns


def test_missing_timestamp_raises_keyerror():
    """Verifies that missing timestamp columns raise KeyError."""
    df_no_ts = pd.DataFrame({
        "open": [10, 11],
        "high": [12, 13],
        "low": [9, 10],
        "close": [11, 12],
        "volume": [100, 200]
    })
    with pytest.raises(KeyError):
        build_features(df_no_ts)


def test_zero_volume_resilience(sample_ohlcv_df):
    """Verifies that zero volume does not produce ZeroDivisionError in relative_volume."""
    df_zero_vol = sample_ohlcv_df.copy()
    df_zero_vol["volume"] = 0.0
    feats = build_features(df_zero_vol)
    assert not np.isinf(feats["relative_volume"]).any()
    assert not np.isnan(feats["relative_volume"].iloc[-1])
