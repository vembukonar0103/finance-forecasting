"""
feature_repo/features.py
------------------------
Feast Feature Store Definitions for Finance Forecasting Platform.

Covers:
1. ticker_entity: Join key ["ticker"], value_type STRING
2. technical_indicators_view: All 25 TA-Lib indicators
3. volatility_regime_view: ATR percentile, Bollinger Band width, RSI slope
4. kronos_pattern_view: Candlestick pattern label, confidence, forecast step
5. timesfm_summary_view: Multi-horizon expected return, confidence, prob_plus_5, p50
"""

import os
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# Ensure data directory exists inside feature_repo
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Define Entity
ticker_entity = Entity(
    name="ticker",
    join_keys=["ticker"],
    value_type=ValueType.STRING,
    description="Stock or crypto asset ticker symbol (e.g., AAPL, NVDA, BTC-USD)"
)

# 2. Define Offline Data Sources (Parquet files relative to feature_store.yaml)
technical_indicators_source = FileSource(
    name="technical_indicators_source",
    path=os.path.join("data", "technical_indicators.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

volatility_regime_source = FileSource(
    name="volatility_regime_source",
    path=os.path.join("data", "volatility_regime.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

kronos_pattern_source = FileSource(
    name="kronos_pattern_source",
    path=os.path.join("data", "kronos_patterns.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

timesfm_summary_source = FileSource(
    name="timesfm_summary_source",
    path=os.path.join("data", "timesfm_summary.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define Feature Views

# Technical Indicators View (25 TA-Lib indicators)
technical_indicators_view = FeatureView(
    name="technical_indicators_view",
    entities=[ticker_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="sma10", dtype=Float32),
        Field(name="sma20", dtype=Float32),
        Field(name="sma50", dtype=Float32),
        Field(name="sma200", dtype=Float32),
        Field(name="ema10", dtype=Float32),
        Field(name="ema20", dtype=Float32),
        Field(name="ema50", dtype=Float32),
        Field(name="ema200", dtype=Float32),
        Field(name="rsi", dtype=Float32),
        Field(name="macd", dtype=Float32),
        Field(name="macd_sig", dtype=Float32),
        Field(name="macd_hist", dtype=Float32),
        Field(name="bb_upper", dtype=Float32),
        Field(name="bb_middle", dtype=Float32),
        Field(name="bb_lower", dtype=Float32),
        Field(name="atr", dtype=Float32),
        Field(name="adx", dtype=Float32),
        Field(name="cci", dtype=Float32),
        Field(name="slowk", dtype=Float32),
        Field(name="slowd", dtype=Float32),
        Field(name="willr", dtype=Float32),
        Field(name="obv", dtype=Float32),
        Field(name="adosc", dtype=Float32),
        Field(name="volume_sma20", dtype=Float32),
        Field(name="relative_volume", dtype=Float32),
    ],
    online=True,
    source=technical_indicators_source,
)

# Volatility Regime View (Custom quantitative regime metrics)
volatility_regime_view = FeatureView(
    name="volatility_regime_view",
    entities=[ticker_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="bb_width", dtype=Float32),
        Field(name="atr_percentile_20d", dtype=Float32),
        Field(name="rsi_slope_5d", dtype=Float32),
    ],
    online=True,
    source=volatility_regime_source,
)

# Kronos Pattern View (Candlestick structural token inference)
kronos_pattern_view = FeatureView(
    name="kronos_pattern_view",
    entities=[ticker_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="pattern", dtype=String),
        Field(name="label", dtype=String),
        Field(name="confidence", dtype=Float32),
        Field(name="forecast_step", dtype=Int64),
    ],
    online=True,
    source=kronos_pattern_source,
)

# TimesFM Summary View (Multi-horizon probabilistic return summary)
timesfm_summary_view = FeatureView(
    name="timesfm_summary_view",
    entities=[ticker_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="expected_return_pct", dtype=Float32),
        Field(name="confidence_score", dtype=Float32),
        Field(name="prob_plus_5", dtype=Float32),
        Field(name="p50_final", dtype=Float32),
    ],
    online=True,
    source=timesfm_summary_source,
)
