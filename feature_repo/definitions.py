from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

# 1. Define Entity
ticker_entity = Entity(
    name="ticker",
    value_type=ValueType.STRING,
    description="Stock/Asset Ticker Symbol"
)

# 2. Define Offline Source (Parquet derived from SQLite features_long table)
ta_indicators_source = FileSource(
    name="ta_indicators_source",
    path="data/features_wide.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define Feature View for 21 TA-Lib Indicators
ta_indicators_fv = FeatureView(
    name="ta_indicators_fv",
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
    ],
    online=True,
    source=ta_indicators_source,
)
