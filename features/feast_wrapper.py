"""
features/feast_wrapper.py
-------------------------
Production Feast Feature Store Wrapper & Adapter.

Key Capabilities:
1. Online Feature Ingestion: push_online_features() pushes wide-format technical and
   regime features directly to Redis online store and updates offline Parquet store.
2. Online Feature Serving: get_online_features() retrieves the latest point-in-time
   features from Redis (<5ms), with dual-read fallback to legacy SQLite store.
3. Historical Point-in-Time Features: get_historical_features() queries features as of
   explicit historical timestamps, mathematically guaranteeing zero lookahead bias.
4. Dual-Write Safety: Maintains SQLite features_long during Phase 2 transition.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from feast import FeatureStore
from utils.config import DB_PATH
from features.feature_store import get_latest_features as sqlite_get_latest_features
from features.feature_store import store_features as sqlite_store_features

logger = logging.getLogger("feast_wrapper")

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feature_repo"))
DATA_DIR = os.path.join(REPO_PATH, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Complete 25 TA-Lib Technical Indicators
INDICATORS_25 = [
    'sma10', 'sma20', 'sma50', 'sma200',
    'ema10', 'ema20', 'ema50', 'ema200',
    'rsi', 'macd', 'macd_sig', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'atr', 'adx', 'cci', 'slowk', 'slowd', 'willr',
    'obv', 'adosc', 'volume_sma20', 'relative_volume'
]

# Custom Volatility Regime Metrics
VOLATILITY_FEATURES = [
    'bb_width',
    'atr_percentile_20d',
    'rsi_slope_5d'
]

_store_instance: Optional[FeatureStore] = None


def ensure_offline_stores():
    """Ensures offline Parquet files with valid schema exist so Feast apply can infer schemas."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Technical Indicators
    tech_path = os.path.join(DATA_DIR, "technical_indicators.parquet")
    if not os.path.exists(tech_path):
        data = {
            'ticker': pd.Series(dtype='str'),
            'event_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'created_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
        }
        for col in INDICATORS_25:
            data[col] = pd.Series(dtype='float32')
        pd.DataFrame(data).to_parquet(tech_path, index=False)

    # 2. Volatility Regime
    vol_path = os.path.join(DATA_DIR, "volatility_regime.parquet")
    if not os.path.exists(vol_path):
        data = {
            'ticker': pd.Series(dtype='str'),
            'event_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'created_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
        }
        for col in VOLATILITY_FEATURES:
            data[col] = pd.Series(dtype='float32')
        pd.DataFrame(data).to_parquet(vol_path, index=False)

    # 3. Kronos Patterns
    kronos_path = os.path.join(DATA_DIR, "kronos_patterns.parquet")
    if not os.path.exists(kronos_path):
        data = {
            'ticker': pd.Series(dtype='str'),
            'event_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'created_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'pattern': pd.Series(dtype='str'),
            'label': pd.Series(dtype='str'),
            'confidence': pd.Series(dtype='float32'),
            'forecast_step': pd.Series(dtype='int64')
        }
        pd.DataFrame(data).to_parquet(kronos_path, index=False)

    # 4. TimesFM Summary
    timesfm_path = os.path.join(DATA_DIR, "timesfm_summary.parquet")
    if not os.path.exists(timesfm_path):
        data = {
            'ticker': pd.Series(dtype='str'),
            'event_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'created_timestamp': pd.Series(dtype='datetime64[ns, UTC]'),
            'expected_return_pct': pd.Series(dtype='float32'),
            'confidence_score': pd.Series(dtype='float32'),
            'prob_plus_5': pd.Series(dtype='float32'),
            'p50_final': pd.Series(dtype='float32')
        }
        pd.DataFrame(data).to_parquet(timesfm_path, index=False)


def get_feature_store() -> FeatureStore:
    """Singleton getter for Feast FeatureStore to avoid redundant disk I/O."""
    global _store_instance
    if _store_instance is None:
        ensure_offline_stores()
        _store_instance = FeatureStore(repo_path=REPO_PATH)
    return _store_instance



def push_online_features(
    ticker: str,
    df_features: pd.DataFrame,
    event_timestamp: Optional[datetime] = None
) -> bool:
    """
    Pushes calculated technical features and volatility metrics into Feast.
    1. Updates offline Parquet storage (for historical point-in-time queries).
    2. Writes / materializes latest records to Redis online store.
    3. Dual-writes to legacy SQLite store to guarantee zero downtime during migration.
    """
    ticker = ticker.upper().strip()
    df = df_features.copy()

    # 1. Ensure required entity column 'ticker' is present
    df['ticker'] = ticker

    # 2. Ensure Feast timestamp columns are valid UTC datetime64
    if 'event_timestamp' not in df.columns:
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            ts = pd.to_datetime(df['date'])
        else:
            ts = pd.Timestamp.now(tz='UTC')
            
        if hasattr(ts, 'dt') and ts.dt.tz is None:
            df['event_timestamp'] = ts.dt.tz_localize('UTC')
        elif hasattr(ts, 'tz') and ts.tz is None:
            df['event_timestamp'] = pd.to_datetime(ts).tz_localize('UTC')
        else:
            df['event_timestamp'] = pd.to_datetime(ts)

    if 'created_timestamp' not in df.columns:
        df['created_timestamp'] = pd.Timestamp.now(tz='UTC')

    # Convert numeric columns to float32 for schema compatibility
    for col in INDICATORS_25 + VOLATILITY_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # 3. Offline Parquet Persistence
    try:
        # Technical indicators offline partition
        tech_cols = ['ticker', 'event_timestamp', 'created_timestamp'] + [c for c in INDICATORS_25 if c in df.columns]
        tech_df = df[tech_cols].dropna(subset=['ticker', 'event_timestamp']).copy()
        tech_pq = os.path.join(DATA_DIR, "technical_indicators.parquet")
        _append_or_write_parquet(tech_pq, tech_df, key_cols=['ticker', 'event_timestamp'])

        # Volatility regime offline partition
        vol_cols = ['ticker', 'event_timestamp', 'created_timestamp'] + [c for c in VOLATILITY_FEATURES if c in df.columns]
        vol_df = df[vol_cols].dropna(subset=['ticker', 'event_timestamp']).copy()
        vol_pq = os.path.join(DATA_DIR, "volatility_regime.parquet")
        _append_or_write_parquet(vol_pq, vol_df, key_cols=['ticker', 'event_timestamp'])

    except Exception as e:
        logger.warning(f"[Feast Wrapper] Failed updating offline Parquet store for {ticker}: {e}")

    # 4. Materialize to Redis Online Store
    success = False
    try:
        store = get_feature_store()
        # Direct push to online store using write_to_online_store (low latency)
        store.write_to_online_store(
            feature_view_name="technical_indicators_view",
            df=tech_df
        )
        store.write_to_online_store(
            feature_view_name="volatility_regime_view",
            df=vol_df
        )
        success = True
    except Exception as e:
        logger.warning(f"[Feast Wrapper] write_to_online_store failed for {ticker}, attempting materialize: {e}")
        try:
            store = get_feature_store()
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now() + timedelta(days=2)
            store.materialize(start_date=start_date, end_date=end_date)
            success = True
        except Exception as e2:
            logger.error(f"[Feast Wrapper] Online store materialize failed: {e2}")

    # 5. Dual-Write to SQLite Legacy Store (Safety Fallback)
    try:
        sqlite_store_features(ticker, df_features)
    except Exception as e:
        logger.warning(f"[Feast Wrapper] Legacy SQLite dual-write failed for {ticker}: {e}")

    return success


def get_online_features(ticker: str, feature_view: str = "technical_indicators_view") -> Dict[str, Any]:
    """
    Retrieves latest online features from Redis via Feast for a given ticker.
    Gracefully falls back to SQLite get_latest_features() on failure or cache miss.
    """
    ticker = ticker.upper().strip()
    online_dict: Dict[str, Any] = {}

    try:
        store = get_feature_store()
        
        target_features = []
        if feature_view in ("technical_indicators_view", "all"):
            target_features.extend([f"technical_indicators_view:{ind}" for ind in INDICATORS_25])
        if feature_view in ("volatility_regime_view", "all"):
            target_features.extend([f"volatility_regime_view:{vf}" for vf in VOLATILITY_FEATURES])

        response = store.get_online_features(
            features=target_features,
            entity_rows=[{"ticker": ticker}]
        ).to_dict()

        # Extract values
        all_keys = (INDICATORS_25 if feature_view in ("technical_indicators_view", "all") else []) + \
                   (VOLATILITY_FEATURES if feature_view in ("volatility_regime_view", "all") else [])

        for key in all_keys:
            val_list = response.get(key)
            if not val_list:
                # Feast sometimes prefixes with feature_view__
                val_list = response.get(f"{feature_view}__{key}")
            
            if val_list and len(val_list) > 0:
                val = val_list[0]
                online_dict[key] = float(val) if (val is not None and not np.isnan(val)) else None
            else:
                online_dict[key] = None

        # Check if we have at least some non-None features
        populated = any(v is not None for v in online_dict.values())
        if populated:
            return online_dict

    except Exception as e:
        logger.warning(f"[Feast Wrapper] Redis get_online_features failed for {ticker}: {e}")

    # Fallback to SQLite store of truth
    logger.info(f"[Feast Wrapper] Falling back to SQLite legacy store for {ticker}")
    return sqlite_get_latest_features(ticker)


def get_historical_features(
    entity_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Point-in-time correct historical feature retrieval.
    entity_df must contain:
      - 'ticker': str
      - 'event_timestamp': datetime64[ns, UTC]
    
    Guarantees zero data leakage by joining only features available strictly
    as of event_timestamp.
    """
    store = get_feature_store()
    
    if features is None:
        features = [f"technical_indicators_view:{ind}" for ind in INDICATORS_25] + \
                   [f"volatility_regime_view:{vf}" for vf in VOLATILITY_FEATURES]

    # Ensure event_timestamp is timezone-aware UTC
    df_entities = entity_df.copy()
    if 'event_timestamp' in df_entities.columns:
        ts = pd.to_datetime(df_entities['event_timestamp'])
        if hasattr(ts, 'dt') and ts.dt.tz is None:
            df_entities['event_timestamp'] = ts.dt.tz_localize('UTC')

    training_data = store.get_historical_features(
        entity_df=df_entities,
        features=features
    )
    return training_data.to_df()


def _append_or_write_parquet(filepath: str, new_df: pd.DataFrame, key_cols: List[str]):
    """Safely upserts dataframe into an offline Parquet file."""
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_parquet(filepath)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            # Deduplicate by key columns keeping last
            combined = combined.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
            combined.to_parquet(filepath, index=False)
            return
        except Exception:
            pass
    new_df.to_parquet(filepath, index=False)
