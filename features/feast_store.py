"""
features/feast_store.py
-----------------------
Feast wrapper around existing SQLite features_long table and local Redis online store.

Scope:
- Wraps 21 TA-Lib indicators from SQLite store into Feast online store (Redis).
- Does not modify existing features/feature_store.py (SQLite long-format stays as source of truth).
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from feast import FeatureStore
from feature_repo.definitions import ticker_entity, ta_indicators_fv
from utils.config import DB_PATH
from features.feature_store import get_latest_features

INDICATORS = [
    'sma10', 'sma20', 'sma50', 'sma200',
    'ema10', 'ema20', 'ema50', 'ema200',
    'rsi', 'macd', 'macd_sig', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'atr', 'adx', 'cci', 'slowk', 'slowd', 'willr'
]

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feature_repo"))


def sync_sqlite_to_feast():
    """
    Exports SQLite long-format features into wide-format parquet file,
    applies Feast definitions, and materializes features to Redis online store.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"SQLite database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df_long = pd.read_sql("SELECT ticker, date, feature_name, feature_value FROM features_long", conn)
    conn.close()

    if df_long.empty:
        print("[Feast Store] No data found in SQLite features_long table.")
        return

    # Pivot long format to wide format
    df_wide = df_long.pivot(index=['ticker', 'date'], columns='feature_name', values='feature_value').reset_index()
    
    # Add Feast required timestamp columns
    df_wide['event_timestamp'] = pd.to_datetime(df_wide['date'])
    df_wide['created_timestamp'] = pd.to_datetime(df_wide['date'])
    
    os.makedirs(os.path.join(REPO_PATH, "data"), exist_ok=True)
    parquet_path = os.path.abspath(os.path.join(REPO_PATH, "data", "features_wide.parquet"))
    df_wide.to_parquet(parquet_path)
    # Initialize Feast FeatureStore, apply definitions, and materialize to Redis
    store = FeatureStore(repo_path=REPO_PATH)
    store.apply([ticker_entity, ta_indicators_fv])
    
    start_date = datetime.now() - timedelta(days=365 * 5)
    end_date = datetime.now() + timedelta(days=2)
    
    store.materialize(start_date=start_date, end_date=end_date)
    print(f"[Feast Store] Successfully materialized features from {parquet_path} to Redis online store.")


def get_online_features(ticker: str) -> Dict[str, Any]:
    """
    Retrieves latest 21 TA-Lib online features from Feast's Redis online store for a given ticker.
    """
    store = FeatureStore(repo_path=REPO_PATH)
    
    feature_refs = [f"ta_indicators_fv:{ind}" for ind in INDICATORS]
    
    response = store.get_online_features(
        features=feature_refs,
        entity_rows=[{"ticker": ticker}]
    ).to_dict()
    
    online_dict = {}
    for ind in INDICATORS:
        feat_name = f"ta_indicators_fv__{ind}"
        val_list = response.get(ind) or response.get(feat_name)
        if val_list and len(val_list) > 0:
            val = val_list[0]
            online_dict[ind] = float(val) if val is not None and not np.isnan(val) else None
        else:
            online_dict[ind] = None

    return online_dict


def run_standalone_test():
    """
    Standalone test: materializes features for AAPL into Feast online store (Redis),
    retrieves them via get_online_features(), and confirms match with SQLite store.
    """
    ticker = "AAPL"
    print(f"=== Running Feast Feature Store Standalone Test for {ticker} ===")
    
    # 1. Sync & Materialize to Redis
    sync_sqlite_to_feast()
    
    # 2. Retrieve online features from Feast (Redis)
    online_feats = get_online_features(ticker)
    
    # 3. Retrieve latest features from SQLite source of truth
    sqlite_feats = get_latest_features(ticker)
    
    print(f"\nComparing 21 TA-Lib Indicators for {ticker}:")
    print(f"{'Indicator':<15} | {'Feast (Redis)':<18} | {'SQLite (Source)':<18} | {'Match?':<10}")
    print("-" * 70)
    
    all_matched = True
    matched_count = 0
    
    for ind in INDICATORS:
        val_feast = online_feats.get(ind)
        val_sqlite = sqlite_feats.get(ind)
        
        if val_feast is None and val_sqlite is None:
            match = True
        elif val_feast is not None and val_sqlite is not None:
            match = abs(val_feast - val_sqlite) < 1e-3
        else:
            match = False
            
        if match:
            matched_count += 1
        else:
            all_matched = False
            
        str_feast = f"{val_feast:.4f}" if val_feast is not None else "N/A"
        str_sqlite = f"{val_sqlite:.4f}" if val_sqlite is not None else "N/A"
        
        print(f"{ind:<15} | {str_feast:<18} | {str_sqlite:<18} | {str(match):<10}")
        
    print("-" * 70)
    print(f"Total Matched: {matched_count}/{len(INDICATORS)}")
    
    if all_matched and matched_count == len(INDICATORS):
        print("PASS: All 21 TA-Lib features in Feast Redis online store match SQLite source of truth perfectly!")
    else:
        print("FLAG: Discrepancies detected between Feast online store and SQLite source of truth.")


if __name__ == "__main__":
    run_standalone_test()
