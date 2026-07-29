import sys
import os
import pandas as pd
import sqlite3

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from features.indicators import build_features
from features.feature_store import store_features, get_latest_features
from utils.config import DB_PATH

def main():
    ticker = "AAPL"
    print(f"1. Downloading 6 months of historical data for {ticker}...")
    df = load_data(ticker, period="6mo", interval="1d")
    print(f"Data retrieved: {df.shape[0]} rows.")
    
    print("\n2. Computing 21 technical indicators via TA-Lib...")
    feats_df = build_features(df)
    print(f"Computed features shape: {feats_df.shape}")
    
    # Check NaN values (expected initial values due to moving average lookbacks)
    print("\nChecking initial NaN values (due to moving average lookbacks):")
    nan_summary = feats_df.isna().sum()
    for col, count in nan_summary.items():
        if count > 0:
            print(f"  {col}: {count} NaNs")
            
    # Check range and values sanity
    print("\nRunning technical indicator sanity checks:")
    # RSI between 0 and 100
    if 'rsi' in feats_df.columns:
        rsi_min = feats_df['rsi'].min()
        rsi_max = feats_df['rsi'].max()
        print(f"  RSI range: [{rsi_min:.2f}, {rsi_max:.2f}] (Expected: within [0, 100])")
    # WILLR between -100 and 0
    if 'willr' in feats_df.columns:
        willr_min = feats_df['willr'].min()
        willr_max = feats_df['willr'].max()
        print(f"  Williams %R range: [{willr_min:.2f}, {willr_max:.2f}] (Expected: within [-100, 0])")
        
    print(f"\n3. Storing features into SQLite long-format table 'features_long'...")
    store_features(ticker, feats_df)
    print("Features stored successfully.")
    
    # Direct SQL query check to verify the long format
    print("\n4. Directly querying SQLite 'features_long' table for verification:")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM features_long WHERE ticker = ? LIMIT 10", (ticker,))
    sample_rows = cursor.fetchall()
    
    # Get column names
    cursor.execute("PRAGMA table_info(features_long)")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()
    
    print(f"  Table columns: {columns}")
    print("  Sample rows (first 10):")
    for row in sample_rows:
        print(f"    {row}")
        
    print(f"\n5. Retrieving latest features for {ticker} via CRUD helper:")
    latest = get_latest_features(ticker)
    print(f"  Retrieved {len(latest)} indicators.")
    print("  Sample indicator values:")
    sample_keys = ['sma10', 'sma200', 'rsi', 'willr', 'macd_hist', 'bb_upper']
    for k in sample_keys:
        if k in latest:
            print(f"    {k}: {latest[k]}")
        
    print("\nVerification Complete!")

if __name__ == "__main__":
    main()
