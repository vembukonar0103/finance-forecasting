import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from features.indicators import build_features
from features.feature_store import store_features

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    for ticker in tickers:
        print(f"Populating features for {ticker}...")
        try:
            df = load_data(ticker, period="6mo", interval="1d", force_download=False)
            feats_df = build_features(df)
            store_features(ticker, feats_df)
            print(f"Successfully populated features for {ticker}. Row count: {feats_df.shape[0]}")
        except Exception as e:
            print(f"Failed to populate features for {ticker}: {e}")

if __name__ == "__main__":
    main()
