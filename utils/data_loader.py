import os
import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf
from utils.config import DATA_RAW_DIR
def load_data(ticker: str, period="2y", interval="1d", force_download=False) -> pd.DataFrame:
    """
    Downloads historical data from yfinance, saves it as a parquet file to data/raw/,
    and returns it formatted with the required columns.
    Supports periods up to 15y / max for multi-year long context window inference (e.g., 4096-16384).
    """
    safe_period = str(period).replace(" ", "_")
    raw_path = os.path.join(DATA_RAW_DIR, f"{ticker}_{safe_period}_{interval}_raw.parquet")
    legacy_path = os.path.join(DATA_RAW_DIR, f"{ticker}_raw.parquet")
    
    if not force_download and os.path.exists(raw_path):
        df = pd.read_parquet(raw_path)
    elif not force_download and period in ["2y", "6mo"] and os.path.exists(legacy_path):
        df = pd.read_parquet(legacy_path)
    else:
        # Download from yfinance
        df = yf.download(ticker, period=period, interval=interval).reset_index()
        # Cache raw data
        try:
            df.to_parquet(raw_path, index=False)
        except Exception as e:
            print(f"[Warning] Failed caching parquet to {raw_path}: {e}")
        
    # Clean the column index if it is a MultiIndex or contains string representations of tuples
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append(col[0])
        else:
            col_str = str(col).strip()
            if col_str.startswith('(') and col_str.endswith(')'):
                try:
                    import ast
                    tup = ast.literal_eval(col_str)
                    if isinstance(tup, tuple) and len(tup) > 0:
                        new_cols.append(tup[0])
                        continue
                except Exception:
                    pass
            new_cols.append(col)
    df.columns = new_cols
    
    # Map columns to lowercase to avoid capitalization issues
    df.columns = [str(c).lower() for c in df.columns]
    
    # Rename columns to standard ones
    rename_map = {
        'date': 'timestamps',
        'datetime': 'timestamps',
        'timestamp': 'timestamps',
        'adj close': 'adj_close'
    }
    df = df.rename(columns=rename_map)
    
    # Clean NaN values in price and volume columns
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0.0)
        
    # Ensure proper datetime parsing, chronological sorting, and deduplication
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').drop_duplicates(subset=['timestamps'])

    df = df.reset_index(drop=True)
    
    # Calculate amount
    if 'close' in df.columns and 'volume' in df.columns:
        df['amount'] = df['close'] * df['volume']
        
    return df
if __name__ == "__main__":
    df = load_data("AAPL")
    print(df.head())

