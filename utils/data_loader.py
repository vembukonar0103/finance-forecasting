import os
import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf
from utils.config import DATA_RAW_DIR
def load_data(ticker: str, period="6mo", interval="1d", force_download=False) -> pd.DataFrame:
    """
    Downloads historical data from yfinance, saves it as a parquet file to data/raw/,
    and returns it formatted with the required columns.
    """
    raw_path = os.path.join(DATA_RAW_DIR, f"{ticker}_raw.parquet")
    
    if not force_download and os.path.exists(raw_path):
        # Load from cache
        df = pd.read_parquet(raw_path)
    else:
        # Download from yfinance
        df = yf.download(ticker, period=period, interval=interval).reset_index()
        # Cache raw data
        df.to_parquet(raw_path, index=False)
        
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
    
    # Calculate amount
    if 'close' in df.columns and 'volume' in df.columns:
        df['amount'] = df['close'] * df['volume']
        
    return df
if __name__ == "__main__":
    df = load_data("AAPL")
    print(df.head())
