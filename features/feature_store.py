import sqlite3
import pandas as pd
import os
from utils.config import DB_PATH

def init_db():
    """
    Initializes the SQLite database and creates the long-format features table if it does not exist.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Define long-format table schema
    query = """
    CREATE TABLE IF NOT EXISTS features_long (
        ticker TEXT,
        date TEXT,
        feature_name TEXT,
        feature_value REAL,
        PRIMARY KEY (ticker, date, feature_name)
    )
    """
    cursor.execute(query)
    conn.commit()
    conn.close()

def store_features(ticker: str, df_features: pd.DataFrame):
    """
    Converts wide-format technical features into long-format and saves them to the SQLite database.
    Overwrites/updates rows with the same (ticker, date, feature_name).
    """
    init_db()
    
    # Prepare wide data to melt
    df = df_features.copy()
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    elif 'timestamps' in df.columns:
        df = df.rename(columns={'timestamps': 'date'})
        
    # Ensure 'date' is a string formatted as YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Melt dataframe to long format
    long_df = df.melt(id_vars=['date'], var_name='feature_name', value_name='feature_value')
    long_df.insert(0, 'ticker', ticker)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # SQL query for bulk replace
    query = """
    INSERT OR REPLACE INTO features_long (ticker, date, feature_name, feature_value)
    VALUES (?, ?, ?, ?)
    """
    
    # Prepare list of tuples, mapping NaNs to None for SQL NULL
    data_tuples = []
    for _, row in long_df.iterrows():
        val = row['feature_value']
        # Convert NaN to None for proper SQL NULL storage
        sql_val = None if pd.isna(val) else float(val)
        data_tuples.append((row['ticker'], row['date'], row['feature_name'], sql_val))
        
    cursor.executemany(query, data_tuples)
    conn.commit()
    conn.close()

def get_latest_features(ticker: str) -> dict:
    """
    Returns the latest technical feature values (as a dict) for a given ticker.
    First finds the latest date, then pulls all features for that date.
    Returns empty dict if no data found.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Find the latest date for the ticker
    cursor.execute("SELECT MAX(date) FROM features_long WHERE ticker = ?", (ticker,))
    latest_date_row = cursor.fetchone()
    
    if not latest_date_row or latest_date_row[0] is None:
        conn.close()
        return {}
        
    latest_date = latest_date_row[0]
    
    # Pull all features for this date
    cursor.execute(
        "SELECT feature_name, feature_value FROM features_long WHERE ticker = ? AND date = ?",
        (ticker, latest_date)
    )
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to dictionary
    feature_dict = {}
    for feature_name, feature_value in rows:
        feature_dict[feature_name] = feature_value
        
    return feature_dict
