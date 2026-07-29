import sqlite3
import json
import os
import pandas as pd
from datetime import datetime
from utils.config import BASE_DIR

# Put the database in the prediction or data directory
DB_PATH = os.path.join(BASE_DIR, "data", "experiments.db")

def init_experiment_db():
    """
    Initializes the SQLite database for tracking experiment runs.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        model_version TEXT,
        parameters TEXT,
        dataset TEXT,
        period TEXT,
        dry_run INTEGER,
        cagr REAL,
        sharpe REAL,
        sortino REAL,
        max_drawdown REAL,
        win_rate REAL,
        profit_factor REAL,
        mae REAL,
        rmse REAL,
        directional_accuracy REAL
    )
    """
    cursor.execute(query)
    
    # Safely alter table to add columns for existing databases without modifying or backfilling old rows
    try:
        cursor.execute("ALTER TABLE experiments ADD COLUMN period TEXT;")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE experiments ADD COLUMN dry_run INTEGER;")
    except sqlite3.OperationalError:
        pass
        
    # Create indexes for the new columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_period ON experiments(period);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dry_run ON experiments(dry_run);")
    
    # Create component_runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS component_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        component_name TEXT,
        parameters TEXT,
        output_summary TEXT
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_component_runs_name ON component_runs(component_name);")
    
    conn.commit()
    conn.close()

def log_experiment(
    model_version: str,
    parameters: dict,
    dataset: dict,
    metrics: dict
):
    """
    Logs an experiment run into the database.
    """
    init_experiment_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract period and dry_run at write time
    period = dataset.get("period")
    dry_run = dataset.get("dry_run")
    if dry_run is None:
        dry_run = parameters.get("dry_run")
    dry_run_int = 1 if dry_run else 0 if dry_run is not None else None
    
    query = """
    INSERT INTO experiments (
        timestamp, model_version, parameters, dataset, period, dry_run,
        cagr, sharpe, sortino, max_drawdown, win_rate, profit_factor,
        mae, rmse, directional_accuracy
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    cursor.execute(query, (
        datetime.now().isoformat(),
        model_version,
        json.dumps(parameters),
        json.dumps(dataset),
        period,
        dry_run_int,
        metrics.get("cagr"),
        metrics.get("sharpe"),
        metrics.get("sortino"),
        metrics.get("max_drawdown"),
        metrics.get("win_rate"),
        metrics.get("profit_factor"),
        metrics.get("mae"),
        metrics.get("rmse"),
        metrics.get("directional_accuracy")
    ))
    
    conn.commit()
    conn.close()
    print(f"Logged experiment for {model_version} to SQLite database.")

def get_experiments() -> pd.DataFrame:
    """
    Returns a DataFrame of all logged experiments.
    """
    init_experiment_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def log_component_run(
    component_name: str,
    parameters: dict,
    output_summary: dict
):
    """
    Logs a lightweight component run into the SQLite database.
    """
    init_experiment_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
    INSERT INTO component_runs (
        timestamp, component_name, parameters, output_summary
    ) VALUES (?, ?, ?, ?)
    """
    
    cursor.execute(query, (
        datetime.now().isoformat(),
        component_name,
        json.dumps(parameters),
        json.dumps(output_summary)
    ))
    
    conn.commit()
    conn.close()
    print(f"Logged component run for '{component_name}' to SQLite database.")
