import os
import sys
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.indicators import build_features

def generate_rolling_windows(df: pd.DataFrame, initial_context: int = 500, horizon: int = 30, val_days: int = 60):
    """
    Generates rolling train/val/test splits across historical data.
    Uses non-overlapping test splits of size `horizon`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical price data containing 'timestamps', 'close', etc.
    initial_context : int
        Minimum rows of history needed before starting test predictions.
    horizon : int
        Size of the out-of-sample forecast window (test window).
    val_days : int
        Size of the validation window just before the test split.
        
    Returns
    -------
    list of dict
        Each dict contains sliced dataframes: 'train', 'val', 'context', 'test',
        as well as metadata like 'split_date' and 'window_idx'.
    """
    # Ensure sorted by date
    df = df.sort_values('timestamps').reset_index(drop=True)
    N = len(df)
    
    windows = []
    T = initial_context
    window_idx = 0
    
    while T + horizon <= N:
        context_df = df.iloc[:T].copy()
        test_df = df.iloc[T:T+horizon].copy()
        
        train_df = df.iloc[:T-val_days].copy()
        val_df = df.iloc[T-val_days:T].copy()
        
        split_date = df.loc[T-1, 'timestamps']
        
        windows.append({
            'window_idx': window_idx,
            'split_date': split_date,
            'context': context_df,
            'train': train_df,
            'val': val_df,
            'test': test_df
        })
        
        T += horizon
        window_idx += 1
        
    return windows

def verify_no_lookahead_bias(df: pd.DataFrame, initial_context: int = 500) -> bool:
    """
    Verifies that features built on a subset/context dataframe are exactly identical
    to features built on the full dataframe up to the same date.
    This prevents look-ahead leakage from cumulative/moving window calculations.
    """
    # 1. Compute features on the entire dataset
    full_features = build_features(df)
    
    # 2. Pick a split index and slice context
    T = min(initial_context, len(df) - 50)
    context_df = df.iloc[:T]
    
    # 3. Compute features on just the context (representing walk-forward scenario)
    context_features = build_features(context_df)
    
    # 4. Compare the overlapping row at index T-1
    # Note: features output timestamp as a string. Let's align on index or date.
    full_row = full_features.iloc[T-1]
    context_row = context_features.iloc[T-1]
    
    # Compare all columns except timestamp if string formats differ
    cols_to_compare = [c for c in full_features.columns if c not in ('timestamp', 'timestamps', 'date')]
    
    mismatches = []
    for col in cols_to_compare:
        v_full = full_row[col]
        v_context = context_row[col]
        
        if pd.isna(v_full) and pd.isna(v_context):
            continue
        if not np.isclose(v_full, v_context, rtol=1e-5, atol=1e-5):
            mismatches.append((col, v_full, v_context))
            
    if mismatches:
        print(f"LOOK-AHEAD BIAS DETECTED! Context features do not match full dataset features at index {T-1}:")
        for col, vf, vc in mismatches:
            print(f"  {col}: Full={vf} | Sliced={vc}")
        return False
    else:
        print("Look-ahead bias safeguard check PASSED. Feature extraction is causality-preserving.")
        return True

if __name__ == "__main__":
    # Test on a dummy series
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=600, freq='D')
    dummy_df = pd.DataFrame({
        'timestamps': dates,
        'open': np.random.randn(600).cumsum() + 100,
        'high': np.random.randn(600).cumsum() + 102,
        'low': np.random.randn(600).cumsum() + 98,
        'close': np.random.randn(600).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, size=600)
    })
    
    print("Testing generate_rolling_windows...")
    windows = generate_rolling_windows(dummy_df, initial_context=500, horizon=30, val_days=60)
    print(f"Generated {len(windows)} windows.")
    if windows:
        print(f"First window context shape: {windows[0]['context'].shape}, test shape: {windows[0]['test'].shape}")
        
    print("\nRunning look-ahead verification on dummy data...")
    verify_no_lookahead_bias(dummy_df, initial_context=500)
