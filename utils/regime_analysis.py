import pandas as pd
import numpy as np

def classify_market_regimes(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Classifies each row into:
    - bull_regime (bool): True if SPY close > 200-day SMA of SPY close (or asset itself if spy_df is None)
    - high_volatility (bool): True if rolling 20-day return std dev > historical median.
    
    Returns a DataFrame containing 'timestamps', 'bull_regime', and 'high_volatility'.
    """
    df_sorted = df.sort_values('timestamps').copy()
    df_sorted['timestamps'] = pd.to_datetime(df_sorted['timestamps'])
    
    # 1. Market Trend (Bull vs Bear)
    if spy_df is not None and not spy_df.empty:
        spy = spy_df.sort_values('timestamps').copy()
        spy['timestamps'] = pd.to_datetime(spy['timestamps'])
        spy['spy_sma200'] = spy['close'].rolling(window=200).mean()
        
        # Merge SPY regime back to main df
        spy_regime = pd.DataFrame({
            'timestamps': spy['timestamps'],
            'bull_regime': spy['close'] > spy['spy_sma200']
        })
        df_merged = pd.merge_asof(df_sorted, spy_regime, on='timestamps', direction='backward')
        bull_regime = df_merged['bull_regime'].fillna(True) # Fallback to True
    else:
        # Use own 200-day SMA
        own_sma200 = df_sorted['close'].rolling(window=200).mean()
        bull_regime = df_sorted['close'] > own_sma200
        bull_regime = bull_regime.fillna(True)
        
    # 2. Volatility Regime (High vs Low)
    daily_ret = df_sorted['close'].pct_change().fillna(0.0)
    rolling_vol = daily_ret.rolling(window=20).std().fillna(0.0)
    median_vol = rolling_vol.median()
    high_volatility = rolling_vol > median_vol
    
    regimes = pd.DataFrame({
        'timestamps': df_sorted['timestamps'],
        'bull_regime': bull_regime,
        'high_volatility': high_volatility
    })
    
    return regimes

def analyze_regime_performance(
    daily_strat_returns: pd.Series, 
    daily_actual_returns: pd.Series,
    regime_df: pd.DataFrame
) -> dict:
    """
    Computes performance metrics (CAGR, Sharpe, and Volatility) under different regimes:
    - Bull vs Bear market
    - High vs Low volatility
    
    Parameters
    ----------
    daily_strat_returns : pd.Series
        Series of daily strategy returns indexed by Datetime.
    daily_actual_returns : pd.Series
        Series of daily benchmark/asset returns indexed by Datetime.
    regime_df : pd.DataFrame
        DataFrame with 'timestamps' index or column, containing 'bull_regime' and 'high_volatility'.
    """
    regime_df = regime_df.copy()
    if 'timestamps' in regime_df.columns:
        regime_df.index = pd.to_datetime(regime_df['timestamps'])
        
    # Align indices
    common_idx = daily_strat_returns.index.intersection(regime_df.index)
    if len(common_idx) == 0:
        return {}
        
    strat_ret = daily_strat_returns.loc[common_idx]
    actual_ret = daily_actual_returns.loc[common_idx]
    reg_aligned = regime_df.loc[common_idx]
    
    results = {}
    
    # Define combinations
    regime_groups = {
        "all": pd.Series(True, index=common_idx),
        "bull": reg_aligned['bull_regime'] == True,
        "bear": reg_aligned['bull_regime'] == False,
        "high_vol": reg_aligned['high_volatility'] == True,
        "low_vol": reg_aligned['high_volatility'] == False
    }
    
    for name, mask in regime_groups.items():
        sub_strat = strat_ret[mask]
        sub_actual = actual_ret[mask]
        
        if len(sub_strat) < 5:
            results[name] = {"cagr": 0.0, "sharpe": 0.0, "count": len(sub_strat)}
            continue
            
        # CAGR
        n_days = len(sub_strat)
        cum_ret = (1.0 + sub_strat).prod()
        # Approximate CAGR based on number of trading days (assuming 252 trading days/year)
        cagr = (cum_ret) ** (252.0 / max(1, n_days)) - 1.0
        
        # Sharpe
        rf_daily = 0.04 / 252.0
        excess = sub_strat - rf_daily
        std = sub_strat.std()
        sharpe = np.sqrt(252.0) * excess.mean() / std if std > 1e-6 else 0.0
        
        # Mean Absolute Forecast Error if applicable
        results[name] = {
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "count": int(n_days)
        }
        
    return results

def classify_sideways(
    df: pd.DataFrame, 
    sma_short: int = 50, 
    sma_long: int = 200, 
    flat_threshold_pct: float = 2.0
) -> pd.Series:
    """
    Classifies periods where short-term and long-term SMAs are close, 
    indicating a sideways/rangebound market.
    
    Parameters
    ----------
    df : pd.DataFrame
        Asset data containing 'close' and 'timestamps'.
    sma_short : int
        Window size for short-term SMA.
    sma_long : int
        Window size for long-term SMA.
    flat_threshold_pct : float
        Percentage threshold within which SMAs are considered flat/converged.
        
    Returns
    -------
    pd.Series
        Boolean series indicating whether each date is classified as sideways.
    """
    df_sorted = df.sort_values('timestamps').copy()
    close = df_sorted['close'].astype(float)
    
    short_sma = close.rolling(window=sma_short).mean()
    long_sma = close.rolling(window=sma_long).mean()
    
    pct_diff = (short_sma - long_sma).abs() / long_sma
    is_sideways = pct_diff <= (flat_threshold_pct / 100.0)
    
    # Fill initial NaN values as False
    return is_sideways.fillna(False)

def classify_current_regime(
    df: pd.DataFrame, 
    sma_short: int = 50, 
    sma_long: int = 200, 
    flat_threshold_pct: float = 2.0,
    vol_window: int = 20
) -> dict:
    """
    Classifies the trend and volatility regimes for the most recent date in df.
    
    Parameters
    ----------
    df : pd.DataFrame
        Asset data containing 'close' and 'timestamps'.
    sma_short : int
        Window size for short-term SMA.
    sma_long : int
        Window size for long-term SMA.
    flat_threshold_pct : float
        Percentage threshold for sideways trend detection.
    vol_window : int
        Window size for realized volatility standard deviation.
        
    Returns
    -------
    dict
        {"trend": "Bull"|"Bear"|"Sideways", "volatility": "High"|"Low"}
    """
    if len(df) < max(sma_long, vol_window):
        return {"trend": "Sideways", "volatility": "Low"}
        
    df_sorted = df.sort_values('timestamps').copy()
    close = df_sorted['close'].astype(float)
    
    # 1. Trend Classification
    short_sma = close.rolling(window=sma_short).mean()
    long_sma = close.rolling(window=sma_long).mean()
    
    latest_close = close.iloc[-1]
    latest_short = short_sma.iloc[-1]
    latest_long = long_sma.iloc[-1]
    
    pct_diff = abs(latest_short - latest_long) / latest_long
    
    if pct_diff <= (flat_threshold_pct / 100.0):
        trend = "Sideways"
    elif latest_close > latest_long:
        trend = "Bull"
    else:
        trend = "Bear"
        
    # 2. Volatility Classification
    daily_ret = close.pct_change().fillna(0.0)
    rolling_vol = daily_ret.rolling(window=vol_window).std().fillna(0.0)
    
    latest_vol = rolling_vol.iloc[-1]
    median_vol = rolling_vol.median()
    
    volatility = "High" if latest_vol > median_vol else "Low"
    
    return {
        "trend": trend,
        "volatility": volatility
    }
