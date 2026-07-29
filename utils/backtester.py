import pandas as pd
import numpy as np

def compute_backtest_metrics(
    daily_returns: pd.Series, 
    trade_returns: list[float], 
    dates: pd.DatetimeIndex,
    rf_annual: float = 0.04
) -> dict:
    """
    Computes CAGR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, and Profit Factor.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Series of daily returns of the strategy.
    trade_returns : list of float
        List of returns per trade (window).
    dates : pd.DatetimeIndex
        Timestamps matching the daily returns.
    rf_annual : float
        Annualized risk-free rate.
    """
    metrics = {
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0
    }
    
    if len(daily_returns) == 0:
        return metrics
        
    # Cumulative Value Curve
    cum_value = (1.0 + daily_returns).cumprod()
    v_start = 1.0
    v_end = cum_value.iloc[-1]
    
    # CAGR
    n_days = (dates[-1] - dates[0]).days
    if n_days > 10:
        cagr = (v_end / v_start) ** (365.25 / n_days) - 1.0
    else:
        cagr = 0.0
    metrics["cagr"] = float(cagr)
    
    # Sharpe Ratio (Annualized)
    # Daily risk-free rate assuming 252 trading days
    rf_daily = rf_annual / 252.0
    excess_returns = daily_returns - rf_daily
    mean_excess = excess_returns.mean()
    std_returns = daily_returns.std()
    
    if std_returns > 1e-6:
        sharpe = np.sqrt(252.0) * (mean_excess / std_returns)
    else:
        sharpe = 0.0
    metrics["sharpe"] = float(sharpe)
    
    # Sortino Ratio (Annualized)
    # Downside deviation uses returns below the daily risk-free rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 2:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
    else:
        downside_std = daily_returns[daily_returns < 0].std()
        
    # Zero trades or no downside volatility — Sortino is undefined, not a real signal; return 0.0 rather than dividing by ~0
    if len(trade_returns) == 0 or daily_returns.std() <= 1e-6 or all(abs(r) < 1e-6 for r in trade_returns) or pd.isna(downside_std) or downside_std <= 1e-6:
        sortino = 0.0
    else:
        sortino = np.sqrt(252.0) * (mean_excess / downside_std)
    metrics["sortino"] = float(sortino)
    
    # Maximum Drawdown
    running_max = cum_value.cummax()
    drawdowns = (cum_value - running_max) / running_max
    max_dd = drawdowns.min()
    metrics["max_drawdown"] = float(max_dd)
    
    # Win Rate (based on trades)
    if len(trade_returns) > 0:
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / len(trade_returns)
    else:
        win_rate = 0.0
    metrics["win_rate"] = float(win_rate)
    
    # Profit Factor (based on trades)
    if len(trade_returns) > 0:
        gross_profits = sum(r for r in trade_returns if r > 0)
        gross_losses = sum(abs(r) for r in trade_returns if r < 0)
        if gross_losses > 1e-6:
            profit_factor = gross_profits / gross_losses
        elif gross_profits > 1e-6:
            profit_factor = float('inf')
        else:
            profit_factor = 1.0
    else:
        profit_factor = 0.0
    metrics["profit_factor"] = float(profit_factor)
    
    return metrics

def run_asset_backtest(
    df: pd.DataFrame, 
    windows: list[dict], 
    signals: list[int], 
    allow_short: bool = True,
    tx_cost_pct: float = 0.0010
) -> tuple[pd.Series, list[float], pd.DatetimeIndex]:
    """
    Simulates a daily backtest for a single asset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete historical data (to align dates).
    windows : list of dict
        Walk-forward windows.
    signals : list of int
        Trading signals (+1, -1, 0) corresponding to each window.
    allow_short : bool
        If False, shorts (-1) are treated as Flat (0).
    tx_cost_pct : float
        Transaction cost per position change (10 bps = 0.0010).
        
    Returns
    -------
    daily_returns : pd.Series
        Daily strategy returns series.
    trade_returns : list of float
        Return per trade.
    dates : pd.DatetimeIndex
        Timestamps of the returns.
    """
    # Sort and set index by date
    df = df.sort_values('timestamps').reset_index(drop=True)
    df.index = pd.to_datetime(df['timestamps'])
    
    # Collect all out-of-sample daily data
    all_test_dfs = []
    window_signals = {}
    
    for w, sig in zip(windows, signals):
        test_df = w['test'].copy()
        test_df['window_idx'] = w['window_idx']
        all_test_dfs.append(test_df)
        window_signals[w['window_idx']] = sig
        
    if not all_test_dfs:
        return pd.Series(), [], pd.DatetimeIndex([])
        
    # Concatenate all out-of-sample periods
    test_df_all = pd.concat(all_test_dfs).sort_values('timestamps').reset_index(drop=True)
    test_df_all.index = pd.to_datetime(test_df_all['timestamps'])
    
    # Calculate daily asset return
    asset_close = test_df_all['close'].astype(float)
    asset_daily_ret = asset_close.pct_change().fillna(0.0)
    
    # Map day to active window signal
    daily_signals = []
    prev_sig = 0
    trade_returns = []
    current_trade_ret = 0.0
    current_trade_len = 0
    
    daily_strat_returns = []
    
    # Group by window_idx to calculate trade returns
    for w_idx in sorted(window_signals.keys()):
        sig = window_signals[w_idx]
        if not allow_short and sig < 0:
            sig = 0
            
        # Get prices for this window
        w_df = test_df_all[test_df_all['window_idx'] == w_idx]
        if w_df.empty:
            continue
            
        w_close = w_df['close'].astype(float).values
        # Trade return is simply the return of holding the position
        p_start = w_df['close'].iloc[0]
        # Previous close before window start: we can look up in full df
        w_start_date = w_df.index[0]
        full_idx = df.index.get_loc(w_start_date)
        p_prev = df['close'].iloc[full_idx - 1] if full_idx > 0 else p_start
        
        p_end = w_df['close'].iloc[-1]
        
        # Gross trade return
        if sig == 1:
            trade_ret = (p_end - p_prev) / p_prev
        elif sig == -1:
            trade_ret = (p_prev - p_end) / p_prev
        else:
            trade_ret = 0.0
            
        # Apply transaction costs on entry and exit
        # Position change cost
        sig_change = abs(sig - prev_sig)
        trade_ret -= sig_change * tx_cost_pct
        prev_sig = sig
        
        trade_returns.append(trade_ret)
        
        # Calculate daily strategy returns for this window
        w_daily_asset_ret = asset_daily_ret.loc[w_df.index]
        
        # First day of the window incurs transaction costs
        w_daily_strat_ret = sig * w_daily_asset_ret
        if len(w_daily_strat_ret) > 0:
            w_daily_strat_ret.iloc[0] -= sig_change * tx_cost_pct
            
        daily_strat_returns.append(w_daily_strat_ret)
        
    daily_strat_returns_series = pd.concat(daily_strat_returns)
    
    return daily_strat_returns_series, trade_returns, daily_strat_returns_series.index

def run_portfolio_backtest(
    ticker_signals: dict, 
    capital_allocation: dict,
    rf_annual: float = 0.04
) -> dict:
    """
    Runs a unified multi-asset portfolio simulation using custom capital allocations.
    
    Parameters
    ----------
    ticker_signals : dict
        A dictionary mapping ticker to a sub-dict containing:
        - 'daily_returns': pd.Series
        - 'trade_returns': list[float]
    capital_allocation : dict
        A dictionary mapping ticker to capital allocation percentage (e.g. {'AAPL': 20.0}).
        Does not need to sum to 100.0, but values will be normalized to sum to 1.0.
    rf_annual : float
        Annualized risk-free rate.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'daily_returns': pd.Series (combined portfolio returns)
        - 'metrics': dict of backtest metrics (CAGR, Sharpe, Max Drawdown, etc.)
          plus 'sample_size' and 'low_confidence_warning' fields.
    """
    if not ticker_signals or not capital_allocation:
        return {"daily_returns": pd.Series(), "metrics": {}}
        
    # Normalize allocations to sum to 1.0 (decimals)
    alloc_sum = sum(capital_allocation.values())
    if alloc_sum == 0.0:
        norm_allocation = {ticker: 1.0 / len(capital_allocation) for ticker in capital_allocation}
    else:
        norm_allocation = {ticker: val / alloc_sum for ticker, val in capital_allocation.items()}
        
    daily_ret_cols = []
    tickers_present = []
    
    for ticker, signals in ticker_signals.items():
        if ticker in norm_allocation and "daily_returns" in signals:
            col_ret = signals["daily_returns"]
            if not col_ret.empty:
                weight = norm_allocation[ticker]
                daily_ret_cols.append(col_ret * weight)
                tickers_present.append(ticker)
                
    if not daily_ret_cols:
        return {"daily_returns": pd.Series(), "metrics": {}}
        
    # Align dates and sum the weighted daily returns
    port_df = pd.concat(daily_ret_cols, axis=1).fillna(0.0)
    port_daily_ret = port_df.sum(axis=1)
    dates = port_daily_ret.index
    
    # Combine trade returns across all present assets
    port_trade_rets = []
    for ticker in tickers_present:
        trade_rets = ticker_signals[ticker].get("trade_returns", [])
        port_trade_rets.extend(trade_rets)
        
    # Run the core backtester metrics
    port_metrics = compute_backtest_metrics(
        daily_returns=port_daily_ret,
        trade_returns=port_trade_rets,
        dates=dates,
        rf_annual=rf_annual
    )
    
    sample_size = len(port_trade_rets)
    port_metrics["sample_size"] = sample_size
    port_metrics["low_confidence_warning"] = sample_size < 20
    
    return {
        "daily_returns": port_daily_ret,
        "metrics": port_metrics
    }
