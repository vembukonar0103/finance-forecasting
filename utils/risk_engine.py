import numpy as np
import pandas as pd

def compute_stop_take(
    current_price: float, 
    atr: float, 
    direction: str, 
    atr_multiplier: float = 2.0, 
    reward_risk_ratio: float = 1.5
) -> tuple[float, float]:
    """
    Computes stop-loss and take-profit levels using ATR-based volatility stops.
    
    Parameters
    ----------
    current_price : float
        The current entry price of the asset.
    atr : float
        Average True Range (current volatility measure).
    direction : str
        Direction of the trade: "long", "buy", "+1", or "short", "sell", "-1".
    atr_multiplier : float
        Multiplier for the ATR to set the stop-loss distance.
    reward_risk_ratio : float
        Ratio of take-profit distance to stop-loss distance.
        
    Returns
    -------
    tuple of (float, float)
        (stop_loss, take_profit)
    """
    stop_distance = atr_multiplier * atr
    take_distance = stop_distance * reward_risk_ratio
    
    dir_clean = str(direction).lower().strip()
    if dir_clean in ("long", "buy", "+1", "1"):
        stop_loss = current_price - stop_distance
        take_profit = current_price + take_distance
    elif dir_clean in ("short", "sell", "-1"):
        stop_loss = current_price + stop_distance
        take_profit = current_price - take_distance
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be long/buy or short/sell.")
        
    return float(stop_loss), float(take_profit)

def compute_volatility_score(returns_series: pd.Series, window: int = 20) -> float:
    """
    Computes realized volatility as a percentile rank against the asset's own 
    historical distribution, producing a normalized score in [0, 1].
    
    Parameters
    ----------
    returns_series : pd.Series
        Historical daily returns series of the asset.
    window : int
        Rolling window size for calculating standard deviation of returns.
        
    Returns
    -------
    float
        Percentile rank score [0, 1] of the latest volatility compared to historical levels.
    """
    if len(returns_series) < window:
        return 0.0
        
    # Calculate rolling standard deviation of daily returns
    rolling_vol = returns_series.rolling(window=window).std().dropna()
    if rolling_vol.empty:
        return 0.0
        
    latest_vol = rolling_vol.iloc[-1]
    
    # Compute the percentile rank of the latest volatility value
    percentile_rank = (rolling_vol < latest_vol).mean()
    return float(percentile_rank)

def classify_risk_level(volatility_score: float) -> str:
    """
    Classifies risk level into Low/Medium/High using tertile cuts on the volatility score.
    
    Parameters
    ----------
    volatility_score : float
        Normalized volatility score between 0.0 and 1.0.
        
    Returns
    -------
    str
        "Low", "Medium", or "High" risk classification.
    """
    if volatility_score < 1.0 / 3.0:
        return "Low"
    elif volatility_score < 2.0 / 3.0:
        return "Medium"
    else:
        return "High"

def suggested_position_size(
    capital: float, 
    stop_loss_distance_pct: float, 
    risk_per_trade_pct: float = 1.0
) -> float:
    """
    Computes suggested position size using fixed-fractional position sizing.
    
    Parameters
    ----------
    capital : float
        Total trading capital available.
    stop_loss_distance_pct : float
        Stop-loss distance as a percentage of entry price (e.g. 5.0 for 5%).
    risk_per_trade_pct : float
        Fixed percentage of capital to risk per trade (e.g. 1.0 for 1%).
        
    Returns
    -------
    float
        Suggested position size in currency units.
    """
    if stop_loss_distance_pct <= 0.0:
        return 0.0
    
    # Position size = (Capital * Risk%) / StopLoss%
    position_size = capital * (risk_per_trade_pct / stop_loss_distance_pct)
    return float(position_size)

def expected_drawdown(volatility_score: float, confidence_score: float) -> float:
    """
    Computes a simple heuristic estimate of expected drawdown.
    
    ===========================================================================
    IMPORTANT NOTE: This is a HEURISTIC ESTIMATE scaled by volatility rank and
    dampened by forecast confidence. It is NOT a statistically derived
    drawdown forecast (such as historical value-at-risk or backtested drawdown)
    at the individual prediction level.
    ===========================================================================
    
    Parameters
    ----------
    volatility_score : float
        Normalized volatility score [0, 1].
    confidence_score : float
        Normalized confidence score [0, 1].
        
    Returns
    -------
    float
        Heuristic drawdown estimate (expressed as decimal return, e.g. 0.15 for 15%).
    """
    # Scale max heuristic drawdown (e.g., 30% baseline) by volatility, dampened by confidence
    max_baseline = 0.30
    drawdown = max_baseline * volatility_score * (1.5 - confidence_score)
    
    # Cap between 0% and 100%
    return float(np.clip(drawdown, 0.0, 1.0))
