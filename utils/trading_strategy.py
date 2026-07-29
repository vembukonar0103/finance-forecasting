import pandas as pd
import numpy as np

# Configurable return thresholds for the 5-state signal engine
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.021,   # >= 2.1% expected return
    "buy": 0.0076,         # >= 0.76% expected return
    "sell": -0.0140,       # <= -1.40% expected return
    "strong_sell": -0.0327 # <= -3.27% expected return
}

def generate_signal(
    pred_df: pd.DataFrame, 
    c0: float, 
    threshold: float = 0.02,
    prob_plus5pct: float = None,
    min_confidence: float = 0.40
) -> int:
    """
    Generates a trading signal (+1: BUY/LONG, -1: SELL/SHORT, 0: HOLD/FLAT)
    based on predicted returns and confidence levels.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Forecast DataFrame containing 'point' column, and optionally 'p10', 'p90'.
    c0 : float
        Current close price (just before forecast start).
    threshold : float
        Expected return threshold to trigger a signal (e.g. 0.02 = 2%).
    prob_plus5pct : float, optional
        TimesFM probability P(>= +5% in 30 days).
    min_confidence : float
        Minimum confidence level required to act on the signal.
        If using TimesFM probability bands, confidence is computed as:
        1 - (p90 - p10) / (2 * point)
    
    Returns
    -------
    int
        Signal: +1, -1, or 0.
    """
    if pred_df.empty or 'point' not in pred_df.columns:
        return 0
        
    # 1. Expected Return over the 30-day horizon
    pred_final = pred_df['point'].iloc[-1]
    expected_return = (pred_final - c0) / c0
    
    # 2. Confidence level calculation
    confidence = 1.0  # Default if no uncertainty bands
    
    if 'p10' in pred_df.columns and 'p90' in pred_df.columns:
        p10 = pred_df['p10'].iloc[-1]
        p90 = pred_df['p90'].iloc[-1]
        point = pred_df['point'].iloc[-1]
        
        # Relative spread: narrower = higher confidence
        rel_spread = (p90 - p10) / max(1e-5, np.abs(point))
        # Map relative spread to a confidence score [0, 1]
        # (e.g. if spread is 10% of price, confidence is high; if spread is 50%, confidence is low)
        confidence = max(0.0, 1.0 - rel_spread)
        
    # If explicit probability of +5% return is provided, combine it
    if prob_plus5pct is not None:
        # P(>= 5%) can act as a bullish confidence
        # For buy: we want prob_plus5pct to be high
        # For sell: we want prob_plus5pct to be low
        pass
        
    # 3. Decision Rules
    # Buy rule: high expected positive return AND sufficient confidence
    if expected_return > threshold:
        if prob_plus5pct is not None:
            # If prob_plus5pct is high, it supports the buy
            if prob_plus5pct >= min_confidence:
                return 1
            else:
                return 0 # Not confident enough
        elif confidence >= min_confidence:
            return 1
        else:
            return 0
            
    # Sell rule: high expected negative return AND sufficient confidence
    elif expected_return < -threshold:
        # Note: If expected return is negative, prob_plus5pct should be very low.
        # So we check if the model is confident in its bearishness.
        if prob_plus5pct is not None:
            # P(>= 5%) should be low for a sell signal
            if prob_plus5pct < (1.0 - min_confidence):
                return -1
            else:
                return 0
        elif confidence >= min_confidence:
            return -1
        else:
            return 0
            
    else:
        return 0 # Hold/Flat

def generate_signal_v2(
    expected_return_pct: float,
    confidence_score: float,
    min_confidence: float = 0.653,
    prob_plus5pct: float = None
) -> str:
    """
    Generates a 5-state trading signal ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
    based on expected return, confidence score, and optional probability confirmation.
    
    Parameters
    ----------
    expected_return_pct : float
        Expected return of the asset over the horizon. Should match the scale of SIGNAL_THRESHOLDS
        (e.g., 0.05 for 5% return if thresholds are configured as decimals).
    confidence_score : float
        General confidence score of the forecast [0, 1].
    min_confidence : float
        Minimum confidence level required to generate a non-Hold signal.
    prob_plus5pct : float, optional
        Symmetric confirmation probability P(>= +5% return) [0, 1].
        Kept in signature for informational context, but no longer used for gating.
    """
    # 1. Check confidence gate
    if confidence_score < min_confidence:
        return "Hold"
        
    # 2. Map expected return magnitude to 5 states
    if expected_return_pct >= SIGNAL_THRESHOLDS["strong_buy"]:
        return "Strong Buy"
        
    elif expected_return_pct >= SIGNAL_THRESHOLDS["buy"]:
        return "Buy"
        
    elif expected_return_pct <= SIGNAL_THRESHOLDS["strong_sell"]:
        return "Strong Sell"
        
    elif expected_return_pct <= SIGNAL_THRESHOLDS["sell"]:
        return "Sell"
        
    else:
        return "Hold"
