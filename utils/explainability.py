def generate_reasoning(
    indicators: dict, 
    forecast_result: dict, 
    prior_indicators: dict = None
) -> list[str]:
    """
    Generates rule-based explainability reasoning strings based on current/prior
    technical indicators and forecast outputs.
    
    Parameters
    ----------
    indicators : dict
        Current technical indicators (e.g., {'rsi': 25, 'macd': 1.2, 'macd_sig': 0.8}).
    forecast_result : dict
        Forecast result parameters:
        - 'model_used': str (e.g., 'TimesFM' or 'ensemble')
        - 'expected_return_pct': float (decimal return, e.g., 0.08 for 8%)
        - 'confidence_score': float (confidence, [0, 1])
        - 'ensemble_agreement': bool (optional)
        - 'individual_forecasts': dict (optional mapping of model name to expected return)
    prior_indicators : dict, optional
        Prior Technical indicators for crossover detection.
        
    Returns
    -------
    list of str
        List of generated reasoning strings.
    """
    reasons = []
    
    # 1. Check RSI Bounds
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            reasons.append(f"RSI Oversold ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI Overbought ({rsi:.1f})")
            
    # 2. Check MACD Crossover (requires prior indicators)
    if prior_indicators is not None:
        macd_curr = indicators.get("macd")
        sig_curr = indicators.get("macd_sig") or indicators.get("macd_signal")
        macd_prior = prior_indicators.get("macd")
        sig_prior = prior_indicators.get("macd_sig") or prior_indicators.get("macd_signal")
        
        if all(v is not None for v in [macd_curr, sig_curr, macd_prior, sig_prior]):
            # Bullish crossover: MACD crosses above signal line
            if macd_prior <= sig_prior and macd_curr > sig_curr:
                reasons.append("MACD Bullish Crossover")
            # Bearish crossover: MACD crosses below signal line
            elif macd_prior >= sig_prior and macd_curr < sig_curr:
                reasons.append("MACD Bearish Crossover")
                
    # 3. Model prediction summary
    model = forecast_result.get("model_used")
    expected_ret = forecast_result.get("expected_return_pct")
    if model is not None and expected_ret is not None:
        # Scale to percentage
        pct_return = expected_ret * 100.0
        reasons.append(f"{model.title()} predicts {pct_return:+.1f}%")
        
    # 4. Multi-model trend confirmation
    agreement = forecast_result.get("ensemble_agreement")
    individual_forecasts = forecast_result.get("individual_forecasts")
    
    if agreement is True:
        reasons.append("Multi-model trend confirmation")
    elif isinstance(individual_forecasts, dict) and len(individual_forecasts) > 1:
        # Check if all individual predictions point in the same direction (all positive or all negative)
        forecast_values = list(individual_forecasts.values())
        if all(val > 0 for val in forecast_values) or all(val < 0 for val in forecast_values):
            reasons.append("Multi-model trend confirmation")
            
    # 5. Kronos Pattern Confirmation/Conflict
    kronos = forecast_result.get("kronos_pattern")
    if kronos is not None:
        pattern_label = kronos.get("label", "")
        pattern_conf = kronos.get("confidence", 0.0)
        
        if expected_ret is not None and pattern_label != "No Pattern":
            is_timesfm_bullish = expected_ret > 0
            is_timesfm_bearish = expected_ret < 0
            
            is_kronos_bullish = "Bullish" in pattern_label
            is_kronos_bearish = "Bearish" in pattern_label
            
            if (is_timesfm_bullish and is_kronos_bullish) or (is_timesfm_bearish and is_kronos_bearish):
                reasons.append(f"Kronos trend confirmation ({pattern_label}, {pattern_conf:.0%} confidence)")
            elif (is_timesfm_bullish and is_kronos_bearish) or (is_timesfm_bearish and is_kronos_bullish):
                reasons.append(f"Kronos pattern conflicts with forecast direction ({pattern_label}, {pattern_conf:.0%} confidence)")
                
    # 6. Overall Confidence Score
    confidence = forecast_result.get("confidence_score")
    if confidence is not None:
        reasons.append(f"Overall Confidence: {confidence * 100.0:.0f}%")
        
    return reasons
