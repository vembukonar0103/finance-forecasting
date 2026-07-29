"""
utils/composite_signal.py
-------------------------
Combines Kronos pattern detections and TimesFM forecasts into a unified AI Rating & Composite Signal.
Reuses the confirmation/conflict rules established in explainability.
"""

from typing import Dict, Any


def compute_composite_rating(kronos_result: Dict[str, Any], timesfm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines Kronos pattern detection with TimesFM forecast outputs into a single AI rating payload.

    Parameters
    ----------
    kronos_result : dict
        Output from Kronos inference endpoint (keys: pattern, label, confidence, forecast_step).
    timesfm_result : dict
        Output from TimesFM inference endpoint (keys: horizons -> horizon_30 -> point_forecast/p50, probability_plus_5_pct).

    Returns
    -------
    dict
        Structured composite AI rating payload.
    """
    pattern_label = kronos_result.get("label", "No Pattern")
    pattern_conf = float(kronos_result.get("confidence", 0.0))
    pattern_name = kronos_result.get("pattern", "CDLENGULFING")

    # Extract horizon 30 from timesfm_result
    horizons = timesfm_result.get("horizons", {})
    h30 = horizons.get("horizon_30", horizons.get("30", timesfm_result))
    
    if isinstance(h30, dict):
        p50_list = h30.get("p50", h30.get("point_forecast", []))
        prob_plus_5 = float(h30.get("probability_plus_5_pct", 0.0))
    else:
        p50_list = []
        prob_plus_5 = 0.0

    if p50_list and len(p50_list) >= 2:
        expected_ret = (p50_list[-1] - p50_list[0]) / max(1e-5, abs(p50_list[0]))
    else:
        expected_ret = 0.0

    is_timesfm_bullish = expected_ret > 0
    is_timesfm_bearish = expected_ret < 0
    timesfm_direction = "Bullish" if is_timesfm_bullish else ("Bearish" if is_timesfm_bearish else "Neutral")

    is_kronos_bullish = "Bullish" in pattern_label
    is_kronos_bearish = "Bearish" in pattern_label

    signal_status = "neutral"
    summary_reason = f"TimesFM predicts {expected_ret:+.1%} (P(+5%)={prob_plus_5:.0%})"

    if pattern_label != "No Pattern":
        if (is_timesfm_bullish and is_kronos_bullish) or (is_timesfm_bearish and is_kronos_bearish):
            signal_status = "confirmation"
            ai_rating = f"Strong {timesfm_direction}"
            summary_reason = f"Kronos trend confirmation ({pattern_label}, {pattern_conf:.0%} confidence)"
        elif (is_timesfm_bullish and is_kronos_bearish) or (is_timesfm_bearish and is_kronos_bullish):
            signal_status = "conflict"
            ai_rating = "Conflicted / Neutral"
            summary_reason = f"Kronos pattern conflicts with forecast direction ({pattern_label}, {pattern_conf:.0%} confidence)"
        else:
            ai_rating = timesfm_direction
    else:
        ai_rating = timesfm_direction

    return {
        "ai_rating": ai_rating,
        "signal_status": signal_status,
        "kronos_pattern": pattern_name if pattern_label != "No Pattern" else "None",
        "kronos_label": pattern_label,
        "kronos_confidence": pattern_conf,
        "timesfm_prob_plus_5": prob_plus_5,
        "timesfm_direction": timesfm_direction,
        "summary_reason": summary_reason
    }
