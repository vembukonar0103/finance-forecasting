import pandas as pd

def rank_opportunities(predictions: list[dict]) -> list[dict]:
    """
    Ranks opportunities by expected return adjusted by volatility-based risk.
    
    Opportunity Score = expected_return_pct / max(risk_score, 0.05)
    
    Parameters
    ----------
    predictions : list of dict
        A list of dictionaries containing prediction parameters:
        - 'ticker': str
        - 'expected_return_pct': float (expected return, e.g., 0.08 for 8%)
        - 'confidence_score': float (confidence score, [0, 1])
        - 'risk_score': float (volatility score from risk engine, [0, 1])
        
    Returns
    -------
    list of dict
        The opportunities ranked descending by opportunity_score, with
        'opportunity_score' added to each dictionary.
    """
    ranked = []
    for pred in predictions:
        pred_copy = pred.copy()
        expected_return = pred_copy.get("expected_return_pct", 0.0)
        risk_score = pred_copy.get("risk_score", 1.0)
        
        # Floor the risk score to avoid division by zero or explosive scores
        floored_risk = max(risk_score, 0.05)
        pred_copy["opportunity_score"] = expected_return / floored_risk
        ranked.append(pred_copy)
        
    # Sort descending by opportunity_score
    ranked.sort(key=lambda x: x["opportunity_score"], reverse=True)
    return ranked

def suggest_allocation(ranked_opportunities: list[dict]) -> dict[str, float]:
    """
    Suggests portfolio allocation using inverse-risk weighting normalized to 100%.
    
    ===========================================================================
    IMPORTANT NOTE: This function produces a simple inverse-risk-weighted
    allocation normalized to sum to 100%. It is NOT a mean-variance portfolio
    optimization (such as Markowitz optimization), as it ignores asset
    correlations, expected returns, and portfolio-level risk constraints.
    ===========================================================================
    
    Parameters
    ----------
    ranked_opportunities : list of dict
        A list of opportunities, each containing 'ticker' and 'risk_score'.
        
    Returns
    -------
    dict of (str: float)
        Mapping of ticker to suggested allocation percentage (e.g. {'AAPL': 25.0}).
        The sum of allocations is normalized to exactly 100.0%.
    """
    if not ranked_opportunities:
        return {}
        
    inv_risks = {}
    total_inv_risk = 0.0
    
    for opp in ranked_opportunities:
        ticker = opp["ticker"]
        risk_score = opp.get("risk_score", 1.0)
        floored_risk = max(risk_score, 0.05)
        weight = 1.0 / floored_risk
        inv_risks[ticker] = weight
        total_inv_risk += weight
        
    if total_inv_risk == 0.0:
        # Equal weight fallback
        share = 100.0 / len(ranked_opportunities)
        return {opp["ticker"]: round(share, 2) for opp in ranked_opportunities}
        
    # Normalize allocations to sum to 100%
    allocations = {}
    for ticker, weight in inv_risks.items():
        allocations[ticker] = round((weight / total_inv_risk) * 100.0, 2)
        
    # Correct any rounding error so they sum to exactly 100.0
    alloc_sum = sum(allocations.values())
    if alloc_sum != 100.0 and allocations:
        largest_ticker = max(allocations, key=allocations.get)
        diff = round(100.0 - alloc_sum, 2)
        allocations[largest_ticker] = round(allocations[largest_ticker] + diff, 2)
        
    return allocations
