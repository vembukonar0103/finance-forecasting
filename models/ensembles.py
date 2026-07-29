import pandas as pd
import numpy as np

def compute_weighted_ensemble(pred_tfm: pd.DataFrame, pred_kronos: pd.DataFrame, weight_tfm: float = 0.5) -> pd.DataFrame:
    """
    Computes a static weighted ensemble of TimesFM and Kronos point forecasts.
    
    Parameters
    ----------
    pred_tfm : pd.DataFrame
        Forecast DataFrame from TimesFM containing 'point' column.
    pred_kronos : pd.DataFrame
        Forecast DataFrame from Kronos containing 'point' column.
    weight_tfm : float
        Weight to assign to TimesFM point forecast (between 0 and 1).
        
    Returns
    -------
    pd.DataFrame
        Combined forecast DataFrame.
    """
    # Align on index
    common_idx = pred_tfm.index.intersection(pred_kronos.index)
    if len(common_idx) == 0:
        # If indexes are not aligned, reset index and align by position
        tfm_vals = pred_tfm['point'].values
        kronos_vals = pred_kronos['point'].values
        n = min(len(tfm_vals), len(kronos_vals))
        point_ens = weight_tfm * tfm_vals[:n] + (1.0 - weight_tfm) * kronos_vals[:n]
        return pd.DataFrame({'point': point_ens}, index=pred_tfm.index[:n])
        
    tfm_aligned = pred_tfm.loc[common_idx, 'point'].astype(float)
    kronos_aligned = pred_kronos.loc[common_idx, 'point'].astype(float)
    
    point_ens = weight_tfm * tfm_aligned + (1.0 - weight_tfm) * kronos_aligned
    
    return pd.DataFrame({'point': point_ens}, index=common_idx)

def compute_confidence_ensemble(
    pred_tfm: pd.DataFrame, 
    pred_kronos: pd.DataFrame,
    mae_tfm_val: float, 
    mae_kronos_val: float,
    val_tfm_bands: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Computes a dynamic confidence-based ensemble of TimesFM and Kronos.
    
    Weights are:
    1. Inverse-variance/error weighted based on Validation MAEs:
       w_tfm_base = (1/mae_tfm_val) / (1/mae_tfm_val + 1/mae_kronos_val)
       w_kronos_base = 1.0 - w_tfm_base
    2. Scaled by TimesFM's prediction confidence on the active test set:
       We compute the width of the active P10-P90 band relative to the point forecast:
       width_test = mean((p90 - p10) / point)
       If val_tfm_bands is provided, we compare width_test to width_val (the width of the bands in validation).
       If width_test < width_val, TimesFM is more confident than usual, so we scale up its weight.
       If width_test > width_val, we scale down its weight.
    """
    # Safeguard against division by zero in MAE
    mae_tfm_val = max(1e-6, mae_tfm_val)
    mae_kronos_val = max(1e-6, mae_kronos_val)
    
    # 1. Base weights from validation MAE
    inv_tfm = 1.0 / mae_tfm_val
    inv_kronos = 1.0 / mae_kronos_val
    total_inv = inv_tfm + inv_kronos
    
    w_tfm = inv_tfm / total_inv
    
    # 2. Adjust using confidence interval width
    # Check if we have quantiles in TimesFM forecast
    if 'p10' in pred_tfm.columns and 'p90' in pred_tfm.columns:
        p10 = pred_tfm['p10'].values
        p90 = pred_tfm['p90'].values
        point = pred_tfm['point'].values
        
        # Avoid division by zero
        denom = np.where(point == 0, 1.0, point)
        width_test = np.mean((p90 - p10) / np.abs(denom))
        
        # Default val width factor
        width_val = 0.15 # Heuristic 15% range width
        
        if val_tfm_bands is not None and 'p10' in val_tfm_bands.columns and 'p90' in val_tfm_bands.columns:
            val_p10 = val_tfm_bands['p10'].values
            val_p90 = val_tfm_bands['p90'].values
            val_point = val_tfm_bands['point'].values
            val_denom = np.where(val_point == 0, 1.0, val_point)
            width_val = np.mean((val_p90 - val_p10) / np.abs(val_denom))
            
        width_val = max(1e-6, width_val)
        
        # Multiplier: if test width is narrower, multiplier > 1 (higher confidence)
        multiplier = width_val / max(1e-6, width_test)
        # Clip multiplier to avoid extreme weights
        multiplier = np.clip(multiplier, 0.5, 2.0)
        
        # Apply multiplier to TimesFM's odd ratio: w_tfm / (1 - w_tfm)
        odds = w_tfm / max(1e-6, 1.0 - w_tfm)
        adjusted_odds = odds * multiplier
        w_tfm = adjusted_odds / (1.0 + adjusted_odds)
        
    w_tfm = np.clip(w_tfm, 0.05, 0.95) # Keep weights bounded
    
    return compute_weighted_ensemble(pred_tfm, pred_kronos, weight_tfm=w_tfm)
