"""
utils/metrics.py
----------------
Evaluation metrics for time-series forecasting.

Supports both point-forecast models (Kronos) and quantile-forecast models (TimesFM).
All functions accept plain numpy arrays and return Python scalars.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def mae(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Mean Absolute Error."""
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return float(np.mean(np.abs(actuals - preds)))


def rmse(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))


def mape(actuals: np.ndarray, preds: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).
    Returns value in percent (e.g. 2.5 means 2.5%).
    Uses `epsilon` guard to avoid division by zero on zero-valued actuals.
    """
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return float(np.mean(np.abs((actuals - preds) / (np.abs(actuals) + epsilon))) * 100.0)


def directional_accuracy(
    actuals: np.ndarray,
    preds: np.ndarray,
    c0: float
) -> float:
    """
    Directional Accuracy (%).
    For each forecast step t, checks whether the model correctly predicts
    the direction (up or down) relative to the last known actual price c0.
    Returns value in percent (e.g. 60.0 means 60%).

    Parameters
    ----------
    actuals : array of actual prices over the holdout period
    preds   : array of predicted prices (point forecast) over the same period
    c0      : last known actual price just before the holdout starts
    """
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)

    # Direction relative to previous step (prepend c0 as the baseline)
    actual_prev = np.concatenate([[c0], actuals[:-1]])
    pred_prev = np.concatenate([[c0], preds[:-1]])

    actual_dir = np.sign(actuals - actual_prev)
    pred_dir = np.sign(preds - pred_prev)

    matches = np.sum(actual_dir == pred_dir)
    return float(matches / len(actuals) * 100.0)


def pinball_loss(
    actuals: np.ndarray,
    quantile_vals: np.ndarray,
    q_level: float
) -> float:
    """
    Pinball (quantile) loss for a single quantile level.

    L_q(y, f) = (y - f) * q        if y >= f
                (f - y) * (1 - q)  if y < f

    Parameters
    ----------
    actuals       : actual observed values
    quantile_vals : forecast quantile values at level q_level
    q_level       : quantile level in (0, 1), e.g. 0.1 for P10

    Returns the mean pinball loss (lower is better).
    """
    actuals = np.asarray(actuals, dtype=float)
    quantile_vals = np.asarray(quantile_vals, dtype=float)
    errors = actuals - quantile_vals
    loss = np.where(errors >= 0, q_level * errors, (q_level - 1.0) * errors)
    return float(np.mean(loss))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

QUANTILE_COLS = {
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
}


def compute_all_metrics(
    actuals: np.ndarray,
    pred_df: pd.DataFrame,
    c0: float,
    point_col: str = "point",
) -> dict:
    """
    Compute the full metric suite and return a flat dictionary.

    Parameters
    ----------
    actuals  : array of actual close prices over the holdout window
    pred_df  : forecast DataFrame.
                - Must contain a point-forecast column (default 'point').
                  Falls back to 'p50' or 'close' if 'point' is absent.
                - May optionally contain quantile columns: p10, p25, p50, p75, p90.
    c0       : last known actual close price before the holdout window starts.
    point_col: name of the point-forecast column in pred_df.

    Returns
    -------
    dict with keys:
        mae, rmse, mape, directional_accuracy,
        pinball_p10, pinball_p25, pinball_p50, pinball_p75, pinball_p90  (NaN if missing)
    """
    actuals = np.asarray(actuals, dtype=float)

    # --- resolve point forecast column ---
    if point_col in pred_df.columns:
        point_vals = pred_df[point_col].values.astype(float)
    elif "p50" in pred_df.columns:
        point_vals = pred_df["p50"].values.astype(float)
    elif "close" in pred_df.columns:
        point_vals = pred_df["close"].values.astype(float)
    else:
        # last resort: first numeric column
        numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("pred_df contains no usable numeric columns for point forecast.")
        point_vals = pred_df[numeric_cols[0]].values.astype(float)

    # Trim to same length (model may return exactly horizon rows)
    n = min(len(actuals), len(point_vals))
    actuals = actuals[:n]
    point_vals = point_vals[:n]

    metrics: dict = {
        "mae": mae(actuals, point_vals),
        "rmse": rmse(actuals, point_vals),
        "mape": mape(actuals, point_vals),
        "directional_accuracy": directional_accuracy(actuals, point_vals, c0),
    }

    # --- pinball losses (only if quantile columns exist) ---
    for col, q_level in QUANTILE_COLS.items():
        key = f"pinball_{col}"
        if col in pred_df.columns:
            q_vals = pred_df[col].values.astype(float)[:n]
            metrics[key] = pinball_loss(actuals, q_vals, q_level)
        else:
            metrics[key] = float("nan")

    return metrics
