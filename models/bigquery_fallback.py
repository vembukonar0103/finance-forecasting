"""
models/bigquery_fallback.py
---------------------------
Fallback Forecasting Service using BigQuery AI.FORECAST or Statistical Trend Extrapolation.

Phase 5 (Task 6A) Deliverable:
- Serves as the resilience fallback when TimesFM encounters exceptions (OOM, timeout, or breaker tripped).
- Queries Google Cloud BigQuery ML.FORECAST / pre-materialized forecasts if GCP credentials are configured.
- Gracefully degrades to a robust Holt-Winters / Exponential Moving Average statistical trend model
  with quantile uncertainty bands if GCP is unreachable.
- Flags the response with 'fallback': True and indicates the provider used.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

try:
    from google.cloud import bigquery
    HAS_BIGQUERY = True
except ImportError:
    HAS_BIGQUERY = False


class BigQueryForecast:
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.dataset_id = dataset_id or os.getenv("BIGQUERY_DATASET", "finance_forecasting")
        self.client = None

        if HAS_BIGQUERY and self.project_id and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                self.client = bigquery.Client(project=self.project_id)
                print(f"[BigQueryFallback] Initialized BigQuery Client for GCP project: {self.project_id}")
            except Exception as e:
                print(f"[BigQueryFallback Warning] Failed initializing BigQuery client: {e}. Using statistical fallback.")
        else:
            print("[BigQueryFallback] GCP BigQuery not configured. Utilizing statistical trend extrapolation engine.")

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        horizon: int = 30,
        freq: str = 'B'
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Executes fallback forecasting.
        Returns:
          - pred_df: DataFrame with point, p10, p25, p50, p75, p90 columns
          - prob_plus_5: P(>= +5% in 30 days)
          - metadata: dict with provider, fallback flag, latency/status
        """
        # Try BigQuery query if client is active
        if self.client:
            try:
                return self._predict_bigquery(df, target_col, horizon, freq)
            except Exception as e:
                print(f"[BigQueryFallback Warning] BigQuery query failed ({e}). Falling back to statistical engine.")

        # Statistical Trend Extrapolation Engine
        return self._predict_statistical(df, target_col, horizon, freq)

    def _predict_bigquery(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int,
        freq: str
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        ticker = str(df.get('ticker', ['UNKNOWN'])[0] if 'ticker' in df.columns else 'UNKNOWN')
        query = f"""
            SELECT forecast_timestamp, forecast_value, prediction_interval_lower_bound, prediction_interval_upper_bound
            FROM `{self.project_id}.{self.dataset_id}.forecasts`
            WHERE ticker = @ticker
            ORDER BY forecast_timestamp ASC
            LIMIT {horizon}
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker)
            ]
        )
        query_job = self.client.query(query, job_config=job_config)
        bq_df = query_job.to_dataframe()

        if bq_df.empty:
            raise ValueError(f"No BigQuery materialized forecasts found for {ticker}")

        # Map BigQuery results
        last_val = float(df[target_col].iloc[-1])
        future_dates = pd.to_datetime(bq_df['forecast_timestamp'])
        point = bq_df['forecast_value'].astype(float).values
        lower = bq_df['prediction_interval_lower_bound'].astype(float).values
        upper = bq_df['prediction_interval_upper_bound'].astype(float).values

        pred_df = pd.DataFrame({
            'point': point,
            'p10': lower,
            'p25': 0.7 * lower + 0.3 * point,
            'p50': point,
            'p75': 0.3 * point + 0.7 * upper,
            'p90': upper
        }, index=future_dates)

        exp_ret = (point[-1] - last_val) / max(1e-5, last_val)
        prob_plus_5 = max(0.0, min(1.0, 0.5 + exp_ret * 2.0))

        metadata = {
            "fallback": True,
            "provider": "BigQuery_ML_FORECAST",
            "project": self.project_id
        }
        return pred_df, prob_plus_5, metadata

    def _predict_statistical(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int,
        freq: str
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Robust statistical trend fallback using Exponential Moving Average drift + rolling volatility.
        """
        # Ensure timestamp column
        ts_col = None
        for col in ['timestamps', 'timestamp', 'date', 'datetime']:
            if col in df.columns:
                ts_col = col
                break
        if ts_col:
            df_sorted = df.sort_values(ts_col).copy()
            last_timestamp = pd.to_datetime(df_sorted[ts_col].iloc[-1])
        else:
            df_sorted = df.copy()
            last_timestamp = pd.to_datetime(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        prices = df_sorted[target_col].astype(float).values
        last_price = prices[-1]

        # Compute annualized/daily drift and volatility
        returns = np.diff(prices) / prices[:-1]
        lookback = min(len(returns), 60)
        recent_returns = returns[-lookback:]
        
        # Exponentially-weighted drift
        weights = np.exp(np.linspace(-1, 0, len(recent_returns)))
        weights /= weights.sum()
        drift = float(np.sum(recent_returns * weights))
        vol = float(np.std(recent_returns) or 0.015)

        # Forecast steps
        steps = np.arange(1, horizon + 1)
        point_forecast = last_price * np.exp(drift * steps)
        
        # Quantile dispersion expands with sqrt(t)
        sigma_t = vol * np.sqrt(steps) * last_price
        p10 = point_forecast - 1.28 * sigma_t
        p25 = point_forecast - 0.67 * sigma_t
        p50 = point_forecast
        p75 = point_forecast + 0.67 * sigma_t
        p90 = point_forecast + 1.28 * sigma_t

        future_dates = pd.date_range(
            start=last_timestamp,
            periods=horizon + 1,
            freq=freq
        )[1:]

        pred_df = pd.DataFrame({
            'point': point_forecast,
            'p10': p10,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90
        }, index=future_dates)

        # Heuristic probability of >= +5%
        target_price = last_price * 1.05
        z_score = (target_price - point_forecast[-1]) / max(1e-5, sigma_t[-1])
        # Standard normal survival function approx
        prob_plus_5 = float(max(0.01, min(0.99, 1.0 / (1.0 + np.exp(1.7 * z_score)))))

        metadata = {
            "fallback": True,
            "provider": "Statistical_Drift_Fallback",
            "drift_daily": round(drift, 5),
            "volatility_daily": round(vol, 4)
        }
        return pred_df, prob_plus_5, metadata
