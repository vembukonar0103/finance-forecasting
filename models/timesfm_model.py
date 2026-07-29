import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
import timesfm
from models.device import get_device

class TimesFMForecast:
    def __init__(self):
        self.device = torch.device(get_device())
        print(f"Initializing TimesFM 2.5 Torch on device: {self.device}")
        
        # Load timesfm model from pretrained
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=False
        )
        
        # Explicitly wire the model to the target device
        self.model.model.to(self.device)
        self.model.model.device = self.device
        self.model.device = self.device

        
        self._compiled_horizon = None

    def predict(self, df: pd.DataFrame, target_col='close', horizon=30, freq='B') -> tuple[pd.DataFrame, float]:
        """
        Runs forecast using TimesFM 2.5.
        df should have columns: 'timestamps' (dates) and target_col.
        Returns:
          - A DataFrame of forecasts containing columns:
              'point', 'p10', 'p25', 'p50', 'p75', 'p90'
            indexed by future business day/calendar day timestamps.
          - The probability P(>= +5% in 30 days) at day 30.
        """
        # Ensure input data is clean and sorted
        df_sorted = df.sort_values('timestamps').copy()
        
        if target_col not in df_sorted.columns:
            raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")
            
        close_prices = df_sorted[target_col].astype(np.float32).values
        last_timestamp = pd.to_datetime(df_sorted['timestamps'].iloc[-1])
        
        # Compile model dynamically if horizon changes
        if self._compiled_horizon != horizon:
            print(f"Compiling TimesFM model for horizon={horizon}...")
            config = timesfm.ForecastConfig(
                max_context=512,
                max_horizon=horizon,
                normalize_inputs=True,
                fix_quantile_crossing=True,
            )
            self.model.compile(config)
            self._compiled_horizon = horizon
            
        # Run forecasting
        # Inputs must be a list of numpy arrays
        point, quantiles = self.model.forecast(horizon=horizon, inputs=[close_prices])
        
        # Point shape: (1, horizon), Quantiles shape: (1, horizon, 10)
        point_forecast = point[0]
        quantile_forecast = quantiles[0]
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_timestamp, 
            periods=horizon + 1, 
            freq=freq
        )[1:]
        
        # Map quantiles to standard bands
        p10 = quantile_forecast[:, 1]
        p25 = 0.5 * quantile_forecast[:, 2] + 0.5 * quantile_forecast[:, 3]
        p50 = quantile_forecast[:, 5]
        p75 = 0.5 * quantile_forecast[:, 7] + 0.5 * quantile_forecast[:, 8]
        p90 = quantile_forecast[:, 9]
        
        pred_df = pd.DataFrame({
            'point': point_forecast,
            'p10': p10,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90
        }, index=future_dates)
        
        # Calculate Probability P(>= +5% in 30 days) at the final step (day-30)
        c0 = close_prices[-1]  # Price at day 0 (the last known actual close)
        day30_quantiles = quantile_forecast[-1] # Shape (10,)
        
        prob_plus_5 = self._calculate_prob_return(c0, day30_quantiles, target_pct=0.05)
        
        return pred_df, prob_plus_5

    def predict_timed(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        horizon: int = 30,
        freq: str = 'B'
    ) -> tuple:
        """
        Wraps predict() and additionally measures:
          - Wall-clock inference time (seconds)
          - Peak memory delta (MB) — GPU if available, else CPU via tracemalloc

        Returns
        -------
        (pred_df, prob_plus_5, inference_time_s, memory_mb)
        """
        use_gpu = str(self.device).startswith('cuda')

        if use_gpu:
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)
        else:
            tracemalloc.start()

        t_start = time.perf_counter()
        pred_df, prob_plus_5 = self.predict(df, target_col=target_col, horizon=horizon, freq=freq)
        t_end = time.perf_counter()

        if use_gpu:
            torch.cuda.synchronize(self.device)
            mem_after = torch.cuda.memory_allocated(self.device)
            memory_mb = max(0.0, (mem_after - mem_before) / 1024 ** 2)
        else:
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = mem_peak / 1024 ** 2  # bytes -> MB

        inference_time_s = t_end - t_start
        return pred_df, prob_plus_5, inference_time_s, memory_mb

    def _calculate_prob_return(self, c0: float, day_q: np.ndarray, target_pct=0.05) -> float:
        """
        Calculates the probability that the final price meets or exceeds the target return.
        Uses deciles q1-q9 (indices 1-9) of the quantile array to interpolate/extrapolate the CDF.
        
        NOTE: This is a prototype heuristic tail approximation. The sub-P10 and super-P90
        extrapolations are sensible approximations to avoid sudden cliffs and should not
        be cited as precise statistical densities in client-facing materials.
        """
        target_price = c0 * (1.0 + target_pct)
        
        # Extract decile values corresponding to cumulative probabilities [0.1, 0.2, ..., 0.9]
        q_vals = [day_q[i] for i in range(1, 10)]
        p_vals = [0.1 * i for i in range(1, 10)]
        
        # Extrapolation below P10 (linear down to 0)
        if target_price <= q_vals[0]:
            if q_vals[0] > 0:
                f_t = p_vals[0] * max(0.0, target_price / q_vals[0])
            else:
                f_t = p_vals[0]
        # Extrapolation above P90 (exponential decay towards 1.0)
        elif target_price >= q_vals[-1]:
            scale = q_vals[-1]
            if scale > 0:
                diff = target_price - q_vals[-1]
                # Decay cumulative probability tail: F(T) = 0.9 + 0.1 * (1 - exp(-diff / scale))
                f_t = p_vals[-1] + (1.0 - p_vals[-1]) * (1.0 - np.exp(-diff / scale))
            else:
                f_t = 1.0
            f_t = min(1.0, f_t)
        # Linear interpolation between deciles
        else:
            f_t = 0.5 # Default fallback
            for i in range(len(q_vals) - 1):
                if q_vals[i] <= target_price < q_vals[i+1]:
                    span = q_vals[i+1] - q_vals[i]
                    if span > 0:
                        fraction = (target_price - q_vals[i]) / span
                        f_t = p_vals[i] + fraction * (p_vals[i+1] - p_vals[i])
                    else:
                        f_t = p_vals[i]
                    break
                    
        # P(Price >= target_price) = 1.0 - F(target_price)
        return float(1.0 - f_t)
