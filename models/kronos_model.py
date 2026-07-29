import os
import sys
import time
import tracemalloc
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import talib
import torch

# Add the cloned Kronos repository to python path so it can find its 'model' package
KRONOS_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "kronos_repo"))
if KRONOS_REPO_PATH not in sys.path:
    sys.path.insert(0, KRONOS_REPO_PATH)

# pyrefly: ignore [missing-import]
from model import Kronos, KronosTokenizer, KronosPredictor
from models.device import get_device


class KronosForecast:
    def __init__(self, model_name="NeoQuasar/Kronos-base", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"):
        self.device = get_device()
        print(f"Initializing Kronos ({model_name}) on device: {self.device}")

        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        self.model = Kronos.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=self.device,
            max_context=512
        )

    def predict(self, df: pd.DataFrame, target_col='close', lookback=400, horizon=30, freq='B') -> pd.DataFrame:
        """
        freq : pandas offset string for future date generation.
               Use 'B' (business days) for equities, 'D' (calendar days) for BTC-USD and
               other assets that trade every day including weekends.
        """
        df_sorted = df.sort_values('timestamps').copy()

        if target_col not in df_sorted.columns:
            raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")

        # .copy() here fixes the SettingWithCopyWarning
        df_lookback = df_sorted.tail(lookback).copy()

        last_timestamp = pd.to_datetime(df_lookback['timestamps'].iloc[-1])
        df_lookback['timestamps'] = pd.to_datetime(df_lookback['timestamps'])

        # Wrap in pd.Series — Kronos requires a Series (uses .dt accessor internally),
        # a bare pd.date_range() DatetimeIndex will break calc_time_stamps()
        future_dates = pd.Series(pd.date_range(
            start=last_timestamp,
            periods=horizon + 1,
            freq=freq          # <-- parameterised: 'B' for equities, 'D' for BTC-USD
        )[1:]).reset_index(drop=True)

        x_timestamp = df_lookback['timestamps'].reset_index(drop=True)

        pred_df = self.predictor.predict(
            df=df_lookback,
            x_timestamp=x_timestamp,
            y_timestamp=future_dates,
            pred_len=horizon
        )

        return pred_df

    def predict_timed(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        lookback: int = 400,
        horizon: int = 30,
        freq: str = 'B'
    ) -> tuple:
        """
        Wraps predict() and additionally measures:
          - Wall-clock inference time (seconds)
          - Peak memory delta (MB) — GPU if available, else CPU via tracemalloc

        Parameters
        ----------
        freq : 'B' for equity business days, 'D' for BTC-USD calendar days

        Returns
        -------
        (pred_df, inference_time_s, memory_mb)
        """
        use_gpu = str(self.device).startswith('cuda')

        if use_gpu:
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)
        else:
            tracemalloc.start()

        t_start = time.perf_counter()
        pred_df = self.predict(df, target_col=target_col, lookback=lookback, horizon=horizon, freq=freq)
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
        return pred_df, inference_time_s, memory_mb

    def detect_pattern(self, pred_df: pd.DataFrame) -> dict:
        """
        Runs talib.CDLENGULFING on predicted OHLCV candles to detect patterns.
        Returns a dictionary containing pattern name, label ('bullish', 'bearish', or 'none'),
        and a heuristic strength confidence.

        NOTE: This confidence score is a heuristic strength score (engulfing body size relative to
        prior candle's range), not a statistical probability, as required for the Guard Pipeline.
        """
        pred_sorted = pred_df.sort_index().copy()

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in pred_sorted.columns for col in required_cols):
            return {"pattern": "CDLENGULFING", "label": "none", "confidence": 0.0}

        o = pred_sorted['open'].astype(float).values
        h = pred_sorted['high'].astype(float).values
        l = pred_sorted['low'].astype(float).values
        c = pred_sorted['close'].astype(float).values

        patterns = talib.CDLENGULFING(o, h, l, c)

        latest_label = "none"
        latest_confidence = 0.0
        latest_day = -1

        for t in range(1, len(patterns)):
            pattern_val = patterns[t]
            if pattern_val != 0:
                curr_body = abs(c[t] - o[t])
                prev_range = max(1e-5, h[t-1] - l[t-1])
                confidence = min(1.0, curr_body / prev_range)
                label = "bullish" if pattern_val > 0 else "bearish"

                latest_label = label
                latest_confidence = confidence
                latest_day = t

        return {
            "pattern": "CDLENGULFING",
            "label": latest_label,
            "confidence": latest_confidence,
            "forecast_step": latest_day
        }