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
    def __init__(self, model_name="NeoQuasar/Kronos-base", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base", quantize_int8: bool = False):
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
        self.is_quantized = False
        self.last_attention_heatmap = None

        if quantize_int8 or os.getenv("KRONOS_QUANTIZE_INT8", "0") == "1":
            self.quantize()

    def quantize(self):
        """
        Applies PyTorch Dynamic INT8 Quantization (Linear layers) to Kronos for CPU inference speedup.
        """
        if "cpu" in str(self.device).lower():
            try:
                print("Applying dynamic INT8 quantization to Kronos model on CPU...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.predictor.model = self.model
                self.is_quantized = True
                print("Kronos dynamic INT8 quantization complete.")
            except Exception as e:
                print(f"[Warning] Kronos INT8 quantization failed: {e}")
        else:
            print(f"Skipping INT8 quantization on device {self.device} (optimized for CPU).")

    def predict(
        self, 
        df: pd.DataFrame, 
        target_col='close', 
        lookback=400, 
        horizon=30, 
        freq='B',
        extract_attention: bool = True
    ) -> pd.DataFrame:
        """
        freq : pandas offset string for future date generation.
               Use 'B' (business days) for equities, 'D' (calendar days) for BTC-USD and
               other assets that trade every day including weekends.
        extract_attention : If True, extracts multihead attention heatmaps from transformer layers.
        """
        df_sorted = df.sort_values('timestamps').copy()

        if target_col not in df_sorted.columns:
            raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")

        # Clean any NaN values in price / volume columns
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df_sorted.columns]
        if price_cols:
            df_sorted = df_sorted.dropna(subset=price_cols)
        if 'volume' in df_sorted.columns:
            df_sorted['volume'] = df_sorted['volume'].fillna(0.0)
            if 'amount' in df_sorted.columns:
                df_sorted['amount'] = df_sorted['close'] * df_sorted['volume']

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

        # Enable attention saving if requested
        if extract_attention and hasattr(self.model, 'transformer') and len(self.model.transformer) > 0:
            for idx in [0, -1]:
                if hasattr(self.model.transformer[idx], 'self_attn'):
                    self.model.transformer[idx].self_attn.save_attention = True

        pred_df = self.predictor.predict(
            df=df_lookback,
            x_timestamp=x_timestamp,
            y_timestamp=future_dates,
            pred_len=horizon
        )

        # Extract attention weights after inference
        if extract_attention and hasattr(self.model, 'transformer') and len(self.model.transformer) > 0:
            self.last_attention_heatmap = self._extract_attention_map(lookback_steps=min(30, lookback))
            # Reset flags
            for idx in [0, -1]:
                if hasattr(self.model.transformer[idx], 'self_attn'):
                    self.model.transformer[idx].self_attn.save_attention = False

        return pred_df

    def _extract_attention_map(self, lookback_steps: int = 30) -> dict:
        """
        Extracts and formats attention weights into a lightweight heatmap dictionary.
        """
        heatmap_dict = {
            "layers": {},
            "lookback_steps": lookback_steps,
            "status": "extracted"
        }
        try:
            for l_idx, l_name in [(0, "layer_0"), (-1, "layer_last")]:
                layer = self.model.transformer[l_idx]
                if hasattr(layer, 'self_attn') and getattr(layer.self_attn, 'last_attn_weights', None) is not None:
                    raw_w = layer.self_attn.last_attn_weights
                    # raw_w shape: [batch, n_heads, seq, seq]
                    if raw_w.dim() == 4:
                        # Average across heads and take batch 0
                        avg_w = raw_w.mean(dim=1)[0].numpy()
                        # Slice the last lookback_steps x lookback_steps
                        sub_matrix = avg_w[-lookback_steps:, -lookback_steps:]
                        heatmap_dict["layers"][l_name] = np.round(sub_matrix, 4).tolist()
            if not heatmap_dict["layers"]:
                heatmap_dict["status"] = "unavailable"
        except Exception as e:
            heatmap_dict["status"] = f"extraction_error: {str(e)}"
        return heatmap_dict

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