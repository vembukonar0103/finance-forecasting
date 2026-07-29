"""
scripts/run_kronos.py
----------------------
Step 2 — Kronos 30-day hold-out validation across the target 5-asset universe:
  AAPL, MSFT, NVDA, SPY, BTC-USD

Loads each ticker, splits into context/holdout, runs predict_timed(), computes
full metric suite, and saves per-ticker Parquet + a consolidated metrics CSV.

Note on BTC-USD: trades 7 days/week, so freq='D' (calendar days) is used
for future-date generation instead of 'B' (business days) to avoid date drift.
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from utils.metrics import compute_all_metrics
from utils.config import DATA_PREDICTIONS_DIR, PLOTS_DIR
from models.kronos_model import KronosForecast
from utils.plotting import plot_forecast

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
HOLDOUT_DAYS = 30
LOOKBACK = 400

# BTC-USD trades every calendar day — use 'D'; equities use business days 'B'
TICKER_FREQ = {t: 'D' if t == 'BTC-USD' else 'B' for t in TICKERS}


def _safe_filename(ticker: str) -> str:
    return ticker.replace("-", "_")


def main():
    os.makedirs(DATA_PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Initializing Kronos Forecast Wrapper (loaded once, reused for all tickers)...")
    model = KronosForecast()

    all_metrics = []

    for ticker in TICKERS:
        freq = TICKER_FREQ[ticker]
        print(f"\n{'='*60}")
        print(f"  Running Kronos on {ticker}  (freq='{freq}')")
        print(f"{'='*60}")

        try:
            df = load_data(ticker, period="2y", interval="1d")
            if df.shape[0] < HOLDOUT_DAYS + 50:
                print(f"  Skipping {ticker}: insufficient data ({df.shape[0]} rows).")
                continue

            df_context = df.iloc[:-HOLDOUT_DAYS]
            df_holdout = df.iloc[-HOLDOUT_DAYS:]
            c0 = df_context['close'].iloc[-1]
            print(f"  Context: {df_context.shape[0]} rows | Holdout: {df_holdout.shape[0]} rows | c0={c0:.2f}")

            # ── Timed inference ───────────────────────────────────────────────
            pred_df, infer_time, mem_mb = model.predict_timed(
                df_context, target_col='close',
                lookback=LOOKBACK, horizon=HOLDOUT_DAYS,
                freq=freq
            )
            print(f"  Inference: {infer_time:.2f}s | Memory: {mem_mb:.2f} MB")

            actuals = df_holdout['close'].values.astype(float)

            # ── Resolve point forecast column ─────────────────────────────────
            if 'close' in pred_df.columns:
                point_col = 'close'
            elif 'point' in pred_df.columns:
                point_col = 'point'
            else:
                numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric column found in Kronos output.")
                point_col = numeric_cols[0]

            pred_df_metrics = pred_df.rename(columns={point_col: 'point'})
            metrics = compute_all_metrics(actuals, pred_df_metrics, c0, point_col='point')

            # ── Detect candlestick pattern ────────────────────────────────────
            pattern = model.detect_pattern(pred_df)

            # ── Console output ────────────────────────────────────────────────
            print(f"\n  {ticker} Results:")
            print(f"    MAE                 : {metrics['mae']:.4f}")
            print(f"    RMSE                : {metrics['rmse']:.4f}")
            print(f"    MAPE                : {metrics['mape']:.2f}%")
            print(f"    Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
            pb50_display = 'N/A (point-only)' if np.isnan(metrics['pinball_p50']) else f"{metrics['pinball_p50']:.4f}"
            print(f"    Pinball P50         : {pb50_display}")
            print(f"    Pattern (CDLENGLF)  : {pattern['label']} (conf={pattern['confidence']:.2f}, step={pattern['forecast_step']})")
            print(f"    Inference Time      : {infer_time:.3f}s")
            print(f"    Peak Memory         : {mem_mb:.2f} MB")

            # ── Save predictions (Parquet + CSV) ──────────────────────────────
            safe = _safe_filename(ticker)
            save_df = pred_df.copy()
            save_df['ticker'] = ticker
            save_df['model'] = 'kronos'
            save_df['run_timestamp'] = pd.Timestamp.now(tz='UTC').isoformat()

            parquet_path = os.path.join(DATA_PREDICTIONS_DIR, f"{safe}_kronos_predictions.parquet")
            save_df.to_parquet(parquet_path)
            print(f"\n  Predictions saved → {parquet_path}")

            csv_path = os.path.join(DATA_PREDICTIONS_DIR, f"{safe}_kronos_predictions.csv")
            save_df.to_csv(csv_path, index=True)
            print(f"  Predictions saved → {csv_path}")

            # ── Save plot ─────────────────────────────────────────────────────
            plot_path = os.path.join(PLOTS_DIR, f"{ticker}_kronos_forecast.png")
            if 'timestamps' in pred_df.columns:
                pred_dates = pd.to_datetime(pred_df['timestamps'])
            else:
                pred_dates = pd.to_datetime(pred_df.index)
            pred_values = pred_df[point_col]
            actual_dates = pd.to_datetime(df['timestamps'].tail(100))
            actual_values = df['close'].tail(100)

            plot_forecast(
                actual_dates=actual_dates,
                actual_values=actual_values,
                pred_dates=pred_dates,
                pred_values=pred_values,
                output_path=plot_path,
                title=f"{ticker} 30-Day Kronos Forecast | MAE={metrics['mae']:.2f} DA={metrics['directional_accuracy']:.1f}%"
            )
            print(f"  Plot saved        → {plot_path}")

            all_metrics.append({
                "ticker": ticker,
                "model": "kronos",
                "context_size": df_context.shape[0],
                "inference_time_s": round(infer_time, 3),
                "memory_mb": round(mem_mb, 2),
                "pattern_label": pattern['label'],
                "pattern_confidence": round(pattern['confidence'], 4),
                **{k: round(v, 4) if not np.isnan(v) else float('nan') for k, v in metrics.items()},
            })

        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")
            import traceback; traceback.print_exc()

    # ── Consolidated metrics summary ──────────────────────────────────────────
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(DATA_PREDICTIONS_DIR, "kronos_validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "="*80)
        print("  KRONOS VALIDATION SUMMARY (all tickers)")
        print("="*80)
        display_cols = ['ticker', 'mae', 'rmse', 'mape', 'directional_accuracy',
                        'inference_time_s', 'memory_mb', 'pattern_label']
        avail = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[avail].to_string(index=False))
        print("="*80)
        print(f"\nSummary CSV saved → {summary_path}")
        print(f"\nExpected outputs in data/predictions/:")
        for ticker in TICKERS:
            safe = _safe_filename(ticker)
            print(f"  {safe}_kronos_predictions.parquet")
            print(f"  {safe}_kronos_predictions.csv")
        print(f"  kronos_validation_summary.csv")


if __name__ == "__main__":
    main()
