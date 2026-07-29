"""
scripts/run_timesfm_validation.py
----------------------------------
TimesFM 30-day hold-out validation across the target 5-asset universe:
  AAPL, MSFT, NVDA, SPY, BTC-USD

Metrics computed per ticker:
  • MAE, RMSE, MAPE
  • Directional Accuracy
  • P10-P90 Band Hit Rate
  • Pinball Loss at p10, p25, p50, p75, p90
  • Probability P(≥ +5% in 30 days)

Outputs:
  data/predictions/{ticker}_timesfm_predictions.parquet   — full forecast frames with timestamps
  data/predictions/timesfm_validation_summary.csv         — aggregated metrics table
  plots/{ticker}_timesfm_validation.png                    — per-ticker forecast charts
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
from models.timesfm_model import TimesFMForecast
from utils.plotting import plot_forecast

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
HOLDOUT_DAYS = 30


def _safe_parquet_ticker(ticker: str) -> str:
    return ticker.replace("-", "_")


def main():
    os.makedirs(DATA_PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Initializing TimesFM Forecast Wrapper...")
    model = TimesFMForecast()

    results = []

    print(f"\nStarting Validation across {len(TICKERS)} tickers: {TICKERS}\n")
    for ticker in TICKERS:
        print(f"{'─'*60}")
        print(f"  Validating: {ticker}")
        print(f"{'─'*60}")
        try:
            # Download 2 years of daily data
            df = load_data(ticker, period="2y", interval="1d")
            if df.shape[0] < HOLDOUT_DAYS + 50:
                print(f"  Skipping {ticker}: insufficient data ({df.shape[0]} rows).")
                continue

            df_context = df.iloc[:-HOLDOUT_DAYS]
            df_holdout = df.iloc[-HOLDOUT_DAYS:]
            c0 = df_context['close'].iloc[-1]
            print(f"  Context: {df_context.shape[0]} rows | Holdout: {df_holdout.shape[0]} rows | c0={c0:.2f}")

            # ── Timed inference ──────────────────────────────────────────────
            pred_df, prob_plus_5, infer_time, mem_mb = model.predict_timed(
                df_context, target_col='close', horizon=HOLDOUT_DAYS
            )
            print(f"  Inference: {infer_time:.2f}s | Memory: {mem_mb:.2f} MB")

            actuals = df_holdout['close'].values.astype(float)

            # ── Core metrics ─────────────────────────────────────────────────
            metrics = compute_all_metrics(actuals, pred_df, c0, point_col='point')

            # ── Band hit rate (actuals inside P10-P90) ───────────────────────
            p10_vals = pred_df['p10'].values
            p90_vals = pred_df['p90'].values
            hits = sum(1 for act, lo, hi in zip(actuals, p10_vals, p90_vals) if lo <= act <= hi)
            hit_rate = hits / HOLDOUT_DAYS

            # ── Day-30 details ───────────────────────────────────────────────
            day30_idx = HOLDOUT_DAYS - 1
            day30_actual_return = (actuals[-1] - c0) / c0

            result = {
                "ticker": ticker,
                "context_size": df_context.shape[0],
                "hit_rate_p10_p90": round(hit_rate, 4),
                "prob_plus5pct": round(prob_plus_5, 4),
                "actual_day30_return": round(day30_actual_return, 4),
                "inference_time_s": round(infer_time, 3),
                "memory_mb": round(mem_mb, 2),
                **{k: round(v, 4) if not np.isnan(v) else float('nan') for k, v in metrics.items()},
            }
            results.append(result)

            # ── Per-ticker console output ─────────────────────────────────────
            print(f"\n  {ticker} Results:")
            print(f"    MAE                 : {metrics['mae']:.4f}")
            print(f"    RMSE                : {metrics['rmse']:.4f}")
            print(f"    MAPE                : {metrics['mape']:.2f}%")
            print(f"    Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
            print(f"    P10-P90 Hit Rate    : {hit_rate*100:.1f}%")
            print(f"    Pinball P10         : {metrics['pinball_p10']:.4f}")
            print(f"    Pinball P25         : {metrics['pinball_p25']:.4f}")
            print(f"    Pinball P50         : {metrics['pinball_p50']:.4f}")
            print(f"    Pinball P75         : {metrics['pinball_p75']:.4f}")
            print(f"    Pinball P90         : {metrics['pinball_p90']:.4f}")
            print(f"    P(≥+5% in 30d)      : {prob_plus_5*100:.1f}%")
            print(f"    Actual day-30 return: {day30_actual_return*100:.1f}%")

            # ── Save per-ticker Parquet ───────────────────────────────────────
            safe = _safe_parquet_ticker(ticker)
            parquet_path = os.path.join(DATA_PREDICTIONS_DIR, f"{safe}_timesfm_predictions.parquet")
            save_df = pred_df.copy()
            save_df['ticker'] = ticker
            save_df['model'] = 'timesfm'
            save_df['prob_plus5pct'] = prob_plus_5
            save_df['run_timestamp'] = pd.Timestamp.now(tz='UTC').isoformat()
            save_df.to_parquet(parquet_path)
            print(f"\n  Predictions saved → {parquet_path}")

            # ── Plot ─────────────────────────────────────────────────────────
            plot_path = os.path.join(PLOTS_DIR, f"{ticker}_timesfm_validation.png")
            plot_actuals = df.tail(100)
            actual_dates = pd.to_datetime(plot_actuals['timestamps'])
            actual_values = plot_actuals['close']
            pred_dates = pd.to_datetime(pred_df.index)
            quantiles_dict = {
                'p10': pred_df['p10'].values,
                'p25': pred_df['p25'].values,
                'p75': pred_df['p75'].values,
                'p90': pred_df['p90'].values,
            }
            plot_forecast(
                actual_dates=actual_dates,
                actual_values=actual_values,
                pred_dates=pred_dates,
                pred_values=pred_df['point'].values,
                output_path=plot_path,
                title=f"{ticker} 30-Day TimesFM | MAE={metrics['mae']:.2f} DA={metrics['directional_accuracy']:.1f}% Hit={hit_rate*100:.0f}%",
                quantiles=quantiles_dict
            )
            print(f"  Plot saved → {plot_path}")

        except Exception as e:
            print(f"  ERROR validating {ticker}: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  TIMESFM VALIDATION SUMMARY")
    print("=" * 90)
    summary_df = pd.DataFrame(results)

    display_cols = [
        'ticker', 'mae', 'rmse', 'mape', 'directional_accuracy',
        'hit_rate_p10_p90', 'pinball_p10', 'pinball_p50', 'pinball_p90',
        'prob_plus5pct', 'actual_day30_return', 'inference_time_s', 'memory_mb'
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    display_df = summary_df[available_cols].copy()

    # hit_rate_p10_p90, prob_plus5pct, actual_day30_return are stored as fractions (0–1) → multiply ×100
    # mape and directional_accuracy are already stored as percent values (e.g. 4.07, 56.7) → do NOT multiply
    pct_cols = {'hit_rate_p10_p90', 'prob_plus5pct', 'actual_day30_return'}
    for col in available_cols:
        if col == 'ticker':
            continue
        display_df[col] = display_df[col].apply(
            lambda x: f"{x*100:.1f}%" if col in pct_cols and not pd.isna(x) else (f"{x:.4f}" if not pd.isna(x) else "N/A")
        )

    print(display_df.to_string(index=False))
    print("=" * 90)

    # ── Persist summary CSV ───────────────────────────────────────────────────
    summary_path = os.path.join(DATA_PREDICTIONS_DIR, "timesfm_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved → {summary_path}")


if __name__ == "__main__":
    main()
