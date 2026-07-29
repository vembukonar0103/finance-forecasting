"""
scripts/benchmark_comparison.py
---------------------------------
Day 2 — Model Comparison Harness

Runs both Kronos and TimesFM on 5 assets and reports:
  • MAE, RMSE, MAPE, Directional Accuracy
  • Pinball Loss at p10, p25, p50, p75, p90 (TimesFM only)
  • Inference time (s) and peak memory (MB)

Outputs:
  data/predictions/benchmark_results.csv          — summary (1 row per model × ticker)
  data/predictions/{ticker}_{model}_predictions.parquet
  plots/benchmark_comparison.png                   — metric bar charts
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.data_loader import load_data
from utils.metrics import compute_all_metrics
from utils.config import DATA_PREDICTIONS_DIR, PLOTS_DIR
from models.kronos_model import KronosForecast
from models.timesfm_model import TimesFMForecast

# ─── Configuration ────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
HOLDOUT_DAYS = 30
LOOKBACK = 400  # for Kronos

# BTC-USD trades 7 days/week — use calendar days to avoid date alignment drift
TICKER_FREQ = {t: 'D' if t == 'BTC-USD' else 'B' for t in TICKERS}


def _safe_parquet_ticker(ticker: str) -> str:
    """Replace characters not suitable for filenames."""
    return ticker.replace("-", "_")


def run_kronos_on_ticker(kronos: KronosForecast, ticker: str, df: pd.DataFrame, df_context: pd.DataFrame, df_holdout: pd.DataFrame) -> dict:
    """Run Kronos on one ticker and return result dict."""
    freq = TICKER_FREQ.get(ticker, 'B')
    print(f"  [Kronos] Running on {ticker} (freq='{freq}')...")
    try:
        pred_df, infer_time, mem_mb = kronos.predict_timed(
            df_context, target_col='close', lookback=LOOKBACK, horizon=HOLDOUT_DAYS,
            freq=freq
        )

        actuals = df_holdout['close'].values.astype(float)
        c0 = df_context['close'].iloc[-1]

        # Resolve point forecast column from Kronos output
        if 'close' in pred_df.columns:
            point_col = 'close'
        elif 'point' in pred_df.columns:
            point_col = 'point'
        else:
            numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
            point_col = numeric_cols[0] if numeric_cols else None

        if point_col is None:
            raise ValueError("Cannot find numeric forecast column in Kronos output.")

        # Rename to 'point' for compute_all_metrics
        pred_df_metrics = pred_df.rename(columns={point_col: 'point'})
        metrics = compute_all_metrics(actuals, pred_df_metrics, c0, point_col='point')

        # Save predictions to Parquet
        safe = _safe_parquet_ticker(ticker)
        out_path = os.path.join(DATA_PREDICTIONS_DIR, f"{safe}_kronos_predictions.parquet")
        pred_df['ticker'] = ticker
        pred_df['model'] = 'kronos'
        pred_df['run_timestamp'] = pd.Timestamp.now(tz='UTC').isoformat()
        pred_df.to_parquet(out_path)
        print(f"  [Kronos] Saved predictions → {out_path}")

        return {
            "model": "kronos",
            "ticker": ticker,
            "inference_time_s": round(infer_time, 3),
            "memory_mb": round(mem_mb, 2),
            **{k: round(v, 4) if not np.isnan(v) else float('nan') for k, v in metrics.items()},
        }

    except Exception as e:
        print(f"  [Kronos] ERROR on {ticker}: {e}")
        return {"model": "kronos", "ticker": ticker, "error": str(e)}


def run_timesfm_on_ticker(tfm: TimesFMForecast, ticker: str, df: pd.DataFrame, df_context: pd.DataFrame, df_holdout: pd.DataFrame) -> dict:
    """Run TimesFM on one ticker and return result dict."""
    print(f"  [TimesFM] Running on {ticker}...")
    try:
        pred_df, prob_plus5, infer_time, mem_mb = tfm.predict_timed(
            df_context, target_col='close', horizon=HOLDOUT_DAYS
        )

        actuals = df_holdout['close'].values.astype(float)
        c0 = df_context['close'].iloc[-1]

        metrics = compute_all_metrics(actuals, pred_df, c0, point_col='point')

        # Save predictions to Parquet
        safe = _safe_parquet_ticker(ticker)
        out_path = os.path.join(DATA_PREDICTIONS_DIR, f"{safe}_timesfm_predictions.parquet")
        save_df = pred_df.copy()
        save_df['ticker'] = ticker
        save_df['model'] = 'timesfm'
        save_df['prob_plus5pct'] = prob_plus5
        save_df['run_timestamp'] = pd.Timestamp.now(tz='UTC').isoformat()
        save_df.to_parquet(out_path)
        print(f"  [TimesFM] Saved predictions → {out_path}")

        return {
            "model": "timesfm",
            "ticker": ticker,
            "inference_time_s": round(infer_time, 3),
            "memory_mb": round(mem_mb, 2),
            **{k: round(v, 4) if not np.isnan(v) else float('nan') for k, v in metrics.items()},
            "prob_plus5pct": round(prob_plus5, 4),
        }

    except Exception as e:
        print(f"  [TimesFM] ERROR on {ticker}: {e}")
        return {"model": "timesfm", "ticker": ticker, "error": str(e)}


def build_comparison_chart(results_df: pd.DataFrame) -> None:
    """Generate a 2×3 grid of bar charts comparing models across metrics."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    METRICS = [
        ("mae", "MAE (↓ better)", False),
        ("rmse", "RMSE (↓ better)", False),
        ("mape", "MAPE % (↓ better)", False),
        ("directional_accuracy", "Directional Accuracy % (↑ better)", True),
        ("inference_time_s", "Inference Time s (↓ better)", False),
        ("memory_mb", "Peak Memory MB (↓ better)", False),
    ]

    models = results_df['model'].unique().tolist()
    colors = {"kronos": "#4F86F7", "timesfm": "#FF6B6B"}

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#0F1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (col, label, higher_better) in enumerate(METRICS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor('#1A1D27')
        ax.tick_params(colors='#C0C0C0')
        ax.spines[:].set_color('#333344')

        x = np.arange(len(TICKERS))
        bar_width = 0.35

        for m_idx, model in enumerate(models):
            model_df = results_df[results_df['model'] == model]
            vals = []
            for tk in TICKERS:
                row = model_df[model_df['ticker'] == tk]
                if row.empty or col not in row.columns or pd.isna(row[col].values[0]):
                    vals.append(0.0)
                else:
                    vals.append(float(row[col].values[0]))

            offset = (m_idx - 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals,
                width=bar_width,
                label=model.capitalize(),
                color=colors.get(model, '#AAAAAA'),
                alpha=0.85,
                edgecolor='none',
                zorder=3
            )
            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{v:.1f}",
                        ha='center', va='bottom',
                        fontsize=7, color='#C0C0C0'
                    )

        ax.set_title(label, fontsize=10, color='#E0E0E0', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(TICKERS, fontsize=8, color='#C0C0C0', rotation=20, ha='right')
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()], fontsize=7, color='#C0C0C0')
        ax.grid(axis='y', linestyle=':', alpha=0.3, color='#555566', zorder=0)
        ax.legend(fontsize=8, facecolor='#1A1D27', edgecolor='#333344', labelcolor='#C0C0C0')

    fig.suptitle(
        "Kronos vs TimesFM — 30-Day Forecast Benchmark (5 Assets)",
        fontsize=15, fontweight='bold', color='#FFFFFF', y=0.98
    )

    out_path = os.path.join(PLOTS_DIR, "benchmark_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nBenchmark chart saved → {out_path}")


def print_summary_table(results_df: pd.DataFrame) -> None:
    DISPLAY_COLS = ['model', 'ticker', 'mae', 'rmse', 'mape',
                    'directional_accuracy', 'inference_time_s', 'memory_mb',
                    'pinball_p10', 'pinball_p50', 'pinball_p90']
    cols = [c for c in DISPLAY_COLS if c in results_df.columns]
    display_df = results_df[cols].copy()

    float_cols = [c for c in cols if c not in ('model', 'ticker')]
    for c in float_cols:
        display_df[c] = display_df[c].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
        )

    print("\n" + "=" * 100)
    print("  BENCHMARK SUMMARY — Kronos vs TimesFM — 5 Assets")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)


def main():
    print("=" * 70)
    print("  BENCHMARK: Kronos vs TimesFM  |  5 Assets  |  30-day holdout")
    print("=" * 70)

    # --- Load data for all tickers (done once) ---
    print("\n[1/4] Loading market data...")
    ticker_data: dict[str, tuple] = {}
    for ticker in TICKERS:
        try:
            df = load_data(ticker, period="2y", interval="1d", force_download=False)
            if df.shape[0] < HOLDOUT_DAYS + 50:
                print(f"  Skipping {ticker}: insufficient data ({df.shape[0]} rows).")
                continue
            df_context = df.iloc[:-HOLDOUT_DAYS]
            df_holdout = df.iloc[-HOLDOUT_DAYS:]
            ticker_data[ticker] = (df, df_context, df_holdout)
            print(f"  {ticker}: {df.shape[0]} rows loaded | context={df_context.shape[0]} | holdout={df_holdout.shape[0]}")
        except Exception as e:
            print(f"  ERROR loading {ticker}: {e}")

    results = []

    # --- Kronos benchmark ---
    print("\n[2/4] Running Kronos inference...")
    kronos = KronosForecast()
    for ticker, (df, df_context, df_holdout) in ticker_data.items():
        res = run_kronos_on_ticker(kronos, ticker, df, df_context, df_holdout)
        results.append(res)

    # --- TimesFM benchmark ---
    print("\n[3/4] Running TimesFM inference...")
    tfm = TimesFMForecast()
    for ticker, (df, df_context, df_holdout) in ticker_data.items():
        res = run_timesfm_on_ticker(tfm, ticker, df, df_context, df_holdout)
        results.append(res)

    # --- Persist & report ---
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(results)

    csv_path = os.path.join(DATA_PREDICTIONS_DIR, "benchmark_results.csv")
    results_df['run_timestamp'] = pd.Timestamp.now(tz='UTC').isoformat()
    results_df.to_csv(csv_path, index=False)
    print(f"  Benchmark CSV saved → {csv_path}")

    # Chart (only rows without errors)
    if 'error' in results_df.columns:
        clean_df = results_df[results_df['error'].isna()].copy()
    else:
        clean_df = results_df.copy()

    # --- Chart generation: 2x3 grid, Kronos vs TimesFM per metric ---
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('dark_background')

    metrics_to_plot = [
        ('mae', 'MAE'),
        ('rmse', 'RMSE'),
        ('mape', 'MAPE (%)'),
        ('directional_accuracy', 'Directional Accuracy (%)'),
        ('inference_time_s', 'Inference Time (s)'),
        ('memory_mb', 'Memory (MB)'),
    ]

    tickers = sorted(clean_df['ticker'].unique())
    x = np.arange(len(tickers))
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Kronos vs TimesFM — Benchmark Comparison (5 Assets, 30-day holdout)", fontsize=14)

    for ax, (col, label) in zip(axes.flat, metrics_to_plot):
        kronos_vals, timesfm_vals = [], []
        for t in tickers:
            row_k = clean_df[(clean_df['ticker'] == t) & (clean_df['model'] == 'kronos')]
            row_t = clean_df[(clean_df['ticker'] == t) & (clean_df['model'] == 'timesfm')]
            kronos_vals.append(row_k[col].values[0] if not row_k.empty and col in row_k else np.nan)
            timesfm_vals.append(row_t[col].values[0] if not row_t.empty and col in row_t else np.nan)

        ax.bar(x - width/2, kronos_vals, width, label='Kronos', color='#e07a5f')
        ax.bar(x + width/2, timesfm_vals, width, label='TimesFM', color='#3a86ff')
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Benchmark chart saved → {chart_path}")

    print_summary_table(results_df)
    print(f"\nAll outputs written to:\n  {DATA_PREDICTIONS_DIR}\n  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
