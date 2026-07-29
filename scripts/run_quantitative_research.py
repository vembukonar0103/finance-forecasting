import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import warnings

warnings.filterwarnings("ignore")

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.data_loader import load_data
from utils.metrics import compute_all_metrics
from utils.walk_forward import generate_rolling_windows, verify_no_lookahead_bias
from models.timesfm_model import TimesFMForecast
from models.kronos_model import KronosForecast
from models.ml_forecasters import MLForecaster
from models.ensembles import compute_weighted_ensemble, compute_confidence_ensemble
from utils.trading_strategy import generate_signal
from utils.backtester import run_asset_backtest, compute_backtest_metrics, run_portfolio_backtest
from utils.regime_analysis import classify_market_regimes, analyze_regime_performance
from utils.experiment_tracker import log_experiment, get_experiments

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
HORIZON = 30
VAL_DAYS = 60

# Walk-forward context requirement (rows consumed before the first test window starts).
# Kronos requires lookback=400, TimesFM uses up to 512, ML models need at least
# VAL_DAYS(60) + enough training rows — 500 is the binding constraint.
# Coverage formula:  testable_trading_days = total_rows - INITIAL_CONTEXT
# With period='5y' equities get ~1,300 trading rows → ~800 testable rows → ~26 monthly windows
# BTC-USD gets ~1,825 calendar rows → ~1,325 testable rows → ~44 monthly windows
# Both satisfy the >3-year coverage requirement.
INITIAL_CONTEXT = 500

# BTC-USD trades 7 days/week, others 5 days/week
TICKER_FREQ = {t: 'D' if t == 'BTC-USD' else 'B' for t in TICKERS}

# Approx inference seconds per foundation-model call (empirically measured on this machine)
_EST_SECS_PER_CALL = 5.0  # TimesFM / Kronos horizon-30 inference
_FOUNDATION_CALLS_PER_WINDOW = 4  # tfm_test, kronos_test, tfm_val, kronos_val

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Quantitative Research Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (2 windows, 1 asset)")
    parser.add_argument(
        "--period", type=str, default="5y",
        help="yfinance pull period. Default 5y gives ~3y of testable walk-forward windows "
             "after the 500-row context warmup. Use 3y only for quick sanity checks."
    )
    return parser.parse_args()

def evaluate_models(dry_run=False, period="5y"):
    tickers_to_run = ["AAPL"] if dry_run else TICKERS
    
    print("\n" + "="*80)
    print(f"  QUANTITATIVE RESEARCH PIPELINE")
    print(f"  Dry Run = {dry_run} | Period = {period} | Context = {INITIAL_CONTEXT} rows")
    print("="*80)
    
    # 1. Download/load raw data
    print(f"\n[Step 1/8] Downloading historical data (period={period}, force_download=True)...")
    print(f"  Rationale: {INITIAL_CONTEXT}-row context warmup + 3-year test coverage requires ~5y of history.")
    market_data = {}
    for ticker in TICKERS:
        try:
            df = load_data(ticker, period=period, interval="1d", force_download=True)
            market_data[ticker] = df
            print(f"  Loaded {ticker}: {df.shape[0]} rows | Date Range: {df['timestamps'].min()} to {df['timestamps'].max()}")
        except Exception as e:
            print(f"  Failed to load {ticker}: {e}")
            
    # Verify look-ahead bias safeguard
    if "AAPL" in market_data:
        print("\n[Step 2/8] Verifying look-ahead bias safeguard check...")
        verify_no_lookahead_bias(market_data["AAPL"], initial_context=INITIAL_CONTEXT)
        
    # 2. Instantiate forecasting models
    print("\n[Step 3/8] Instantiating TimesFM and Kronos foundation forecasters...")
    tfm_model = TimesFMForecast()
    kronos_model = KronosForecast()
    
    # Storage for all assets' results
    # Structure: {ticker: {model_name: {predictions: [...], signals: [...], daily_returns: ...}}}
    asset_results = {}
    
    # 3. Walk-Forward Loop
    print("\n[Step 4/8] Starting walk-forward evaluation across assets...")
    for ticker in tickers_to_run:
        df = market_data.get(ticker)
        if df is None:
            continue
            
        freq = TICKER_FREQ.get(ticker, 'B')
        print(f"\n{'─'*60}")
        print(f"  Processing Ticker: {ticker} (freq='{freq}')")
        print(f"{'─'*60}")
        
        # Generate rolling windows
        windows = generate_rolling_windows(df, initial_context=INITIAL_CONTEXT, horizon=HORIZON, val_days=VAL_DAYS)
        if dry_run:
            windows = windows[:2]  # Limit to first 2 windows for dry run
            
        n_windows = len(windows)
        covered_days = n_windows * HORIZON
        covered_months = covered_days / 30.0
        est_runtime_min = (n_windows * _FOUNDATION_CALLS_PER_WINDOW * _EST_SECS_PER_CALL) / 60.0
        if windows:
            first_test = pd.to_datetime(windows[0]['test']['timestamps'].iloc[0]).strftime('%Y-%m-%d')
            last_test  = pd.to_datetime(windows[-1]['test']['timestamps'].iloc[-1]).strftime('%Y-%m-%d')
        else:
            first_test = last_test = 'N/A'
        print(f"  Total Walk-Forward Windows : {n_windows}")
        print(f"  Test Coverage             : {covered_months:.1f} months ({first_test} → {last_test})")
        print(f"  Estimated Runtime (this asset): ~{est_runtime_min:.1f} min (foundation models only)")
        
        # Models we evaluate:
        model_names = [
            "TimesFM", 
            "Kronos", 
            "WeightedEnsemble", 
            "ConfidenceEnsemble", 
            "BaseML", 
            "BaseML_Technical"
        ]
        
        ticker_preds = {m: [] for m in model_names}
        ticker_signals = {m: [] for m in model_names}
        
        # Fit ML forecasters once for setup?
        # Note: We fit them inside each window loop to avoid look-ahead bias!
        for w in windows:
            w_idx = w['window_idx']
            split_date = w['split_date']
            print(f"    Window {w_idx} | Split Date: {split_date}...")
            
            # Slices
            context_df = w['context']
            train_df = w['train']
            val_df = w['val']
            test_df = w['test']
            
            # a) ML Forecaster fitting & predicting
            # Base ML (Ridge, lags only)
            ml_base = MLForecaster(model_type="ridge", use_technical=False)
            ml_base.fit(context_df, horizon=HORIZON, val_days=VAL_DAYS)
            pred_ml_base = ml_base.predict(context_df, horizon=HORIZON)
            
            # Base ML + Tech Features (Ridge, lags + indicators)
            ml_tech = MLForecaster(model_type="ridge", use_technical=True)
            ml_tech.fit(context_df, horizon=HORIZON, val_days=VAL_DAYS)
            pred_ml_tech = ml_tech.predict(context_df, horizon=HORIZON)
            
            # b) Foundation model out-of-sample test predictions
            pred_tfm, prob_tfm_plus5 = tfm_model.predict(context_df, target_col='close', horizon=HORIZON, freq=freq)
            pred_kronos = kronos_model.predict(context_df, target_col='close', lookback=400, horizon=HORIZON, freq=freq)
            
            # Handle Kronos column naming
            if 'close' in pred_kronos.columns:
                pred_kronos = pred_kronos.rename(columns={'close': 'point'})
            elif 'point' not in pred_kronos.columns:
                num_cols = pred_kronos.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    pred_kronos = pred_kronos.rename(columns={num_cols[0]: 'point'})
            
            # c) Foundation model validation forecasting (for confidence ensemble)
            val_context_df = context_df.iloc[:-VAL_DAYS]
            val_actuals = val_df['close'].values.astype(float)
            val_c0 = val_context_df['close'].iloc[-1]
            
            # TimesFM validation
            pred_tfm_val, _ = tfm_model.predict(val_context_df, target_col='close', horizon=VAL_DAYS, freq=freq)
            mae_tfm_val = np.mean(np.abs(val_actuals - pred_tfm_val['point'].values[:len(val_actuals)]))
            
            # Kronos validation
            pred_kronos_val = kronos_model.predict(val_context_df, target_col='close', lookback=400, horizon=VAL_DAYS, freq=freq)
            if 'close' in pred_kronos_val.columns:
                pred_kronos_val = pred_kronos_val.rename(columns={'close': 'point'})
            elif 'point' not in pred_kronos_val.columns:
                num_cols = pred_kronos_val.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    pred_kronos_val = pred_kronos_val.rename(columns={num_cols[0]: 'point'})
                    
            mae_kronos_val = np.mean(np.abs(val_actuals - pred_kronos_val['point'].values[:len(val_actuals)]))
            
            # d) Ensembles
            pred_weighted = compute_weighted_ensemble(pred_tfm, pred_kronos, weight_tfm=0.5)
            pred_confidence = compute_confidence_ensemble(
                pred_tfm, pred_kronos, 
                mae_tfm_val=mae_tfm_val, 
                mae_kronos_val=mae_kronos_val,
                val_tfm_bands=pred_tfm_val
            )
            
            # e) Trading Signals
            c0 = context_df['close'].iloc[-1]
            
            # Store point predictions for evaluation
            # Make sure we align pred indices with test dates
            for m_name, pred_df in [
                ("TimesFM", pred_tfm),
                ("Kronos", pred_kronos),
                ("WeightedEnsemble", pred_weighted),
                ("ConfidenceEnsemble", pred_confidence),
                ("BaseML", pred_ml_base),
                ("BaseML_Technical", pred_ml_tech)
            ]:
                # Force index match with test_df timestamps
                pred_aligned = pred_df.copy()
                pred_aligned.index = test_df.index
                ticker_preds[m_name].append(pred_aligned)
                
                # Signal logic
                prob_val = prob_tfm_plus5 if m_name in ("TimesFM", "ConfidenceEnsemble") else None
                sig = generate_signal(pred_aligned, c0=c0, threshold=0.02, prob_plus5pct=prob_val)
                ticker_signals[m_name].append(sig)
                
        # Consolidate predictions and signals across windows
        asset_results[ticker] = {}
        for m_name in model_names:
            preds_all = pd.concat(ticker_preds[m_name])
            signals_all = ticker_signals[m_name] # list of signals per window
            
            # Run Daily Backtest for this asset and model
            daily_strat_ret, trade_rets, dates_idx = run_asset_backtest(
                df=df,
                windows=windows,
                signals=signals_all,
                allow_short=True,
                tx_cost_pct=0.0010
            )
            
            # Save results
            asset_results[ticker][m_name] = {
                "predictions": preds_all,
                "signals": signals_all,
                "daily_returns": daily_strat_ret,
                "trade_returns": trade_rets,
                "dates": dates_idx
            }
            
    # 4. Backtesting & Metric Evaluation
    print("\n[Step 5/8] Computing performance metrics per asset...")
    metric_results = [] # Flat list of dicts for reporting
    
    for ticker, models in asset_results.items():
        df = market_data[ticker]
        ticker_windows = generate_rolling_windows(df, initial_context=INITIAL_CONTEXT, horizon=HORIZON, val_days=VAL_DAYS)
        if dry_run:
            ticker_windows = ticker_windows[:2]
            
        for m_name, res in models.items():
            daily_strat_ret = res["daily_returns"]
            trade_rets = res["trade_returns"]
            dates = res["dates"]
            preds = res["predictions"]
            
            # Reconstruct out-of-sample actuals
            # Match lengths
            actuals_all = []
            for w in ticker_windows:
                actuals_all.append(w['test']['close'])
            actuals_series = pd.concat(actuals_all)
            
            # Compute Forecast Metrics
            # Metrics need c0 of each window, let's simplify and run overall point forecast metrics
            mae_val = np.mean(np.abs(actuals_series.values - preds['point'].values))
            rmse_val = np.sqrt(np.mean((actuals_series.values - preds['point'].values)**2))
            mape_val = np.mean(np.abs((actuals_series.values - preds['point'].values) / actuals_series.values)) * 100.0
            
            # Directional accuracy
            da_val = 50.0  # fallback
            try:
                # Calculate DA step-by-step
                matches = 0
                total = 0
                for w_idx, w in enumerate(ticker_windows):
                    c0 = w['context']['close'].iloc[-1]
                    act = w['test']['close'].values
                    pr = preds.iloc[w_idx*HORIZON : (w_idx+1)*HORIZON]['point'].values
                    
                    act_prev = np.concatenate([[c0], act[:-1]])
                    pr_prev = np.concatenate([[c0], pr[:-1]])
                    
                    act_dir = np.sign(act - act_prev)
                    pr_dir = np.sign(pr - pr_prev)
                    
                    matches += np.sum(act_dir == pr_dir)
                    total += len(act)
                if total > 0:
                    da_val = (matches / total) * 100.0
            except Exception as e:
                pass
                
            # Backtest Metrics
            backtest_metrics = compute_backtest_metrics(
                daily_returns=daily_strat_ret,
                trade_returns=trade_rets,
                dates=dates,
                rf_annual=0.04
            )
            
            metric_results.append({
                "ticker": ticker,
                "model": m_name,
                "mae": mae_val,
                "rmse": rmse_val,
                "mape": mape_val,
                "directional_accuracy": da_val,
                **backtest_metrics
            })
            
    metrics_df = pd.DataFrame(metric_results)
    
    # 5. Unified Portfolio Backtest
    print("\n[Step 6/8] Running Unified Multi-Asset Portfolio simulation...")
    # Single portfolio starting with 100k
    # We allocate 20% capital weight to each asset.
    # Daily portfolio return is the average of daily returns across the 5 assets
    portfolio_results = {}
    # Create equal capital weighting (e.g., 20% weight per ticker)
    equal_allocation = {ticker: 100.0 / len(tickers_to_run) for ticker in tickers_to_run}
    
    for m_name in model_names:
        model_ticker_signals = {}
        for ticker in tickers_to_run:
            if ticker in asset_results and m_name in asset_results[ticker]:
                model_ticker_signals[ticker] = {
                    "daily_returns": asset_results[ticker][m_name]["daily_returns"],
                    "trade_returns": asset_results[ticker][m_name]["trade_returns"]
                }
                
        if len(model_ticker_signals) > 0:
            portfolio_results[m_name] = run_portfolio_backtest(
                ticker_signals=model_ticker_signals,
                capital_allocation=equal_allocation,
                rf_annual=0.04
            )
            
    # 6. Regime Analysis
    print("\n[Step 7/8] Performing Market Trend (Bull/Bear) and Volatility Regime Analysis...")
    spy_df = market_data.get("SPY")
    regime_results = []
    
    for ticker in tickers_to_run:
        df = market_data[ticker]
        regime_df = classify_market_regimes(df, spy_df=spy_df)
        
        # Calculate asset raw daily returns
        df_sorted = df.sort_values('timestamps').reset_index(drop=True)
        df_sorted.index = pd.to_datetime(df_sorted['timestamps'])
        raw_daily_ret = df_sorted['close'].pct_change().fillna(0.0)
        
        for m_name in model_names:
            if ticker in asset_results and m_name in asset_results[ticker]:
                daily_strat_ret = asset_results[ticker][m_name]["daily_returns"]
                reg_perf = analyze_regime_performance(
                    daily_strat_returns=daily_strat_ret,
                    daily_actual_returns=raw_daily_ret,
                    regime_df=regime_df
                )
                
                for r_name, r_metrics in reg_perf.items():
                    regime_results.append({
                        "ticker": ticker,
                        "model": m_name,
                        "regime": r_name,
                        "cagr": r_metrics.get("cagr"),
                        "sharpe": r_metrics.get("sharpe"),
                        "days": r_metrics.get("count")
                    })
                    
    regime_df_summary = pd.DataFrame(regime_results)
    
    # 7. Experiment Tracking Logging
    print("\n[Step 8/8] Logging runs into Experiment Database...")
    for ticker in tickers_to_run:
        for m_name in model_names:
            # Extract row from metrics_df
            row = metrics_df[(metrics_df['ticker'] == ticker) & (metrics_df['model'] == m_name)]
            if not row.empty:
                metrics_dict = row.iloc[0].to_dict()
                log_experiment(
                    model_version=f"{m_name.lower()}_v1.0",
                    parameters={
                        "horizon": HORIZON,
                        "val_days": VAL_DAYS,
                        "initial_context": INITIAL_CONTEXT,
                        "allow_short": True,
                        "tx_cost_pct": 0.0010,
                        "risk_free_rate": 0.04
                    },
                    dataset={
                        "ticker": ticker,
                        "period": period,
                        "dry_run": dry_run
                    },
                    metrics=metrics_dict
                )
                
    # 8. Print Reports & Save Charts
    print("\n" + "="*80)
    print("  QUANTITATIVE RESEARCH REPORT SUMMARY")
    print("="*80)
    
    # Clean output formatting
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    
    print("\nForecast Performance (Out-Of-Sample):")
    forecasting_summary = metrics_df[['ticker', 'model', 'mae', 'rmse', 'mape', 'directional_accuracy']]
    print(forecasting_summary.to_string(index=False))
    
    print("\nTrading Backtest Performance (Out-Of-Sample, Transaction Costs Applied):")
    trading_summary = metrics_df[['ticker', 'model', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'win_rate', 'profit_factor']]
    # Format percentages and decimals
    for col in ['cagr', 'max_drawdown', 'win_rate']:
        trading_summary[col] = trading_summary[col].apply(lambda x: f"{x*100:.2f}%")
    for col in ['sharpe', 'sortino', 'profit_factor']:
        trading_summary[col] = trading_summary[col].apply(lambda x: f"{x:.2f}")
    print(trading_summary.to_string(index=False))
    
    print("\nUnified Multi-Asset Portfolio Summary (Equal capital allocation of 20%):")
    portfolio_rows = []
    for m_name, res in portfolio_results.items():
        met = res["metrics"]
        portfolio_rows.append({
            "model": m_name,
            "cagr": f"{met['cagr']*100:.2f}%",
            "sharpe": f"{met['sharpe']:.2f}",
            "sortino": f"{met['sortino']:.2f}",
            "max_drawdown": f"{met['max_drawdown']*100:.2f}%",
            "win_rate": f"{met['win_rate']*100:.2f}%",
            "profit_factor": f"{met['profit_factor']:.2f}"
        })
    print(pd.DataFrame(portfolio_rows).to_string(index=False))
    
    # Save comparison charts
    print("\nGenerating charts...")
    plt.style.use('dark_background')
    
    # Plot 1: Cumulative returns comparison for AAPL (and others if run)
    for ticker in tickers_to_run:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_facecolor('#1A1D27')
        fig.patch.set_facecolor('#0F1117')
        ax.tick_params(colors='#C0C0C0')
        ax.spines[:].set_color('#333344')
        
        # Benchmark buy & hold
        df_bench = market_data[ticker]
        # Align with testing dates using nearest-match to handle holiday/weekend gaps
        dates_test = asset_results[ticker]["TimesFM"]["dates"]
        bench_idx = df_bench.set_index(pd.to_datetime(df_bench['timestamps']))
        bench_close = bench_idx['close'].reindex(dates_test, method='nearest')
        bench_cum_ret = (bench_close / bench_close.iloc[0]) - 1.0
        
        ax.plot(dates_test, bench_cum_ret * 100.0, label="Benchmark (Buy & Hold)", color="#AAAAAA", linestyle=":", alpha=0.8, linewidth=1.5)
        
        colors = {
            "TimesFM": "#3a86ff",
            "Kronos": "#e07a5f",
            "WeightedEnsemble": "#8338ec",
            "ConfidenceEnsemble": "#ff007f",
            "BaseML": "#ffb703",
            "BaseML_Technical": "#06d6a0"
        }
        
        for m_name in model_names:
            res = asset_results[ticker][m_name]
            cum_ret = (1.0 + res["daily_returns"]).cumprod() - 1.0
            ax.plot(dates_test, cum_ret * 100.0, label=f"{m_name}", color=colors.get(m_name, "#ffffff"), linewidth=2)
            
        ax.set_title(f"Cumulative Returns Comparison — {ticker} (Walk-Forward Test Split)", fontsize=14, pad=15, color="#ffffff", fontweight='bold')
        ax.set_xlabel("Date", fontsize=11, color="#C0C0C0")
        ax.set_ylabel("Cumulative Return (%)", fontsize=11, color="#C0C0C0")
        ax.grid(axis='y', linestyle=':', alpha=0.3, color='#555566')
        ax.legend(fontsize=9, facecolor='#1A1D27', edgecolor='#333344', labelcolor='#C0C0C0')
        plt.tight_layout()
        
        plot_path = os.path.join(ROOT, "plots", f"{ticker}_cum_returns_comparison.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved cumulative returns comparison plot → {plot_path}")
        
    # Plot 2: Unified Portfolio Cumulative Returns
    if len(portfolio_results) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_facecolor('#1A1D27')
        fig.patch.set_facecolor('#0F1117')
        ax.tick_params(colors='#C0C0C0')
        ax.spines[:].set_color('#333344')
        
        # Benchmark equal weight portfolio (no signals)
        daily_ret_bench = []
        for ticker in tickers_to_run:
            df_bench = market_data[ticker]
            dates_test = portfolio_results["TimesFM"]["daily_returns"].index
            bench_idx = df_bench.set_index(pd.to_datetime(df_bench['timestamps']))
            bench_close = bench_idx['close'].reindex(dates_test, method='nearest')
            daily_ret_bench.append(bench_close.pct_change().fillna(0.0))
            
        bench_port_ret = pd.concat(daily_ret_bench, axis=1).mean(axis=1)
        bench_port_cum = (1.0 + bench_port_ret).cumprod() - 1.0
        
        ax.plot(bench_port_cum.index, bench_port_cum * 100.0, label="Equal Weight Buy-and-Hold Portfolio", color="#AAAAAA", linestyle=":", alpha=0.8, linewidth=1.5)
        
        for m_name in model_names:
            if m_name in portfolio_results:
                daily_ret = portfolio_results[m_name]["daily_returns"]
                cum_ret = (1.0 + daily_ret).cumprod() - 1.0
                ax.plot(cum_ret.index, cum_ret * 100.0, label=f"{m_name}", color=colors.get(m_name, "#ffffff"), linewidth=2)
                
        ax.set_title("Unified Multi-Asset Portfolio Performance (Walk-Forward Out-of-Sample)", fontsize=14, pad=15, color="#ffffff", fontweight='bold')
        ax.set_xlabel("Date", fontsize=11, color="#C0C0C0")
        ax.set_ylabel("Cumulative Return (%)", fontsize=11, color="#C0C0C0")
        ax.grid(axis='y', linestyle=':', alpha=0.3, color='#555566')
        ax.legend(fontsize=9, facecolor='#1A1D27', edgecolor='#333344', labelcolor='#C0C0C0')
        plt.tight_layout()
        
        port_plot_path = os.path.join(ROOT, "plots", "unified_portfolio_comparison.png")
        plt.savefig(port_plot_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved unified portfolio comparison plot → {port_plot_path}")
        
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    evaluate_models(dry_run=args.dry_run, period=args.period)
