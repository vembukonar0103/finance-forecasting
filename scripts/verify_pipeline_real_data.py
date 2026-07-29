import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_data
from models.timesfm_model import TimesFMForecast
from utils.trading_strategy import generate_signal_v2
from utils.experiment_tracker import log_component_run

def run_real_verification():
    print("======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 1: SIGNAL ENGINE")
    print("======================================================================")
    
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    
    # 1. Initialize forecasting model
    print("Initializing TimesFM model...")
    timesfm_model = TimesFMForecast()
    
    results = []
    
    for ticker in tickers:
        print(f"Processing real forecast for {ticker}...")
        
        # Load last 500 days of daily data
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or len(df) < 50:
            print(f"Warning: Not enough data for {ticker}")
            continue
            
        # Run TimesFM prediction
        pred_timesfm, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=30)
        
        # Extract inputs for generate_signal_v2
        latest_close = df['close'].iloc[-1]
        point_forecast = pred_timesfm['point'].iloc[-1]
        
        # Expected return over 30 days
        expected_return_pct = (point_forecast - latest_close) / latest_close
        
        # Confidence score
        p10 = pred_timesfm['p10'].iloc[-1]
        p90 = pred_timesfm['p90'].iloc[-1]
        point = pred_timesfm['point'].iloc[-1]
        rel_spread = (p90 - p10) / max(1e-5, np.abs(point))
        confidence_score = max(0.0, 1.0 - rel_spread)
        
        # Generate signal using the symmetric v2 engine
        signal = generate_signal_v2(
            expected_return_pct=expected_return_pct,
            confidence_score=confidence_score,
            min_confidence=0.4,
            prob_plus5pct=prob_plus_5
        )
        
        results.append({
            "ticker": ticker,
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "prob_plus_5": prob_plus_5,
            "signal": signal
        })
        
    print("\n" + "=" * 80)
    print(f"{'Ticker':<10} | {'Expected Return':<16} | {'Confidence Score':<18} | {'P(>=+5%)':<10} | {'Signal Result':<15}")
    print("-" * 80)
    for res in results:
        prob_str = f"{res['prob_plus_5']:.2f}" if res['prob_plus_5'] is not None else "None"
        print(f"{res['ticker']:<10} | {res['expected_return_pct']:<16.2%} | {res['confidence_score']:<18.4f} | {prob_str:<10} | {res['signal']:<15}")
    print("=" * 80)
    
    # Check signal diversity
    signals_produced = [res["signal"] for res in results]
    unique_signals = set(signals_produced)
    
    print("\nSignal Engine Sanity Check:")
    print(f"  Unique signals generated: {unique_signals}")
    if len(unique_signals) >= 2:
        print("  PASS — At least 2 different signal tiers were successfully generated across the 5 assets.")
    else:
        print("  SUSPICIOUS — Less than 2 unique signal tiers were generated. Thresholds or confidence parameters may be miscalibrated.")
        
    log_component_run("verify_signal_engine", {"tickers": tickers}, {"unique_signals": list(unique_signals)})

def run_real_verification_step2():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 2: RISK ENGINE")
    print("======================================================================")
    
    from features.indicators import build_features
    from utils.risk_engine import compute_stop_take, compute_volatility_score, classify_risk_level, suggested_position_size
    
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    
    capital = 10000.0
    risk_per_trade_pct = 1.0
    print(f"Assumptions: Capital = ${capital:,.2f}, Risk per Trade = {risk_per_trade_pct}%")
    print("-" * 110)
    
    results = []
    for ticker in tickers:
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or len(df) < 250:
            print(f"Warning: Not enough data for {ticker}")
            continue
            
        # Get real ATR
        feats = build_features(df)
        latest_atr = feats['atr'].iloc[-1]
        
        # Real returns series (last 250 days)
        last_250 = df.iloc[-250:].copy()
        returns_series = last_250['close'].pct_change().dropna()
        latest_price = last_250['close'].iloc[-1]
        
        # Compute Risk Metrics
        stop_loss, take_profit = compute_stop_take(
            current_price=latest_price,
            atr=latest_atr,
            direction="long"
        )
        
        vol_score = compute_volatility_score(returns_series, window=20)
        risk_class = classify_risk_level(vol_score)
        
        stop_dist_pct = (latest_price - stop_loss) / latest_price * 100.0
        pos_size = suggested_position_size(
            capital=capital,
            stop_loss_distance_pct=stop_dist_pct,
            risk_per_trade_pct=risk_per_trade_pct
        )
        
        # Sanity check: Stop loss on correct side
        sl_correct = stop_loss < latest_price
        
        results.append({
            "ticker": ticker,
            "price": latest_price,
            "atr": latest_atr,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sl_correct": sl_correct,
            "vol_score": vol_score,
            "risk_class": risk_class,
            "pos_size": pos_size
        })
        
    print(f"{'Ticker':<10} | {'Price':<10} | {'Stop Loss':<12} | {'Take Profit':<12} | {'SL Valid?':<10} | {'Vol Score':<10} | {'Risk':<8} | {'Pos Size ($)':<12}")
    print("-" * 110)
    for res in results:
        print(f"{res['ticker']:<10} | {res['price']:<10.2f} | {res['stop_loss']:<12.2f} | {res['take_profit']:<12.2f} | {str(res['sl_correct']):<10} | {res['vol_score']:<10.4f} | {res['risk_class']:<8} | ${res['pos_size']:<12.2f}")
    
    print("=" * 110)
    print("Risk Engine Sanity Check:")
    spy_res = next((r for r in results if r["ticker"] == "SPY"), None)
    btc_res = next((r for r in results if r["ticker"] == "BTC-USD"), None)
    
    if spy_res and btc_res:
        print(f"  SPY Vol Score: {spy_res['vol_score']:.4f} ({spy_res['risk_class']})")
        print(f"  BTC Vol Score: {btc_res['vol_score']:.4f} ({btc_res['risk_class']})")
        if spy_res['vol_score'] == btc_res['vol_score'] and spy_res['risk_class'] == btc_res['risk_class']:
            print("  FLAG: SPY and BTC-USD have identical volatility scores/classes. Percentile rank may not be working as intended to differentiate asset-specific regimes.")
        else:
            print("  PASS: BTC-USD and SPY volatility scores/classes differ based on their own respective historical distributions.")
            
    log_component_run("verify_risk_engine", {"capital": capital}, {"assets_checked": len(results)})


def run_real_verification_step3():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 3: PORTFOLIO OPTIMIZER")
    print("======================================================================")
    
    from features.indicators import build_features
    from utils.risk_engine import compute_volatility_score
    from utils.portfolio_optimizer import rank_opportunities, suggest_allocation
    
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    
    print("Initializing TimesFM model...")
    timesfm_model = TimesFMForecast()
    
    predictions = []
    
    for ticker in tickers:
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or len(df) < 250:
            continue
            
        # TimesFM expected return
        pred_timesfm, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=30)
        latest_close = df['close'].iloc[-1]
        point_forecast = pred_timesfm['point'].iloc[-1]
        expected_return_pct = (point_forecast - latest_close) / latest_close
        
        p10 = pred_timesfm['p10'].iloc[-1]
        p90 = pred_timesfm['p90'].iloc[-1]
        point = pred_timesfm['point'].iloc[-1]
        rel_spread = (p90 - p10) / max(1e-5, np.abs(point))
        confidence_score = max(0.0, 1.0 - rel_spread)
        
        # Risk score
        last_250 = df.iloc[-250:].copy()
        returns_series = last_250['close'].pct_change().dropna()
        risk_score = compute_volatility_score(returns_series, window=20)
        
        predictions.append({
            "ticker": ticker,
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "risk_score": risk_score
        })
        
    # Rank opportunities
    ranked = rank_opportunities(predictions)
    
    # Suggest allocation
    allocations = suggest_allocation(ranked)
    
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Exp Return':<12} | {'Risk Score':<12} | {'Opp Score':<12} | {'Allocation %':<12}")
    print("-" * 75)
    
    for i, opp in enumerate(ranked):
        ticker = opp["ticker"]
        exp_ret = opp["expected_return_pct"]
        risk = opp["risk_score"]
        opp_score = opp["opportunity_score"]
        alloc = allocations.get(ticker, 0.0)
        
        print(f"{i+1:<5} | {ticker:<10} | {exp_ret:<12.2%} | {risk:<12.4f} | {opp_score:<12.4f} | {alloc:<10.2f}%")
        
    alloc_sum = sum(allocations.values())
    print("=" * 75)
    print(f"Total Allocation: {alloc_sum:.2f}%")
    
    if abs(alloc_sum - 100.0) < 1e-5:
        print("  PASS: Allocations sum exactly to 100%.")
    else:
        print("  FLAG: Allocations do not sum to 100%.")

    log_component_run("verify_portfolio_optimizer", {"tickers": tickers}, {"top_allocation": ranked[0]["ticker"]})


def run_real_verification_step4():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 4: EXPLAINABLE AI")
    print("======================================================================")
    
    from features.indicators import build_features
    from utils.explainability import generate_reasoning
    
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    
    from models.kronos_model import KronosForecast
    
    print("Initializing TimesFM model...")
    timesfm_model = TimesFMForecast()
    print("Initializing Kronos model...")
    kronos_model = KronosForecast()
    
    for ticker in tickers:
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or len(df) < 50:
            continue
            
        feats = build_features(df)
        
        # We need the last row and the prior row
        current_feats = feats.iloc[-1].to_dict()
        prior_feats = feats.iloc[-2].to_dict()
        
        # TimesFM forecast
        pred_timesfm, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=30)
        latest_close = df['close'].iloc[-1]
        point_forecast = pred_timesfm['point'].iloc[-1]
        expected_return_pct = (point_forecast - latest_close) / latest_close
        
        p10 = pred_timesfm['p10'].iloc[-1]
        p90 = pred_timesfm['p90'].iloc[-1]
        point = pred_timesfm['point'].iloc[-1]
        rel_spread = (p90 - p10) / max(1e-5, np.abs(point))
        confidence_score = max(0.0, 1.0 - rel_spread)
        
        # Kronos forecast and pattern detection
        pred_kronos = kronos_model.predict(df, target_col='close', lookback=400, horizon=30)
        kronos_pattern = kronos_model.detect_pattern(pred_kronos)
        
        formatted_label = "No Pattern" if kronos_pattern["label"] == "none" else kronos_pattern["label"].title()
        formatted_kronos = {**kronos_pattern, "label": formatted_label}
        
        forecast_result = {
            "model_used": "TimesFM",
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "kronos_pattern": formatted_kronos
        }
        
        reasoning = generate_reasoning(
            indicators=current_feats,
            forecast_result=forecast_result,
            prior_indicators=prior_feats
        )
        
        print(f"\n--- {ticker} ---")
        print(f"Kronos Pattern: {formatted_kronos['label']} (Conf: {formatted_kronos['confidence']:.2f})")
        print(f"Raw RSI: {current_feats.get('rsi', 0):.2f}")
        print(f"Raw MACD: Curr={current_feats.get('macd', 0):.2f}, Prior={prior_feats.get('macd', 0):.2f}")
        print(f"Raw MACD Sig: Curr={current_feats.get('macd_sig', 0):.2f}, Prior={prior_feats.get('macd_sig', 0):.2f}")
        print("Reasoning Output:")
        for r in reasoning:
            print(f"  - {r}")
            
    log_component_run("verify_explainability", {"tickers": tickers}, {"success": True})


def run_real_verification_step5():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 5: REGIME DETECTION")
    print("======================================================================")
    
    from utils.regime_analysis import classify_current_regime
    
    tickers = ["AAPL", "BTC-USD"]
    
    print(f"{'Ticker':<10} | {'Trend':<12} | {'Volatility':<12}")
    print("-" * 40)
    for ticker in tickers:
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or len(df) < 200:
            continue
            
        regime = classify_current_regime(df)
        print(f"{ticker:<10} | {regime['trend']:<12} | {regime['volatility']:<12}")
        
    log_component_run("verify_regime_detection", {"tickers": tickers}, {"success": True})


def run_real_verification_step6():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 6: PORTFOLIO BACKTEST")
    print("======================================================================")
    
    from utils.backtester import run_portfolio_backtest
    
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    
    # Real allocations from Step 3
    opt_alloc = {"BTC-USD": 0.3754, "SPY": 0.2486, "MSFT": 0.1039, "AAPL": 0.0825, "NVDA": 0.1896}
    eq_alloc = {t: 0.20 for t in tickers}
    
    ticker_signals = {}
    for t in tickers:
        df = load_data(t, period="5y")
        df.index = pd.to_datetime(df['timestamps'])
        ret = df['close'].pct_change().fillna(0.0)
        # Using raw benchmark returns as a surrogate for "real data" strategy returns 
        # to test the pure effect of the allocation weights.
        ticker_signals[t] = {
            "daily_returns": ret,
            "trade_returns": [(1.0 + ret).prod() - 1.0]
        }
        
    res_eq = run_portfolio_backtest(ticker_signals, eq_alloc)
    res_opt = run_portfolio_backtest(ticker_signals, opt_alloc)
    
    metrics_to_print = ['cagr', 'sharpe', 'max_drawdown', 'win_rate', 'profit_factor', 'sample_size', 'low_confidence_warning']
    
    print(f"{'Metric':<25} | {'Equal Weight (20%)':<20} | {'Risk-Weighted (Step 3)':<20}")
    print("-" * 75)
    for m in metrics_to_print:
        val_eq = res_eq['metrics'].get(m, 'N/A')
        val_opt = res_opt['metrics'].get(m, 'N/A')
        
        if isinstance(val_eq, float) and m in ['cagr', 'max_drawdown', 'win_rate']:
            str_eq = f"{val_eq*100:.2f}%" if pd.notna(val_eq) else "N/A"
            str_opt = f"{val_opt*100:.2f}%" if pd.notna(val_opt) else "N/A"
        elif isinstance(val_eq, float):
            str_eq = f"{val_eq:.3f}" if pd.notna(val_eq) else "N/A"
            str_opt = f"{val_opt:.3f}" if pd.notna(val_opt) else "N/A"
        else:
            str_eq = str(val_eq)
            str_opt = str(val_opt)
            
        print(f"{m:<25} | {str_eq:<20} | {str_opt:<20}")


def run_real_verification_step7():
    print("\n======================================================================")
    print("  PHASE 8: VERIFICATION AGAINST REAL DATA — STEP 7: EXPERIMENT TRACKING")
    print("======================================================================")
    import sqlite3
    
    conn = sqlite3.connect("data/experiments.db")
    df = pd.read_sql("SELECT * FROM component_runs WHERE component_name LIKE 'verify_%'", conn)
    conn.close()
    
    print(f"Total verification rows logged: {len(df)}")
    if len(df) > 0:
        print("\nSample Row (Latest):")
        sample = df.iloc[-1]
        print(f"  ID: {sample['id']}")
        print(f"  Timestamp: {sample['timestamp']}")
        print(f"  Component: {sample['component_name']}")
        print(f"  Parameters: {sample['parameters']}")
        print(f"  Summary: {sample['output_summary']}")
    print("======================================================================")

if __name__ == "__main__":
    # run_real_verification()
    # run_real_verification_step2()
    # run_real_verification_step3()
    run_real_verification_step4()
    # run_real_verification_step5()
    # run_real_verification_step6()
    # run_real_verification_step7()
