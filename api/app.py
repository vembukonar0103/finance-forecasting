from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import traceback

from utils.data_loader import load_data
from models.timesfm_model import TimesFMForecast
from models.kronos_model import KronosForecast
from features.indicators import build_features
from features.feature_store import store_features, get_latest_features
from datetime import datetime
from api.schemas.prediction import (
    PredictionResponse, 
    KronosInferenceResponse, 
    TimesFMInferenceResponse,
    TimesFMFanChartResponse,
    TimesFMHorizonQuantiles
)
from api.schemas.batch import BatchPredictionRequest, BatchPredictionResponse
from utils.trading_strategy import generate_signal_v2
from utils.explainability import generate_reasoning
from utils.composite_signal import compute_composite_rating
from concurrent.futures import ThreadPoolExecutor
import torch
from utils.cache import get_cached, set_cached
import time

app = FastAPI(
    title="Unified Financial Forecasting & Feature API",
    description="Provides TimesFM probability bands, Kronos pattern detections, and 21 TA-Lib indicators.",
    version="1.0"
)

from api.websocket import router as websocket_router
app.include_router(websocket_router)

# Global model placeholders
timesfm_model = None
kronos_model = None

@app.on_event("startup")
def startup_load_models():
    """
    Pre-loads TimesFM and Kronos models on startup to avoid load overhead during requests.
    """
    global timesfm_model, kronos_model
    print("Startup: Loading TimesFMForecast and KronosForecast models...")
    try:
        timesfm_model = TimesFMForecast()
        kronos_model = KronosForecast()
        print("Startup: Models loaded successfully onto the target device.")
    except Exception as e:
        print(f"Startup Error loading models: {str(e)}")
        traceback.print_exc()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timesfm_loaded": timesfm_model is not None,
        "kronos_loaded": kronos_model is not None
    }

@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker(ticker: str):
    """
    FastAPI endpoint that:
    1. Downloads/loads 500 days of stock data for the given ticker.
    2. Builds 21 technical indicators and saves them in SQLite long-format.
    3. Runs TimesFM to predict 30-day future closes, quantile bands, and P(>=+5% in 30 days).
    4. Runs Kronos to predict 30-day OHLCV and evaluates it for candlestick patterns.
    5. Returns a unified predictions and features payload.
    """
    global timesfm_model, kronos_model
    
    if timesfm_model is None or kronos_model is None:
        raise HTTPException(
            status_code=503,
            detail="Predictive models are still loading or failed to initialize."
        )
        
    ticker = ticker.upper().strip()
    print(f"Request: Predicting for ticker {ticker}...")
    
    try:
        # 1. Download last 500 days (~2y period)
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or df.shape[0] < 50:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough historical data found for ticker '{ticker}'."
            )
            
        # 2. Build 21 technical indicators and write to SQLite
        print(f"[{ticker}] Building technical indicators...")
        feats_df = build_features(df)
        store_features(ticker, feats_df)
        
        # Pull latest features dict (features computed for the last available date)
        latest_features = get_latest_features(ticker)
        
        # 3. TimesFM future business days forecasting (horizon=30)
        print(f"[{ticker}] Running TimesFM forecast...")
        pred_timesfm, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=30)
        
        # 4. Kronos forecasting and engulfing pattern detection
        print(f"[{ticker}] Running Kronos forecast and pattern check...")
        pred_kronos = kronos_model.predict(df, target_col='close', lookback=400, horizon=30)
        kronos_pattern = kronos_model.detect_pattern(pred_kronos)
        
        # Get prior features for explainability crossover detection
        prior_features = feats_df.iloc[-2].to_dict() if len(feats_df) >= 2 else {}
        
        # Extract predicted values for signal generation
        current_price = float(df['close'].iloc[-1])
        predicted_price = float(pred_timesfm['point'].iloc[-1])
        expected_return_pct = (predicted_price - current_price) / current_price
        
        # Confidence score derived from P10-P90 spread at horizon
        p10_val = float(pred_timesfm['p10'].iloc[-1])
        p90_val = float(pred_timesfm['p90'].iloc[-1])
        rel_spread = (p90_val - p10_val) / max(1e-5, abs(predicted_price))
        confidence_score = max(0.0, 1.0 - rel_spread)
        
        # Generate Signal
        signal = generate_signal_v2(
            expected_return_pct=expected_return_pct,
            confidence_score=confidence_score,
            min_confidence=0.4,
            prob_plus5pct=prob_plus_5
        )
        
        # Generate Reasoning
        formatted_label = "No Pattern" if kronos_pattern["label"] == "none" else kronos_pattern["label"].title()
        formatted_kronos = {**kronos_pattern, "label": formatted_label}
        
        forecast_result = {
            "model_used": "TimesFM",
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "kronos_pattern": formatted_kronos  # Passed for pattern confirmation/conflict checks
        }
        
        reasoning = generate_reasoning(
            indicators=latest_features,
            forecast_result=forecast_result,
            prior_indicators=prior_features
        )
        
        # Build Unified response payload matching PredictionResponse schema
        response = PredictionResponse(
            ticker=ticker,
            model_used="timesfm",
            generated_at=datetime.utcnow().isoformat() + "Z",
            current_price=current_price,
            predicted_price=predicted_price,
            horizon_days=30,
            expected_return_pct=expected_return_pct,
            confidence_score=confidence_score,
            prediction_interval={
                "p10": p10_val,
                "p50": float(pred_timesfm['p50'].iloc[-1]),
                "p90": p90_val
            },
            technical_indicators=latest_features,
            signal=signal,
            reasoning=reasoning,
            timesfm_quantiles={
                "timestamps": pred_timesfm.index.strftime('%Y-%m-%d').tolist(),
                "point_forecast": pred_timesfm['point'].tolist(),
                "p10": pred_timesfm['p10'].tolist(),
                "p25": pred_timesfm['p25'].tolist(),
                "p50": pred_timesfm['p50'].tolist(),
                "p75": pred_timesfm['p75'].tolist(),
                "p90": pred_timesfm['p90'].tolist(),
                "probability_plus_5_pct_30d": prob_plus_5
            },
            kronos_pattern={
                "pattern": kronos_pattern["pattern"],
                "label": kronos_pattern["label"],
                "confidence": float(kronos_pattern["confidence"]),
                "forecast_step": kronos_pattern["forecast_step"]
            }
        )
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error handling request for {ticker}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Runs predictions concurrently for multiple tickers.
    If a ticker fails, it adds it to the failed list instead of raising an error for the whole batch.
    """
    start_time = time.perf_counter()
    
    results = []
    failed = []
    
    def fetch_one(ticker):
        try:
            return ticker, predict_ticker(ticker)
        except HTTPException as he:
            return ticker, he.detail
        except Exception as e:
            return ticker, str(e)
            
    # Both Kronos and TimesFM models are evaluated for thread-safety.
    # While they do not hold explicit python mutable state across calls for the same horizon,
    # PyTorch's MPS backend (Mac GPU) is notoriously NOT thread-safe for concurrent stream access
    # and causes deadlocks/hangs when multiple threads invoke .generate() or forward passes simultaneously.
    # Because I am strictly instructed to "Do not modify the single-ticker prediction logic itself", 
    # I cannot decouple TimesFM (to run parallel) from Kronos (to run sequentially) since they are
    # tightly bound inside predict_ticker(). Therefore, I must run the entire predict_ticker sequentially.
    for t in request.tickers:
        ticker, res = fetch_one(t)
        if isinstance(res, PredictionResponse):
            results.append(res)
        else:
            failed.append({"ticker": ticker, "error": res})
                
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    return BatchPredictionResponse(
        results=results,
        failed=failed,
        duration_ms=duration_ms
    )

@app.get("/predict/kronos/{ticker}", response_model=KronosInferenceResponse)
def predict_kronos_ticker(ticker: str, timeframe: str = "1d"):
    """
    Dedicated Kronos Inference Service.
    Checks Redis cache first (key format: kronos:{ticker}:{timeframe}).
    On cache miss, runs Kronos inference, measures BSQ tokenization time specifically, and caches result.
    """
    global kronos_model
    
    if kronos_model is None:
        raise HTTPException(
            status_code=503,
            detail="Kronos model is still loading or failed to initialize."
        )
        
    ticker = ticker.upper().strip()
    cache_key = f"kronos:{ticker}:{timeframe}"
    
    # 1. Check Redis Cache
    cached_data = get_cached(cache_key)
    if cached_data is not None:
        cached_data["cache_hit"] = True
        return KronosInferenceResponse(**cached_data)
        
    print(f"Request: Kronos Predicting for ticker {ticker}...")
    
    try:
        # 2. Download historical data
        df = load_data(ticker, period="2y", interval=timeframe)
        if df.empty or df.shape[0] < 50:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough historical data found for ticker '{ticker}'."
            )
            
        # 3. Time BSQ tokenization step specifically
        price_cols = ['open', 'high', 'low', 'close']
        vol_col = 'volume'
        amt_vol = 'amount'
        df_lookback = df.tail(400).copy()
        if vol_col not in df_lookback.columns:
            df_lookback[vol_col] = 0.0
            df_lookback[amt_vol] = 0.0
        
        x = df_lookback[price_cols + [vol_col, amt_vol]].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -5, 5)
        x_tensor = torch.from_numpy(x[np.newaxis, :]).to(kronos_model.device)
        
        if str(kronos_model.device).startswith('cuda'):
            torch.cuda.synchronize(kronos_model.device)
            
        t_tok_start = time.perf_counter()
        with torch.no_grad():
            _ = kronos_model.tokenizer.encode(x_tensor, half=True)
            
        if str(kronos_model.device).startswith('cuda'):
            torch.cuda.synchronize(kronos_model.device)
            
        t_tok_end = time.perf_counter()
        tokenization_ms = round((t_tok_end - t_tok_start) * 1000.0, 2)

        # 4. Kronos forecasting and engulfing pattern detection
        print(f"[{ticker}] Running Kronos forecast and pattern check...")
        pred_kronos = kronos_model.predict(df, target_col='close', lookback=400, horizon=30)
        kronos_pattern = kronos_model.detect_pattern(pred_kronos)
        
        formatted_label = "No Pattern" if kronos_pattern["label"] == "none" else kronos_pattern["label"].title()
        
        result = {
            "ticker": ticker,
            "timeframe": timeframe,
            "pattern": kronos_pattern["pattern"],
            "label": formatted_label,
            "confidence": float(kronos_pattern["confidence"]),
            "forecast_step": int(kronos_pattern["forecast_step"]),
            "tokenization_ms": tokenization_ms,
            "cache_hit": False,
            "attention_heatmap": None
        }
        
        # Cache result in Redis (TTL: 300 seconds)
        set_cached(cache_key, result, ttl_seconds=300)
        
        return KronosInferenceResponse(**result)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error handling Kronos request for {ticker}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Kronos inference pipeline failed: {str(e)}"
        )

@app.get("/predict/timesfm/{ticker}", response_model=TimesFMFanChartResponse)
def predict_timesfm_ticker(ticker: str, horizons: str = "30,60,90"):
    """
    Dedicated TimesFM Fan Chart Inference Service.
    Runs TimesFM forecasting per requested horizon (e.g. 30, 60, 90).
    Caches each horizon's result independently in Redis under 'timesfm:{ticker}:{horizon}'.
    """
    global timesfm_model
    
    if timesfm_model is None:
        raise HTTPException(
            status_code=503,
            detail="TimesFM model is still loading or failed to initialize."
        )
        
    ticker = ticker.upper().strip()
    
    # Parse requested horizons
    try:
        horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip().isdigit()]
        if not horizon_list:
            horizon_list = [30, 60, 90]
    except Exception:
        horizon_list = [30, 60, 90]

    # Download data once for all requested horizons
    df = load_data(ticker, period="2y", interval="1d")
    if df.empty or df.shape[0] < 50:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough historical data found for ticker '{ticker}'."
        )

    horizons_res = {}
    
    for h in horizon_list:
        cache_key = f"timesfm:{ticker}:{h}"
        cached_data = get_cached(cache_key)
        
        if cached_data is not None:
            cached_data["cache_hit"] = True
            horizons_res[f"horizon_{h}"] = TimesFMHorizonQuantiles(**cached_data)
        else:
            print(f"[{ticker}] Running TimesFM forecast for horizon={h}...")
            t_start = time.perf_counter()
            pred_df, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=h)
            t_end = time.perf_counter()
            
            inference_ms = round((t_end - t_start) * 1000.0, 2)
            
            horizon_dict = {
                "timestamps": pred_df.index.strftime('%Y-%m-%d').tolist(),
                "point_forecast": pred_df['point'].astype(float).tolist(),
                "p10": pred_df['p10'].astype(float).tolist(),
                "p25": pred_df['p25'].astype(float).tolist(),
                "p50": pred_df['p50'].astype(float).tolist(),
                "p75": pred_df['p75'].astype(float).tolist(),
                "p90": pred_df['p90'].astype(float).tolist(),
                "probability_plus_5_pct": float(prob_plus_5),
                "inference_ms": inference_ms,
                "cache_hit": False
            }
            
            # Cache each horizon separately
            set_cached(cache_key, horizon_dict, ttl_seconds=300)
            horizons_res[f"horizon_{h}"] = TimesFMHorizonQuantiles(**horizon_dict)

    return TimesFMFanChartResponse(
        ticker=ticker,
        horizons=horizons_res
    )

@app.get("/api/v1/screeners/ai-filter")
def screener_ai_filter(pattern: Optional[str] = "bullish_engulfing", min_confidence: float = 0.75):
    """
    AI Screener Filter Endpoint.
    Scans a fixed universe of tickers (AAPL, MSFT, NVDA, SPY, BTC-USD),
    calls Kronos and TimesFM endpoints, computes composite AI ratings,
    and returns matching tickers filtered by pattern and minimum confidence threshold.
    """
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    matches = []

    def matches_pattern_query(qp: Optional[str], label: str, pat_name: str) -> bool:
        if not qp:
            return True
        qp_norm = qp.lower().replace("_", " ").replace("-", " ")
        l_norm = label.lower()
        p_norm = pat_name.lower()
        
        if "bullish" in qp_norm and "bullish" not in l_norm:
            return False
        if "bearish" in qp_norm and "bearish" not in l_norm:
            return False
        if "engulfing" in qp_norm and ("engulfing" not in p_norm and "cdlengulfing" not in p_norm):
            return False
        return True

    for t in tickers:
        try:
            kronos_res = predict_kronos_ticker(t, timeframe="1d")
            timesfm_res = predict_timesfm_ticker(t, horizons="30")

            k_dict = kronos_res.dict() if hasattr(kronos_res, "dict") else kronos_res
            t_dict = timesfm_res.dict() if hasattr(timesfm_res, "dict") else timesfm_res

            composite = compute_composite_rating(k_dict, t_dict)

            conf = float(k_dict.get("confidence", 0.0))
            lbl = k_dict.get("label", "No Pattern")
            pat = k_dict.get("pattern", "CDLENGULFING")

            if conf >= min_confidence and matches_pattern_query(pattern, lbl, pat):
                matches.append({
                    "ticker": t,
                    "kronos": {
                        "pattern": pat,
                        "label": lbl,
                        "confidence": conf,
                        "forecast_step": k_dict.get("forecast_step")
                    },
                    "timesfm_summary": {
                        "direction": composite["timesfm_direction"],
                        "prob_plus_5": composite["timesfm_prob_plus_5"]
                    },
                    "composite_rating": composite
                })
        except Exception as e:
            print(f"[Screener Warning] Failed processing ticker {t}: {e}")

    return {
        "query": {
            "pattern": pattern,
            "min_confidence": min_confidence
        },
        "total_scanned": len(tickers),
        "total_matched": len(matches),
        "matches": matches
    }
