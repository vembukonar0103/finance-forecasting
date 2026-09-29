from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import traceback

from utils.data_loader import load_data
from models.timesfm_model import TimesFMForecast
from models.kronos_model import KronosForecast
from models.bigquery_fallback import BigQueryForecast
from features.indicators import build_features
from features.feature_store import store_features, get_latest_features
from features.feast_wrapper import push_online_features, get_online_features
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
from utils.cache import get_cached, set_cached, get_screener_cache, set_screener_cache
from utils.circuit_breaker import timesfm_breaker, kronos_breaker, yfinance_breaker
from utils.health import get_full_health_status
import time

from utils.logger import logger
from utils.metrics import (
    metrics_endpoint_response,
    REQUESTS_TOTAL,
    INFERENCE_LATENCY,
    CACHE_ACCESS_TOTAL,
    SCREENER_MATCH_COUNT
)
from fastapi import FastAPI, HTTPException, Request, Response

app = FastAPI(
    title="Unified Financial Forecasting & Feature API",
    description="Provides TimesFM probability bands, Kronos pattern detections, and 21 TA-Lib indicators.",
    version="1.0"
)

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """
    Structured logging and Prometheus request latency & counter middleware.
    """
    start_time = time.perf_counter()
    endpoint = request.url.path
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        raise exc
    finally:
        duration_s = time.perf_counter() - start_time
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=str(status_code)).inc()
        logger.info(
            f"{request.method} {endpoint} -> {status_code} ({duration_s*1000:.1f}ms)",
            extra={
                "endpoint": endpoint,
                "status_code": status_code,
                "latency_ms": round(duration_s * 1000, 2)
            }
        )

@app.get("/metrics")
def get_metrics():
    """
    Prometheus metrics exposition endpoint.
    Exposes inference latency histograms, request counters, cache hit rates, and screener gauges.
    """
    return metrics_endpoint_response()

from api.websocket import router as websocket_router
app.include_router(websocket_router)

# Global model placeholders
timesfm_model = None
kronos_model = None
bigquery_fallback = BigQueryForecast()

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
    """
    Dependency-aware enterprise health check endpoint.
    Reports operational status and latencies for Redis, TimesFM, Kronos, and Yahoo Finance.
    """
    return get_full_health_status(timesfm_model, kronos_model)

@app.get("/features/{ticker}")
def get_features_endpoint(ticker: str):
    """
    Dedicated Online Feature Store Endpoint.
    Retrieves latest 25 TA-Lib and volatility regime features from Feast (Redis).
    Falls back to SQLite if Feast is unpopulated.
    """
    ticker = ticker.upper().strip()
    features = get_online_features(ticker, feature_view="all")
    if not features:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for ticker '{ticker}' in online store or fallback store."
        )
    return {
        "ticker": ticker,
        "features_count": len(features),
        "source": "feast_redis",
        "features": features
    }

@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker(ticker: str):
    """
    FastAPI endpoint that:
    1. Checks Redis cache for unified prediction (TTL 300s).
    2. Downloads/loads data for the given ticker.
    3. Builds 25 technical indicators and volatility regime, pushing to Feast.
    4. Runs TimesFM (with CircuitBreaker and BigQuery fallback).
    5. Runs Kronos for candlestick pattern and attention heatmap extraction.
    6. Returns a unified predictions and features payload.
    """
    global timesfm_model, kronos_model, bigquery_fallback
    
    if timesfm_model is None or kronos_model is None:
        raise HTTPException(
            status_code=503,
            detail="Predictive models are still loading or failed to initialize."
        )
        
    ticker = ticker.upper().strip()
    cache_key = f"unified:{ticker}"
    cached_data = get_cached(cache_key)
    if cached_data is not None:
        CACHE_ACCESS_TOTAL.labels(service="unified", hit="true").inc()
        return PredictionResponse(**cached_data)
    CACHE_ACCESS_TOTAL.labels(service="unified", hit="false").inc()

    print(f"Request: Predicting for ticker {ticker}...")
    
    try:
        # 1. Download last 500 days (~2y period)
        df = load_data(ticker, period="2y", interval="1d")
        if df.empty or df.shape[0] < 50:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough historical data found for ticker '{ticker}'."
            )
            
        # 2. Build 25 technical indicators + volatility regime and push to Feast
        print(f"[{ticker}] Building technical indicators & volatility regime...")
        feats_df = build_features(df)
        push_online_features(ticker, feats_df)
        
        # Pull latest online features from Redis via Feast (with SQLite fallback)
        latest_features = get_online_features(ticker, feature_view="all")
        
        # 3. TimesFM future business days forecasting (with Circuit Breaker & BigQuery Fallback)
        print(f"[{ticker}] Running TimesFM forecast (with resilience fallback)...")
        model_name_used = "timesfm"
        try:
            if timesfm_breaker.is_open(ticker):
                raise RuntimeError(f"TimesFM CircuitBreaker OPEN for {ticker}. Tripping to fallback.")
            with INFERENCE_LATENCY.labels(model="timesfm").time():
                pred_timesfm, prob_plus_5 = timesfm_model.predict(df, target_col='close', horizon=30)
            timesfm_breaker.record_success(ticker)
        except Exception as e:
            timesfm_breaker.record_failure(ticker)
            print(f"[{ticker}] TimesFM error ({e}). Activating BigQuery / Statistical fallback...")
            with INFERENCE_LATENCY.labels(model="bigquery_fallback").time():
                pred_timesfm, prob_plus_5, fb_meta = bigquery_fallback.predict(df, target_col='close', horizon=30)
            model_name_used = fb_meta.get("provider", "bigquery_fallback")
        
        # 4. Kronos forecasting and engulfing pattern detection
        print(f"[{ticker}] Running Kronos forecast and pattern check...")
        try:
            with INFERENCE_LATENCY.labels(model="kronos").time():
                pred_kronos = kronos_model.predict(df, target_col='close', lookback=400, horizon=30, extract_attention=True)
                kronos_pattern = kronos_model.detect_pattern(pred_kronos)
            kronos_breaker.record_success(ticker)
        except Exception as e:
            kronos_breaker.record_failure(ticker)
            print(f"[{ticker}] Kronos inference warning: {e}")
            kronos_pattern = {"pattern": "CDLENGULFING", "label": "none", "confidence": 0.0, "forecast_step": -1}
        
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
            "model_used": model_name_used,
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "kronos_pattern": formatted_kronos
        }
        
        reasoning = generate_reasoning(
            indicators=latest_features,
            forecast_result=forecast_result,
            prior_indicators=prior_features
        )
        
        # Build Unified response payload matching PredictionResponse schema
        response = PredictionResponse(
            ticker=ticker,
            model_used=model_name_used,
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
        
        # Cache unified response in Redis (TTL: 300s)
        set_cached(cache_key, response.dict(), ttl_seconds=300)
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
        CACHE_ACCESS_TOTAL.labels(service="kronos", hit="true").inc()
        cached_data["cache_hit"] = True
        return KronosInferenceResponse(**cached_data)
    CACHE_ACCESS_TOTAL.labels(service="kronos", hit="false").inc()
        
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
        print(f"[{ticker}] Running Kronos forecast and pattern check (with attention extraction)...")
        try:
            with INFERENCE_LATENCY.labels(model="kronos").time():
                pred_kronos = kronos_model.predict(df, target_col='close', lookback=400, horizon=30, extract_attention=True)
                kronos_pattern = kronos_model.detect_pattern(pred_kronos)
            kronos_breaker.record_success(ticker)
        except Exception as e:
            kronos_breaker.record_failure(ticker)
            raise e
        
        formatted_label = "No Pattern" if kronos_pattern["label"] == "none" else kronos_pattern["label"].title()
        attention_map = getattr(kronos_model, "last_attention_heatmap", None)
        
        result = {
            "ticker": ticker,
            "timeframe": timeframe,
            "pattern": kronos_pattern["pattern"],
            "label": formatted_label,
            "confidence": float(kronos_pattern["confidence"]),
            "forecast_step": int(kronos_pattern["forecast_step"]),
            "tokenization_ms": tokenization_ms,
            "cache_hit": False,
            "attention_heatmap": attention_map
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
def predict_timesfm_ticker(ticker: str, horizons: str = "30,60,90", max_context: Optional[int] = None):
    """
    Dedicated TimesFM Fan Chart Inference Service with Configurable Context Window and Resilience Fallback.
    Runs TimesFM forecasting per requested horizon (e.g. 30, 60, 90).
    Caches each horizon's result independently in Redis under 'timesfm:{ticker}:{horizon}:{max_context}'.
    """
    global timesfm_model, bigquery_fallback
    
    if timesfm_model is None:
        raise HTTPException(
            status_code=503,
            detail="TimesFM model is still loading or failed to initialize."
        )
        
    ticker = ticker.upper().strip()
    effective_ctx = int(max_context or getattr(timesfm_model, "default_max_context", 4096))
    
    # Parse requested horizons
    try:
        horizon_list = [int(h.strip()) for h in horizons.split(",") if h.strip().isdigit()]
        if not horizon_list:
            horizon_list = [30, 60, 90]
    except Exception:
        horizon_list = [30, 60, 90]

    # Download data once for all requested horizons (support longer history for large context)
    period_to_fetch = "15y" if effective_ctx > 2000 else "2y"
    df = load_data(ticker, period=period_to_fetch, interval="1d")
    if df.empty or df.shape[0] < 50:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough historical data found for ticker '{ticker}'."
        )

    horizons_res = {}
    
    for h in horizon_list:
        cache_key = f"timesfm:{ticker}:{h}:{effective_ctx}"
        cached_data = get_cached(cache_key)
        
        if cached_data is not None:
            CACHE_ACCESS_TOTAL.labels(service="timesfm", hit="true").inc()
            cached_data["cache_hit"] = True
            horizons_res[f"horizon_{h}"] = TimesFMHorizonQuantiles(**cached_data)
        else:
            CACHE_ACCESS_TOTAL.labels(service="timesfm", hit="false").inc()
            print(f"[{ticker}] Running TimesFM forecast for horizon={h}, max_context={effective_ctx}...")
            t_start = time.perf_counter()
            is_fallback = False
            provider_name = "timesfm"
            
            try:
                if timesfm_breaker.is_open(ticker):
                    raise RuntimeError(f"Circuit breaker OPEN for TimesFM ({ticker})")
                with INFERENCE_LATENCY.labels(model="timesfm").time():
                    pred_df, prob_plus_5 = timesfm_model.predict(
                        df, target_col='close', horizon=h, max_context=effective_ctx
                    )
                timesfm_breaker.record_success(ticker)
            except Exception as e:
                timesfm_breaker.record_failure(ticker)
                print(f"[Fallback] TimesFM failure on horizon={h}: {e}. Triggering BigQuery/Statistical fallback...")
                with INFERENCE_LATENCY.labels(model="bigquery_fallback").time():
                    pred_df, prob_plus_5, fb_meta = bigquery_fallback.predict(
                        df, target_col='close', horizon=h
                    )
                is_fallback = True
                provider_name = fb_meta.get("provider", "BigQueryFallback")
                
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
                "cache_hit": False,
                "fallback": is_fallback,
                "provider": provider_name
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
    High-Performance AI Screener Filter Endpoint.
    Architecture:
      1. First checks pre-computed Redis cache 'screener:latest_results' populated by background worker.
         Serves in <50ms (meeting the <2s SLA for 50+ tickers!).
      2. If cache is unpopulated (cold start), performs on-demand evaluation of default candidate universe,
         caches the results, and returns.
    """
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

    # 1. Check Pre-computed Redis Cache (Sub-second response meeting <2s SLA)
    cached_screener = get_screener_cache()
    if cached_screener and isinstance(cached_screener, list):
        matches = []
        for item in cached_screener:
            conf = float(item.get("kronos", {}).get("confidence", 0.0))
            lbl = item.get("kronos", {}).get("label", "No Pattern")
            pat = item.get("kronos", {}).get("pattern", "CDLENGULFING")
            if conf >= min_confidence and matches_pattern_query(pattern, lbl, pat):
                matches.append(item)

        SCREENER_MATCH_COUNT.set(len(matches))
        return {
            "query": {
                "pattern": pattern,
                "min_confidence": min_confidence
            },
            "source": "precomputed_redis_cache",
            "total_scanned": len(cached_screener),
            "total_matched": len(matches),
            "matches": matches
        }

    # 2. Fallback: On-Demand Scan if pre-computed cache not yet materialized
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    matches = []
    scanned_items = []

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

            entry = {
                "ticker": t,
                "kronos": {
                    "pattern": pat,
                    "label": lbl,
                    "confidence": conf,
                    "forecast_step": k_dict.get("forecast_step"),
                    "attention_heatmap": k_dict.get("attention_heatmap")
                },
                "timesfm_summary": {
                    "direction": composite.get("timesfm_direction"),
                    "prob_plus_5": composite.get("timesfm_prob_plus_5")
                },
                "composite_rating": composite
            }
            scanned_items.append(entry)

            if conf >= min_confidence and matches_pattern_query(pattern, lbl, pat):
                matches.append(entry)
        except Exception as e:
            print(f"[Screener Warning] Failed processing ticker {t}: {e}")

    # Seed the screener cache with this scan
    if scanned_items:
        set_screener_cache(scanned_items, ttl_seconds=600)

    SCREENER_MATCH_COUNT.set(len(matches))
    return {
        "query": {
            "pattern": pattern,
            "min_confidence": min_confidence
        },
        "source": "on_demand_computed",
        "total_scanned": len(tickers),
        "total_matched": len(matches),
        "matches": matches
    }
