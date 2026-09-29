# Unified Financial Forecasting API Guide

API Reference and Developer Documentation  
**Base URL:** `http://localhost:8000`  
**API Version:** `v1.0`  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)

---

## 1. System Health & Observability

### `GET /health`
Enterprise dependency-aware health check endpoint. Reports status, reachability, and response latencies for all critical upstream services.

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/health"
```

#### Response (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2026-09-28T08:50:00Z",
  "dependencies": {
    "redis": {
      "status": "up",
      "latency_ms": 1.2
    },
    "timesfm": {
      "status": "up",
      "device": "cuda:0",
      "default_context": 4096
    },
    "kronos": {
      "status": "up",
      "device": "cpu"
    },
    "yahoo_finance": {
      "status": "up",
      "circuit_breaker": "CLOSED"
    }
  }
}
```

---

### `GET /metrics`
Standard Prometheus exposition format endpoint for scraping application telemetry.

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/metrics"
```

#### Metrics Exported:
* `inference_requests_total{endpoint, status}`: Total HTTP requests counter.
* `inference_latency_seconds_bucket{model}`: Latency histogram for TimesFM, Kronos, and BigQuery fallbacks.
* `cache_access_total{service, hit}`: Redis cache hits and misses.
* `screener_match_count`: Current count of matching screener assets.

---

## 2. Feature Store Endpoints

### `GET /features/{ticker}`
Retrieves the latest 25 TA-Lib technical indicators and volatility regime metrics directly from the Feast online store (Redis).

#### Query Parameters:
* `ticker` *(path parameter, string, required)*: Asset ticker symbol (e.g. `AAPL`, `MSFT`, `NVDA`).

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/features/AAPL"
```

#### Response (200 OK):
```json
{
  "ticker": "AAPL",
  "features_count": 28,
  "source": "feast_redis",
  "features": {
    "sma10": 224.50,
    "sma20": 221.15,
    "sma50": 218.40,
    "sma200": 198.80,
    "rsi": 58.42,
    "macd": 2.14,
    "macd_sig": 1.85,
    "macd_hist": 0.29,
    "bb_upper": 230.10,
    "bb_middle": 221.15,
    "bb_lower": 212.20,
    "atr": 3.45,
    "bb_width": 0.081,
    "atr_percentile_20d": 0.65,
    "rsi_slope_5d": 1.20
  }
}
```

---

## 3. Foundation Model Inference Endpoints

### `GET /predict/{ticker}`
Unified multi-modal predictive pipeline combining TimesFM future probabilistic quantiles, Kronos pattern detection, technical indicators, and automated rule-based explainability.

#### Query Parameters:
* `ticker` *(path parameter, string, required)*: Asset symbol.

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/predict/NVDA"
```

#### Response (200 OK):
```json
{
  "ticker": "NVDA",
  "model_used": "timesfm",
  "generated_at": "2026-09-28T08:50:00Z",
  "current_price": 125.40,
  "predicted_price": 138.20,
  "horizon_days": 30,
  "expected_return_pct": 0.102,
  "confidence_score": 0.84,
  "prediction_interval": {
    "p10": 128.50,
    "p50": 138.20,
    "p90": 147.90
  },
  "signal": "Strong Buy",
  "reasoning": [
    "TimesFM predicts +10.2%",
    "Kronos trend confirmation (Bullish Engulfing, 88% confidence)",
    "MACD Bullish Crossover",
    "Overall Confidence: 84%"
  ],
  "timesfm_quantiles": {
    "timestamps": ["2026-09-29", "2026-09-30"],
    "point_forecast": [126.1, 127.3],
    "p10": [124.0, 124.5],
    "p50": [126.1, 127.3],
    "p90": [128.2, 130.1],
    "probability_plus_5_pct_30d": 0.76
  },
  "kronos_pattern": {
    "pattern": "CDLENGULFING",
    "label": "Bullish Engulfing",
    "confidence": 0.88,
    "forecast_step": 3
  }
}
```

---

### `GET /predict/timesfm/{ticker}`
Dedicated TimesFM 2.5 Fan Chart service. Supports long context windows up to 16,384 bars and multiple simultaneous prediction horizons.

#### Query Parameters:
* `ticker` *(path parameter, string, required)*: Asset symbol.
* `horizons` *(query parameter, string, default: `"30,60,90"`)*: Comma-separated forecast horizons.
* `max_context` *(query parameter, int, optional, default: `4096`)*: Context lookback length.

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/predict/timesfm/MSFT?horizons=30,60&max_context=4096"
```

#### Response (200 OK):
```json
{
  "ticker": "MSFT",
  "horizons": {
    "horizon_30": {
      "timestamps": ["2026-09-29", "2026-09-30"],
      "point_forecast": [445.0, 447.2],
      "p10": [435.0, 436.1],
      "p25": [440.0, 441.5],
      "p50": [445.0, 447.2],
      "p75": [450.0, 452.8],
      "p90": [455.0, 458.4],
      "probability_plus_5_pct": 0.68,
      "inference_ms": 34.2,
      "cache_hit": false,
      "fallback": false,
      "provider": "timesfm"
    }
  }
}
```

---

### `GET /predict/kronos/{ticker}`
Dedicated Kronos Candlestick Foundation Model service. Measures BSQ tokenization latency and returns extracted multi-head attention heatmaps for explainability.

#### Query Parameters:
* `ticker` *(path parameter, string, required)*: Asset symbol.
* `timeframe` *(query parameter, string, default: `"1d"`)*: Candle timeframe (`1d`, `1h`, `15m`).

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/predict/kronos/AAPL?timeframe=1d"
```

#### Response (200 OK):
```json
{
  "ticker": "AAPL",
  "timeframe": "1d",
  "pattern": "CDLENGULFING",
  "label": "Bullish",
  "confidence": 0.88,
  "forecast_step": 3,
  "tokenization_ms": 11.4,
  "cache_hit": false,
  "attention_heatmap": {
    "shape": [400],
    "weights": [0.0012, 0.0015, 0.0452]
  }
}
```

---

## 4. High-Performance Screener Endpoints

### `GET /api/v1/screeners/ai-filter`
Queries the pre-computed Redis screener cache populated by the background worker. Delivers sub-50ms responses for 50+ tickers, satisfying strict SLA requirements.

#### Query Parameters:
* `pattern` *(query parameter, string, default: `"bullish_engulfing"`)*: Target candlestick pattern filter (`bullish_engulfing`, `bearish_engulfing`, or `all`).
* `min_confidence` *(query parameter, float, default: `0.75`)*: Minimum pattern confidence threshold `[0.0, 1.0]`.

#### Curl Example:
```bash
curl -X GET "http://localhost:8000/api/v1/screeners/ai-filter?pattern=bullish_engulfing&min_confidence=0.75"
```

#### Response (200 OK):
```json
{
  "query": {
    "pattern": "bullish_engulfing",
    "min_confidence": 0.75
  },
  "source": "precomputed_redis_cache",
  "total_scanned": 61,
  "total_matched": 4,
  "matches": [
    {
      "ticker": "NVDA",
      "kronos": {
        "pattern": "CDLENGULFING",
        "label": "Bullish",
        "confidence": 0.88,
        "forecast_step": 3
      },
      "timesfm_summary": {
        "direction": "Bullish",
        "prob_plus_5": 0.76
      },
      "composite_rating": {
        "ai_rating": "Strong Bullish",
        "signal_status": "confirmation",
        "summary_reason": "Kronos trend confirmation (Bullish, 88% confidence)"
      }
    }
  ]
}
```

---

## 5. Streaming & Batch Endpoints

### `POST /predict/batch`
Batch prediction endpoint. Handles 1 to 10 tickers with individual fault tolerance.

#### Request Body:
```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"]
}
```

#### Response (200 OK):
```json
{
  "results": [ ... ],
  "failed": [],
  "duration_ms": 112.5
}
```

---

### `WebSocket /ws/screener`
Real-time streaming WebSocket endpoint for instant live push notifications whenever the background screener worker materializes a new market scan.

#### Connection:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/screener");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Live Screener Update:", data);
};
```
