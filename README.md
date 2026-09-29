# Unified Financial Forecasting Platform

[![CI Pipeline](https://github.com/shravani-gangan/finance-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/shravani-gangan/finance-forecasting/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Feast](https://img.shields.io/badge/Feast-0.65-blue?logo=feast&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Telemetry-E6522C?logo=prometheus&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An enterprise-grade, high-performance financial forecasting and quantitative screening platform combining **Google's TimesFM 2.5** foundation time-series forecaster, the **Kronos** vision-language candlestick transformer, **25 TA-Lib** indicators, and the **Feast Feature Store**.

---

## 🚀 Key Production Capabilities

* **Foundation Models at Scale:**
  * **Google TimesFM 2.5:** Scaled from legacy 512 context to **4,096 default bars** (configurable up to **16,384 bars**, ~16 years of trading history) with **FP16 half-precision** and GPU memory flushing.
  * **Kronos Candlestick Transformer:** Dynamic **INT8 quantization** for 2x faster CPU inference, plus forward hooks capturing **multi-head attention heatmaps** for visual explainability.
* **Point-in-Time Correct Feature Store:**
  * **Feast Feature Store** backed by Redis for sub-5ms online serving and Parquet for offline point-in-time correctness, mathematically eliminating lookahead bias and data leakage.
* **Sub-50ms High-Performance Screener (<2s SLA):**
  * Asynchronous background worker (`scripts/refresh_screener_cache.py`) pre-computes signals across a 61-asset universe. REST endpoint latency dropped from **191.4s to 23.3ms** (a **99.98% latency reduction**).
  * Real-time WebSocket streaming (`/ws/screener`) pushes instantaneous market scan updates.
* **Multi-Tier Resilience Engineering:**
  * 3-state Circuit Breaker (`CLOSED` -> `OPEN` -> `HALF_OPEN`) protecting TimesFM, Kronos, and Yahoo Finance.
  * Automatic graceful fallback to **BigQuery `ML.FORECAST`** or **Statistical Holt-Winters/EMA Drift** quantile forecasting.
* **Production Observability & Metrics:**
  * Prometheus exposition (`/metrics`), structured JSON logging for log aggregators, and a 5-panel **Grafana Dashboard** (`observability/grafana_dashboard.json`).
* **Containerized Deployment & CI/CD:**
  * Multi-stage `Dockerfile` with TA-Lib C library compilation, `docker-compose.yml` orchestration, and automated GitHub Actions CI.

---

## 🏗 System Architecture

```mermaid
graph TD
    Client[Web & API Clients] -->|HTTP / WebSocket| Ingress[FastAPI Gateway :8000]
    
    subgraph Observability
        Prometheus[/metrics Exposition]
        JSONLogger[Structured JSON Logs]
        Grafana[Grafana Dashboard]
    end
    
    subgraph Data & Feature Layer
        YF[Yahoo Finance API] --> DataLoader[utils/data_loader.py]
        DataLoader --> ParquetStore[(data/raw Parquet Cache)]
        DataLoader --> TALib[features/indicators.py - 25 TA-Lib + Vol Regime]
        TALib --> FeastAdapter[features/feast_wrapper.py]
        FeastAdapter --> RedisOnline[(Redis Online Store :6379)]
        FeastAdapter --> SQLiteFallback[(Legacy SQLite Store)]
    end

    subgraph Inference & Resilience Engine
        Ingress --> CircuitBreaker[utils/circuit_breaker.py]
        CircuitBreaker -->|Normal| TimesFM[models/timesfm_model.py - 4K Context, FP16]
        CircuitBreaker -->|Trip / Failure| BQFallback[models/bigquery_fallback.py - BQ ML & Drift]
        CircuitBreaker --> Kronos[models/kronos_model.py - INT8, Attention Hooks]
    end

    subgraph Signal & Screener Engine
        TimesFM --> Composite[utils/composite_signal.py]
        Kronos --> Composite
        Composite --> ScreenerCache[(Redis 'screener:latest_results')]
        Worker[scripts/refresh_screener_cache.py] -->|Every 5m| ScreenerCache
        ScreenerCache -->|PubSub| WSScreener[WebSocket /ws/screener]
        ScreenerCache -->|<50ms Response| ScreenerAPI[/api/v1/screeners/ai-filter]
    end
```

Detailed architectural decisions are documented in [docs/ARCHITECTURE.md](file:///c:/Users/hp/Downloads/finance-forecasting/docs/ARCHITECTURE.md).

---

## ⚡ Quickstart

### Option A: Docker Compose (Recommended)

Start the entire platform (FastAPI inference engine, Redis cache, and asynchronous screener worker) with one command:

```bash
docker compose up -d --build
```

Verify service health:
```bash
docker compose ps
curl http://localhost:8000/health
```

### Option B: Local / Bare-Metal Environment

#### 1. Setup Virtual Environment
```bash
python3.11 -m venv fin-env
# Linux / macOS
source fin-env/bin/activate
# Windows PowerShell
.\fin-env\Scripts\Activate.ps1
```

#### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Start Redis & API Services
```bash
# Terminal 1: Start Redis (Docker or local)
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 2: Start API Server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 3: Start Screener Background Worker
python scripts/refresh_screener_cache.py --daemon --interval 300
```

---

## 🧪 Testing & Verification

The platform maintains an automated test suite with **50+ unit and integration tests** achieving >70% coverage across core modules:

```bash
# Run pytest with coverage report
pytest --cov=utils --cov=features --cov-report=term-missing
```

| Module | Statements | Coverage |
|---|---|---|
| `features/indicators.py` | 68 | **99%** |
| `features/feature_store.py` | 56 | **95%** |
| `utils/metrics.py` | 8 | **100%** |
| `utils/config.py` | 11 | **100%** |
| `utils/logger.py` | 24 | **96%** |
| `utils/composite_signal.py` | 34 | **91%** |
| `utils/explainability.py` | 48 | **92%** |
| `utils/data_loader.py` | 51 | **88%** |
| `utils/trading_strategy.py` | 46 | **78%** |
| `utils/circuit_breaker.py` | 76 | **72%** |
| `utils/health.py` | 39 | **74%** |

---

## 📡 Core API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Enterprise dependency-aware health check (Redis, TimesFM, Kronos, Yahoo Finance) |
| `/metrics` | `GET` | Prometheus telemetry exposition format |
| `/features/{ticker}` | `GET` | Online feature store lookup (25 TA-Lib + Volatility Regime) |
| `/predict/{ticker}` | `GET` | Unified forecast (TimesFM + Kronos + Features + Signal + Reasoning) |
| `/predict/timesfm/{ticker}` | `GET` | Dedicated TimesFM multi-horizon fan charts with configurable context |
| `/predict/kronos/{ticker}` | `GET` | Dedicated Kronos candlestick pattern detection & attention heatmap |
| `/api/v1/screeners/ai-filter` | `GET` | High-performance pre-computed AI screener (<50ms response) |
| `/predict/batch` | `POST` | Batch ticker forecasting with fault isolation |
| `/ws/screener` | `WS` | Real-time WebSocket streaming live screener market updates |

Detailed curl examples, query parameters, and JSON schemas are in [docs/API_GUIDE.md](file:///c:/Users/hp/Downloads/finance-forecasting/docs/API_GUIDE.md).

---

## 📊 Observability & Monitoring

1. **Prometheus Metrics:** Available at `http://localhost:8000/metrics`.
2. **Grafana Dashboard:** Import `observability/grafana_dashboard.json` into Grafana to monitor:
   - Inference request rates and error codes.
   - P50, P95, and P99 latency percentiles by model.
   - Redis cache hit ratios.
   - AI screener match counts.
   - Live dependency health status.

---

## 📖 Complete Documentation

* [System Architecture & ADRs](file:///c:/Users/hp/Downloads/finance-forecasting/docs/ARCHITECTURE.md)
* [API Developer Guide](file:///c:/Users/hp/Downloads/finance-forecasting/docs/API_GUIDE.md)
* [Production Deployment & Operations](file:///c:/Users/hp/Downloads/finance-forecasting/docs/DEPLOYMENT.md)
* [Phase 3, 4 & 5 Verification Report](file:///c:/Users/hp/Downloads/finance-forecasting/docs/PHASE_3_4_5_REPORT.md)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/hp/Downloads/finance-forecasting/LICENSE) file for details.
