# Production Readiness Technical Debt & Architecture Audit (Day 5)

**Project:** `shravani-gangan/finance-forecasting`  
**Date:** 2026-09-04  
**Audited Directories:** `api/`, `models/`, `features/`, `utils/`, `feature_repo/`  
**Status:** Baseline Completed — Direct Input for Phases 2–7 Execution  

---

## Executive Summary

A comprehensive architectural and static code analysis was conducted across the platform's core modules. While the existing MVP successfully validates the multi-model forecasting concept (TimesFM + Kronos + TA-Lib signals), the audit revealed critical structural vulnerabilities, thread safety flaws, performance bottlenecks, and infrastructure coupling that prevent production deployment.

This document classifies all identified technical debt into **🔴 P0 (Production Blockers)**, **🟡 P1 (Reliability & Performance Gaps)**, and **🟢 P2 (Maintainability & Observability)**, mapping each to its respective resolution phase in the engineering roadmap.

---

## 🔴 P0: Critical Production Blockers (Must Fix for Launch)

### P0-1: Feature Store SQLite EAV Hack & Data Leakage Risk
* **Files:** [features/feature_store.py](file:///c:/Users/hp/Downloads/finance-forecasting/features/feature_store.py#L16-L107), [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L98-L103)
* **Description:** Feature storage relies on an SQLite database (`features_long`) using an Entity-Attribute-Value (EAV) schema (`ticker, date, feature_name, feature_value`). 
* **Impact:** 
  1. **No Point-in-Time Correctness:** Backtests and offline training queries cannot guarantee that feature values observed for a date were actually available at inference time without future data leaking into the calculation.
  2. **Concurrency Lock Contention:** SQLite write locks trigger `sqlite3.OperationalError: database is locked` when multiple concurrent requests attempt to compute and persist features simultaneously.
  3. **High Latency Serialization:** Storing features in long-format requires constant melting, inserting, and pivoting on every query.
* **Resolution (Phase 2):** Full migration to Feast with Redis (`127.0.0.1:6379`) as the online feature store and Parquet for offline point-in-time correctness. Maintain dual-write fallback during the transition.

### P0-2: TimesFM 2.5 Context Length Artificially Hardcoded to 512
* **Files:** [models/timesfm_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/timesfm_model.py#L50-L56)
* **Description:** Inside `TimesFMForecast.predict()`, `ForecastConfig` is hardcoded with `max_context=512`.
* **Impact:** Google TimesFM 2.5 is designed to operate on context windows up to **16,384** time steps (over 16 years of daily market trading). Artificially restricting the context to 512 tokens (~2 years) throws away valuable macroeconomic regimes, market cycles, and historical support/resistance levels.
* **Resolution (Phase 3):** Parameterize `max_context` via environment variable `TIMESFM_MAX_CONTEXT` defaulting to `4096` with support up to `16384` (ADR-002).

### P0-3: Kronos RoPE Attention Buffer Thread Collision Crash
* **Files:** [models/kronos_repo/model/module.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/kronos_repo/model/module.py#L295-L315), [models/kronos_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/kronos_model.py#L39-L82)
* **Description:** Kronos uses `RotaryPositionalEmbedding` inside `MultiHeadAttentionWithRoPE`. Sequence caches (`self.seq_len_cached`, `self.cos_cached`, `self.sin_cached`) are stored directly as mutable instance variables on `self`.
* **Impact:** When concurrent HTTP requests trigger Kronos inference simultaneously, worker threads overwrite these cached tensors mid-generation, throwing:
  ```text
  RuntimeError: The size of tensor a (404) must match the size of tensor b (400) at non-singleton dimension 2
  ```
* **Resolution (Phase 3):** Implement thread synchronization locks (`threading.Lock`) around Kronos forward passes or refactor the positional embedding cache to be thread-local.

### P0-4: Screener Violates <2.0s Production SLA (Takes ~197.5s)
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L409-L476)
* **Description:** The `/api/v1/screeners/ai-filter` endpoint processes tickers sequentially on-demand inside the HTTP request lifecycle. Scanning 5 tickers takes **197.5 seconds** (~3.3 minutes), exceeding the SLA by nearly 100x.
* **Impact:** Interactive and algorithmic clients experience connection timeouts and total server gridlock under concurrent requests.
* **Resolution (Phase 4):** Implement background pre-computation worker ([scripts/refresh_screener_cache.py](file:///c:/Users/hp/Downloads/finance-forecasting/scripts/refresh_screener_cache.py)) that caches composite signals in Redis under `screener:latest_results` every 5 minutes, reducing query response times to `<50ms` (ADR-003).

### P0-5: Kronos Attention Heatmaps Stubbed as `None`
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L319), [models/kronos_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/kronos_model.py)
* **Description:** The API schema defines `attention_heatmap`, but `predict_kronos_ticker()` hardcodes `"attention_heatmap": None`.
* **Impact:** Frontend interfaces and quantitative analysts cannot inspect model interpretability or determine which historical candles influenced the forecast.
* **Resolution (Phase 3):** Hook into `MultiHeadAttentionWithRoPE` when `extract_attention=True` to compute self-attention weight distributions across transformer layers and return structured heatmaps.

### P0-6: Blocking Synchronous Model & I/O Execution in FastAPI Routes
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L67-L199), [utils/data_loader.py](file:///c:/Users/hp/Downloads/finance-forecasting/utils/data_loader.py#L18)
* **Description:** `predict_ticker()`, `predict_batch()`, and `load_data()` execute heavy CPU-bound PyTorch inference and blocking network I/O (`yf.download`) directly in synchronous endpoint handlers.
* **Impact:** Blocks the Python async event loop, preventing FastAPI from processing lightweight health checks (`/health`) and incoming WebSocket connections.
* **Resolution (Phase 4):** Offload CPU-bound inference and external data downloading to a dedicated `ThreadPoolExecutor` using `starlette.concurrency.run_in_threadpool`.

### P0-7: Inactive WebSocket Route with Busy-Wait Loop
* **Files:** [api/websocket.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/websocket.py#L20-L37)
* **Description:** The `/ws/screener` endpoint connects to `localhost` and polls Redis pub/sub using a busy-wait loop (`while True: await asyncio.sleep(0.01)`).
* **Impact:** Inefficient CPU utilization and failure to deliver real-time alert broadcasts to connected clients.
* **Resolution (Phase 4):** Replace polling with asynchronous iterator `async for message in pubsub.listen():`, add ping/pong heartbeats, and bind to `127.0.0.1`.

---

## 🟡 P1: High Priority (Reliability, Scalability & Performance Gaps)

### P1-1: Hardcoded `localhost:6379` Triggering Windows 2,000ms IPv6 Timeout
* **Files:** [utils/cache.py](file:///c:/Users/hp/Downloads/finance-forecasting/utils/cache.py#L19), [api/websocket.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/websocket.py#L20), [feature_repo/feature_store.yaml](file:///c:/Users/hp/Downloads/finance-forecasting/feature_repo/feature_store.yaml#L6)
* **Description:** Redis connection host is hardcoded to `localhost` instead of reading from configuration or environment variables. On Windows, `localhost` triggers a 2,000ms IPv6 `::1` resolution attempt before failing over to IPv4 `127.0.0.1`.
* **Impact:** Inflates cache lookups from **4.3 ms** to **2,031 ms**.
* **Resolution:** Standardize on `os.getenv("REDIS_HOST", "127.0.0.1")` and `os.getenv("REDIS_PORT", "6379")` across all modules.

### P1-2: Lack of Model Quantization (INT8 CPU / FP16 GPU)
* **Files:** [models/kronos_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/kronos_model.py), [models/timesfm_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/timesfm_model.py)
* **Description:** Models execute unquantized FP32 forward passes on CPU devices, taking ~37.6s for Kronos and ~1.9s for TimesFM.
* **Impact:** High CPU utilization, high memory footprint, and slow response times under traffic.
* **Resolution (Phase 3):** Implement dynamic INT8 quantization (`torch.quantization.quantize_dynamic`) for Kronos Linear layers and FP16 mode (`.half()`) on CUDA/MPS devices.

### P1-3: Unified `/predict/{ticker}` Missing Top-Level Cache Layer
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L66-L188)
* **Description:** Unlike `/predict/kronos/{ticker}` and `/predict/timesfm/{ticker}` which cache results in Redis, the unified endpoint `/predict/{ticker}` re-runs full feature extraction, TimesFM, and Kronos on every call.
* **Impact:** Users calling the main unified endpoint always experience a ~40-second response latency.
* **Resolution (Phase 4):** Wrap unified endpoint response with Redis cache key `unified:{ticker}` and 300s TTL.

### P1-4: Hardcoded `lookback=400` in Kronos Sequence Modeling
* **Files:** [models/kronos_model.py](file:///c:/Users/hp/Downloads/finance-forecasting/models/kronos_model.py#L39), [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L279)
* **Description:** Kronos sequence length is hardcoded to 400 throughout the application. Assets with fewer than 400 trading days fail inference, and longer historical contexts cannot be leveraged.
* **Resolution (Phase 3):** Support dynamic lookback with adaptive padding/truncation.

### P1-5: Hardcoded 5-Ticker Universe in Screener
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L417)
* **Description:** `tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]` is hardcoded inside the route handler.
* **Resolution (Phase 4):** Externalize universe definition to `config/universe.yaml` supporting 50+ tickers across equities, indices, and crypto.

### P1-6: Unhandled External Data Fetch Exceptions in `load_data()`
* **Files:** [utils/data_loader.py](file:///c:/Users/hp/Downloads/finance-forecasting/utils/data_loader.py#L18)
* **Description:** `yf.download()` call lacks timeout traps and HTTP retry logic. Network disconnects or Yahoo Finance rate limiting crash the entire prediction request.
* **Resolution (Phase 5):** Wrap with exponential backoff retries, offline Parquet fallback, and synthetic/BigQuery fallback.

### P1-7: Inconsistent Signal Engine Confidence Parameters
* **Files:** [utils/trading_strategy.py](file:///c:/Users/hp/Downloads/finance-forecasting/utils/trading_strategy.py#L106), [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py#L131)
* **Description:** `generate_signal_v2` defines default `min_confidence=0.653`, while `api/app.py` passes `min_confidence=0.4`. In `utils/trading_strategy.py:69`, sell signal probability confirmation logic contains an unhandled `pass` block.
* **Resolution (Phase 4):** Centralize trading strategy hyperparameters in `config/trading_params.yaml`.

---

## 🟢 P2: Medium Priority (Maintainability, Testing & Observability)

### P2-1: Lack of Prometheus Observability Exporter (`/metrics`)
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py)
* **Description:** No telemetry endpoint exists to monitor request counts, latency percentiles, cache hit ratios, or model inference times in Prometheus/Grafana.
* **Resolution (Phase 6):** Integrate `prometheus-fastapi-instrumentator` with custom counters and histograms.

### P2-2: Unstructured Logging across Codebase
* **Files:** Global across `api/`, `models/`, `utils/`
* **Description:** Application logs are emitted via unstructured standard `print()` statements and standard `traceback.print_exc()`.
* **Resolution (Phase 6):** Implement structured JSON logging (`structlog` or `logging.dictConfig`) with correlation IDs.

### P2-3: Missing Automated Unit & Regression Test Suite
* **Files:** Project root
* **Description:** Verification currently relies on manual execution scripts (`scripts/`). No formal `pytest` suite tests endpoint contracts, mock fallbacks, or schema validations.
* **Resolution (Phase 6):** Deliver comprehensive unit and integration tests under `tests/` covering endpoints, Feast fallback, and model wrappers.

### P2-4: Missing Containerization & Orchestration
* **Files:** Project root
* **Description:** No `Dockerfile` or `docker-compose.yml` exists. Deploying to Kubernetes or Cloud Run requires manual environment setup.
* **Resolution (Phase 7):** Create multi-stage production `Dockerfile` and compose stack configuring FastAPI, Redis, and Feast.

### P2-5: Lack of API Rate Limiting
* **Files:** [api/app.py](file:///c:/Users/hp/Downloads/finance-forecasting/api/app.py)
* **Description:** Endpoints lack rate limiting middleware, allowing malicious or runaway clients to saturate model inference resources.
* **Resolution (Phase 5):** Implement Redis-backed token bucket rate limiting middleware (`slowapi`).

---

## Issue Priority Matrix & Roadmap Mapping

| Issue ID | Priority | Category | Module / File | Target Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **P0-1** | 🔴 P0 | Data Integrity | `features/feature_store.py` | **Phase 2 (Feast Migration)** |
| **P0-2** | 🔴 P0 | Model Scaling | `models/timesfm_model.py` | **Phase 3 (Context Sizing)** |
| **P0-3** | 🔴 P0 | Thread Safety | `models/kronos_repo/` | **Phase 3 (Concurrency Lock)** |
| **P0-4** | 🔴 P0 | Performance / SLA | `api/app.py` | **Phase 4 (Pre-computed Screener)** |
| **P0-5** | 🔴 P0 | Interpretability | `models/kronos_model.py` | **Phase 3 (Attention Heatmap)** |
| **P0-6** | 🔴 P0 | Async Architecture | `api/app.py` | **Phase 4 (ThreadPool Offload)** |
| **P0-7** | 🔴 P0 | Real-time Streaming| `api/websocket.py` | **Phase 4 (WebSocket Listener)** |
| **P1-1** | 🟡 P1 | Networking | `utils/cache.py` | **Phase 2 & 4 (Host Config)** |
| **P1-2** | 🟡 P1 | Latency / CPU | `models/` | **Phase 3 (Quantization)** |
| **P1-3** | 🟡 P1 | Caching | `api/app.py` | **Phase 4 (Unified Cache)** |
| **P1-4** | 🟡 P1 | Flexibility | `models/kronos_model.py` | **Phase 3 (Dynamic Lookback)** |
| **P1-5** | 🟡 P1 | Configuration | `api/app.py` | **Phase 4 (Universe YAML)** |
| **P1-6** | 🟡 P1 | Resilience | `utils/data_loader.py` | **Phase 5 (Fallback Architecture)**|
| **P1-7** | 🟡 P1 | Strategy Logic | `utils/trading_strategy.py`| **Phase 4 (Config Parameters)**|
| **P2-1** | 🟢 P2 | Observability | `api/app.py` | **Phase 6 (Prometheus Metrics)** |
| **P2-2** | 🟢 P2 | Observability | Global | **Phase 6 (Structured Logging)** |
| **P2-3** | 🟢 P2 | Quality Assurance | `tests/` | **Phase 6 (Pytest Suite)** |
| **P2-4** | 🟢 P2 | DevOps | Root | **Phase 7 (Docker / Compose)** |
| **P2-5** | 🟢 P2 | Security | `api/app.py` | **Phase 5 (Rate Limiting)** |
