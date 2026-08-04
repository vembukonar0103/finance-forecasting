# Project Handover Document: Unified Finance Forecasting Engine

This document provides all necessary technical details for a developer or manager to assume ownership, run, debug, and maintain the Unified Finance Forecasting Engine.

---

## 1. Project Overview
The Unified Finance Forecasting Engine is an AI-powered financial forecasting, technical analysis, and screening platform. It integrates advanced machine learning foundation models (Google's TimesFM and Kronos) with traditional quantitative indicators (TA-Lib) to generate multi-horizon forecasts, detect candlestick patterns, and compute composite trading signals. The system serves cached real-time predictions via FastAPI REST endpoints and streams live screener updates over WebSockets to support downstream automated trading workflows.

---

## 2. Implemented Features
*   **TimesFM Forecasting:** Probabilistic multi-horizon time-series forecasting (producing decile bands p10 through p90) and heuristic tail return probability calculation.
*   **Kronos Candlestick Detection:** Pattern prediction and evaluation utilizing `TA-Lib`'s pattern recognition (specifically `CDLENGULFING`) on forecasted futures.
*   **TA-Lib Indicators:** High-performance calculation of 21 technical indicators (moving averages, momentum oscillators, volatility, and volume metrics).
*   **Feast Feature Store:** SQLite-backed offline feature repository and Redis-backed online feature store mapping for point-in-time correctness.
*   **Redis Cache:** TTL-based (300s) request caching for model inference endpoints and a Pub/Sub backbone for real-time broadcasts.
*   **FastAPI REST APIs:** High-performance endpoints for single-asset prediction, batch forecasts, and screener queries.
*   **AI Screener:** Dynamic scanner filtering candidate assets against pattern detection and confidence thresholds.
*   **Composite Signal Engine:** Rules-based confirmation/conflict engine aligning TimesFM statistical forecasts and Kronos structural patterns.
*   **WebSocket Notifications:** Real-time push notification streaming for live screener updates.
*   **Walk-Forward Validation:** Robust backtesting framework with look-ahead bias prevention and rolling window generators.
*   **Benchmark Utilities:** Telemetry and comparison scripts measuring inference runtimes, memory usage, and forecast accuracy.

---

## 3. External Dependencies
The system relies on the following key dependencies:

| Dependency | Purpose |
| :--- | :--- |
| **TimesFM** | Probabilistic zero-shot foundation model for time-series forecasting. |
| **Kronos** | Foundational sequence model for structural candlestick and OHLCV prediction. |
| **Redis** | In-memory key-value cache database and async Pub/Sub channel manager. |
| **Feast** | Feature store to orchestrate offline feature materialization and online feature queries. |
| **TA-Lib** | C++ technical indicator library providing fast financial mathematics functions. |
| **Yahoo Finance** | Core historical market data source (queried via the `yfinance` python library). |

---

## 4. Prerequisites
Ensure the following system-level tools are installed on your host system:
*   **Python:** Version `3.11` is strictly required (dependencies may compile incorrectly on `3.12+`).
*   **Git:** Required for cloning the repository and submodules.
*   **Redis:** Version `6.x` or `7.x`.
*   **TA-Lib C++ Library:** Underlying binary headers are needed before running `pip install`.
    *   *macOS:* Install via Homebrew: `brew install ta-lib`
    *   *Linux:* Compile from source (download `ta-lib-0.4.0-src.tar.gz`, then execute `./configure && make && sudo make install`).
    *   *Windows:* Install via `conda install -c conda-forge ta-lib` or download precompiled C++ binaries and point headers locally.

---

## 5. Setup Instructions (Step-by-Step)

### Step 1: Clone the Repository
```bash
git clone <repository-url> finance-forecasting
cd finance-forecasting
```

### Step 2: Create a Virtual Environment
**macOS / Linux:**
```bash
python3.11 -m venv fin-env
source fin-env/bin/activate
```
**Windows (PowerShell):**
```powershell
python -m venv fin-env
.\fin-env\Scripts\Activate.ps1
```
**Windows (Command Prompt):**
```cmd
fin-env\Scripts\activate.bat
```

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note for Windows users:* If `TA-Lib` compilation fails, download the precompiled `.whl` from an unofficial repository (e.g., Christoph Gohlke's archives) matching your architecture and python version, and install it directly: `pip install TA_Lib-0.7.0-cp311-cp311-win_amd64.whl`.

### Step 4: Start Redis
**macOS:**
```bash
brew services start redis
```
**Linux:**
```bash
sudo systemctl start redis-server
```
**Windows (Recommended using Docker):**
```powershell
docker run -d -p 6379:6379 --name fin-redis redis:alpine
```
*(Alternatively, you can run Redis under WSL2 or use Memurai).*

### Step 5: Clone the External Kronos Dependency
Cloning this submodule is a manual step. Run this command from the repository root:
```bash
git clone https://github.com/shiyu-coder/Kronos.git models/kronos_repo
```

### Step 6: Initialize the Feast Feature Store
Navigate to the feature repository and apply definitions:
```bash
cd feature_repo
feast apply
cd ..
```

### Step 7: Environment Variables Setup
Currently, the codebase does not employ a `.env` file or `os.getenv` configuration; all credentials and connection strings are hardcoded to local defaults. If setting up for remote production, you must modify:
*   Redis Connection: Hardcoded to `localhost:6379` in `utils/cache.py`, `api/websocket.py`, `scripts/test_publish_update.py`, and `feature_repo/feature_store.yaml`.
*   SQLite paths: Localized in `utils/config.py`.

### Step 8: Start the API
```bash
uvicorn api.app:app --reload
```

### Step 9: Verify Setup
Submit a test request using curl (PowerShell users can run `Invoke-RestMethod`):
```bash
curl -X GET "http://localhost:8000/predict/AAPL"
```
You should receive a unified JSON payload containing TimesFM forecast bounds, Kronos detected patterns, latest technical indicators, and a composite signal.

---

## 6. Project Structure Map
```text
finance-forecasting/
├── api/                   # REST API routes, schemas, and WebSocket pub/sub logic
│   ├── app.py             # Primary FastAPI app declaration and endpoint routing
│   ├── websocket.py       # Asynchronous WebSocket endpoint broadcasting screener updates
│   └── schemas/           # Pydantic schemas validating API inputs/outputs
├── data/                  # Local parquet caches, benchmark metrics, and SQLite experiments database
├── feature_repo/          # Feast registry, schemas, and database configuration
├── feature_store/         # Holds local sqlite features store (finance_features.db)
├── features/              # Feature engineering and storage modules
│   ├── indicators.py      # Calculations for 21 technical indicators using TA-Lib
│   ├── feature_store.py   # Long-format SQLite storage CRUD utilities
│   └── feast_store.py     # Feast materialization and loading wrappers
├── models/                # ML wrappers, hardware routing, and ensemble calculations
│   ├── device.py          # Auto-detects acceleration backend (CUDA / MPS / CPU)
│   ├── timesfm_model.py   # PyTorch wrapper for TimesFM 2.5 200M model
│   ├── kronos_model.py    # PyTorch wrapper for Kronos pattern model
│   ├── ensembles.py       # Inverse-variance weighted forecasting logic
│   ├── ml_forecasters.py  # Benchmark baseline Ridge and RF models
│   └── kronos_repo/       # Cloned external Kronos code (ignored in Git)
├── plots/                 # Matplotlib outputs showing predictions and backtest curves
├── reports/               # Markdown and docx project analysis reports
├── scripts/               # Benchmark suites, verifications, and smoke tests
└── utils/                 # Caching, risk engine, backtesting, and portfolio optimizer
```

---

## 7. Architecture at a Glance
The system processes data sequentially from ingestion to delivery:
1.  **Yahoo Finance** is polled for historical data.
2.  **TA-Lib** processes this data to generate 21 technical indicators.
3.  **Feast** stores these indicators offline in SQLite and serves them online via Redis.
4.  **TimesFM & Kronos** wrappers run model inference on the target CPU or GPU device (auto-routed by `models/device.py`).
5.  **Composite Signal Engine** processes model predictions to determine trade direction and conflict states.
6.  **Redis Cache** checks for hits and caches new output values for 300 seconds.
7.  **FastAPI REST endpoints** serve predictions to HTTP requests.
8.  **WebSockets** stream updates pushed to Redis Pub/Sub directly to subscribers.

For a deeper dive into the mathematical models, ensembles, and research benchmarks, refer to [reports/detailed_project_report.md](file:///Users/omkargangan/Desktop/Shravani/finance_proj/reports/detailed_project_report.md).

---

## 8. Key Files — Ownership Map

| File/Module | What It Owns | Who Should Touch This First |
| :--- | :--- | :--- |
| [api/app.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/api/app.py) | API entrypoint, routing, preloading logic, batch endpoint concurrency control. | Backend / Infrastructure Engineer |
| [api/websocket.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/api/websocket.py) | Async Redis connection, WebSocket broadcast routing, channel subscription. | Backend / Infrastructure Engineer |
| [models/timesfm_model.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/models/timesfm_model.py) | TimesFM model loading, compiling, forecasting, tail probability. | ML Engineer / Quant Researcher |
| [models/kronos_model.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/models/kronos_model.py) | Kronos model loading, forecasting, TA-Lib pattern detection on forecast. | ML Engineer / Quant Researcher |
| [features/indicators.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/features/indicators.py) | TA-Lib indicator calculations (21 features). | Quant Developer / Data Engineer |
| [utils/trading_strategy.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/utils/trading_strategy.py) | 5-state signal thresholds, target parameters, signal generation. | Portfolio Manager / Quant Researcher |
| [utils/risk_engine.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/utils/risk_engine.py) | Volatility classifications, ATR stop loss / take profit, size multipliers. | Risk Manager / Portfolio Manager |
| [utils/cache.py](file:///Users/omkargangan/Desktop/Shravani/finance_proj/utils/cache.py) | Synchronous Redis connection pool, cache getters and setters. | Infrastructure Engineer |

---

## 9. API Quick Reference

| Endpoint | Method | Description | Cache Status | Model |
| :--- | :--- | :--- | :--- | :--- |
| `/predict/{ticker}` | GET | Combined forecast prediction with indicator extraction and composite signal. | Redis (300s) | TimesFM + Kronos |
| `/predict/kronos/{ticker}` | GET | Kronos-specific pattern detection and tokenization timings. | Redis (300s) | Kronos |
| `/predict/timesfm/{ticker}` | GET | TimesFM-specific multi-horizon quantile forecast bands. | Redis (300s) | TimesFM |
| `/predict/batch` | POST | Sequentially processes batch of forecasts for a list of tickers. | Redis (300s) | TimesFM + Kronos |
| `/api/v1/screeners/ai-filter` | GET | Scans candidate tickers for pattern matches meeting a min confidence limit. | None | Composite |
| `/ws/screener` | WebSocket | Pushes live screener alerts published to Redis channel. | Redis Pub/Sub | N/A |
| `/health` | GET | Verification endpoint check indicating model load statuses. | None | None |

---

## 10. How to Run Things

### Running Utility and Core Scripts
*   **Run Backtest Pipeline:** Execute a 5-asset, 5-year walk-forward backtest and save comparative plots:
    ```bash
    python scripts/run_quantitative_research.py
    ```
*   **Run TimesFM Isolated Validation:** Run TimesFM prediction and save forecast plots:
    ```bash
    python scripts/run_timesfm.py
    ```
*   **Run Kronos Isolated Validation:** Validate Kronos forecasts and save outputs:
    ```bash
    python scripts/run_kronos.py
    ```
*   **Populate Feature Store:** Fetch data and write indicators to offline storage:
    ```bash
    python scripts/populate_feature_store.py
    ```
*   **Quantization Benchmarking:** Run FP16 and INT8 execution timing tests on MPS:
    ```bash
    python scripts/investigate_quantization.py
    ```

### Running Tests
*   **Single Ticker API Test:** Run FastAPI smoke test for `/predict/AAPL`:
    ```bash
    python scripts/test_api.py
    ```
*   **Batch Prediction API Test:** Run concurrent test for multiple tickers:
    ```bash
    python scripts/test_batch_api.py
    ```

### Running Verification Passes
*   **Verify Feature Ingestion:** Execute indicator extraction and EAV SQLite writing checks:
    ```bash
    python scripts/verify_features.py
    ```
*   **Verify Strategy Elements:** Validate signal engine, risk positions, optimizer allocations, and SQL telemetry:
    ```bash
    python scripts/verify_pipeline_real_data.py
    ```

---

## 11. How to Add a New Stock
To support a new financial asset in the system:
1.  **Register Ticker:** Open `scripts/run_quantitative_research.py` and add your ticker to `TICKERS`. Eq. `TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD", "GOOGL"]`.
2.  **Define Frequency:** In `scripts/run_quantitative_research.py`, map its frequency in `TICKER_FREQ` (e.g., `'D'` for 24/7 crypto, `'B'` for business-day equities).
3.  **Fetch & Ingest Data:** Run the ingestion pipeline to fetch data from Yahoo Finance, compute TA-Lib indicators, and store them in SQLite:
    ```python
    from utils.data_loader import load_data
    from features.indicators import build_features
    from features.feature_store import store_features
    
    df = load_data("GOOGL", period="2y", interval="1d")
    feats_df = build_features(df)
    store_features("GOOGL", feats_df)
    ```
4.  **Materialize feature store:** If using Feast views, run `feast materialize` in the `feature_repo` directory.
5.  **Verify Endpoint:** Run the API and request a prediction to verify:
    ```bash
    curl -X GET "http://localhost:8000/predict/GOOGL"
    ```

---

## 12. Configuration & Tunables
Key system constants are defined in the following locations:
*   **Redis Caching TTL (300 seconds):** Defined in `utils/cache.py` (via `ttl_seconds=300` parameter default) and inside the API routes in `api/app.py`.
*   **TimesFM Context Window (512 observations):** Hardcoded inside `models/timesfm_model.py` in `max_context=512`.
*   **Kronos Context Lookback (400 observations):** Hardcoded in `models/kronos_model.py` and `scripts/run_kronos.py` as `lookback=400`.
*   **Trading Transaction Costs (10 bps / 0.0010):** Set in `scripts/run_quantitative_research.py` as `tx_cost_pct=0.0010`.
*   **Screener Confidence Threshold (default 0.75):** Set in `api/app.py` under the `/api/v1/screeners/ai-filter` function parameter `min_confidence=0.75`.
*   **API Port (8000):** Defined at start-up (default for Uvicorn).

---

## 13. How to Debug
Follow these diagnostic steps if a component fails:

*   **If Redis Connection Fails:**
    Verify your Redis server is running locally:
    ```bash
    redis-cli ping
    ```
    *Expected output:* `PONG`. If it fails, check system logs or Docker desktop.
*   **If Feast Ingestion Fails:**
    Validate Feast registry state and reload schemas:
    ```bash
    cd feature_repo
    feast status
    feast apply
    ```
*   **If API Server Crashes on Boot:**
    Ensure python environment `fin-env` is active and model weights are pre-downloaded from HuggingFace. Run manually to print traceback:
    ```bash
    python -m uvicorn api.app:app --reload
    ```
*   **If WebSocket Subscriptions Do Not Receive Data:**
    Ensure Redis server is running. While connected to `/ws/screener`, run this test script in another terminal to publish a mock payload:
    ```bash
    python scripts/test_publish_update.py
    ```

---

## 14. Known Issues / Fragile Areas
*   **PyTorch MPS Concurrency Deadlocks:** Concurrent execution of model `.predict` calls on Mac Apple Silicon GPUs (MPS) causes PyTorch to hang. The `app.py` batch endpoint (`/predict/batch`) uses sequential loops to bypass this issue. Do not parallelize these calls on Mac platforms.
*   **Manual Dependency on External Repository:** The Kronos model requires a clone of `https://github.com/shiyu-coder/Kronos.git` to exist at `models/kronos_repo/`. If missing, the app fails on startup.
*   **Hardcoded Connections:** Redis settings are hardcoded to `localhost` on port `6379`. Moving the app to separate containers requires manual replacement of connection hosts.
*   **Timezone Date Alignment:** Date matching in backtester utilizes Pandas nearest-match to handle differences between calendar days (crypto) and business days (equities).

---

## 15. Security Notes
*   **Hardcoded Redis Connection String:** Connection parameters are hardcoded as `localhost:6379` across multiple python files (`utils/cache.py`, `api/websocket.py`, `scripts/test_publish_update.py`) and inside `feature_repo/feature_store.yaml`. These must be moved to environment variables prior to container staging.
*   **Excluded Files in `.gitignore`:**
    *   Virtual environments (`fin-env/`, `venv/`).
    *   Environment configuration (`.env`).
    *   Offline databases and local caches (`data/raw/`, `data/processed/`, `data/experiments.db`, `data/features_wide.parquet`).
    *   Generated model prediction tables (`data/predictions/` except `benchmark_results.csv`).
    *   Cloned dependency folder `models/kronos_repo/` is ignored to prevent pushing external codebase assets.

---

## 16. Testing Checklist
Verify these items pass before committing changes:
*   [ ] Redis server is running and responds to `redis-cli ping` with `PONG`.
*   [ ] Feast configurations are applied (`feast apply` inside `feature_repo/`).
*   [ ] FastAPI server starts with `uvicorn` showing both TimesFM and Kronos models loaded successfully.
*   [ ] Swagger documentation is accessible at `http://localhost:8000/docs`.
*   [ ] Endpoint `/predict/timesfm/AAPL` returns valid p10-p90 forecast data.
*   [ ] Endpoint `/predict/kronos/AAPL` returns predicted candlesticks and tokenization timings.
*   [ ] Filter `/api/v1/screeners/ai-filter` successfully processes scans over the ticker universe.
*   [ ] WebSocket connection to `ws://localhost:8000/ws/screener` establishes, and testing script `python scripts/test_publish_update.py` pushes mock alerts successfully.
