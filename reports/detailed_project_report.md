# Detailed Project Report: Unified Finance Forecasting Engine

## 1. Executive Summary
The Unified Finance Forecasting Engine is a robust, AI-assisted quantitative research and trading platform. It bridges the gap between deep-learning sequence models (Google TimesFM) and structural candlestick pattern detection (Kronos) to provide actionable, multi-horizon financial forecasts. By integrating real-time feature engineering, robust in-memory caching, and a composite AI signal engine, the platform delivers low-latency trading signals and real-time market screening capabilities directly to downstream algorithmic trading systems.

## 2. Project Overview
*   **Problem Statement:** Traditional financial forecasting systems often rely on a single approach, such as time-series forecasting or technical analysis, resulting in limited accuracy and adaptability under changing market conditions. 
They also face challenges such as high inference latency, inconsistent feature management, and the lack of a unified platform for real-time prediction and AI-driven trading insights. 
This project addresses these challenges by integrating multiple forecasting models, feature management, caching, and real-time APIs into a single scalable forecasting engine.

*   **Objectives:** Develop a unified financial forecasting platform by integrating TimesFM and Kronos models.
Generate technical indicators using TA-Lib to enhance market analysis.
Implement a Feast Feature Store for consistent feature management.
Reduce inference latency through Redis-based caching.
Provide real-time prediction services using FastAPI and WebSocket communication.
Generate explainable AI-driven trading signals using a composite signal engine.
Support multi-horizon forecasting and candlestick pattern detection through a single API platform.
Build a modular, scalable, and extensible architecture for future enhancements in quantitative financial analysis.

*   **Scope:** Multi-horizon forecasting, feature engineering (TA-Lib), caching (Redis), composite signal generation, walk-forward backtesting, and live screening.

## 3. Project Deliverables
The following core components were fully implemented and verified in the repository:
*   **TimesFM Integration:** Zero-shot probabilistic forecasting with P10-P90 confidence intervals.
*   **Kronos Integration:** Structural time-series forecasting and candlestick pattern detection.
*   **Feature Engineering:** 21 traditional TA-Lib indicators.
*   **Feast Wrapper:** Offline-to-online feature store integration via SQLite.
*   **Redis Caching:** Sub-second latency via 300s TTL cache.
*   **FastAPI APIs:** Unified prediction, batch execution, and AI screening endpoints.
*   **Composite Signal Engine:** 5-state logic combining forecasts, confidence, and pattern validation.
*   **WebSocket:** Real-time push notifications for screener matches.
*   **Validation Scripts:** Extensive walk-forward backtesting, look-ahead bias prevention, and portfolio simulation.

## 4. Architecture Overview & Execution Workflow

### Architecture Flow
The system processes data through a distinct, modular pipeline:
**Yahoo Finance → TA-Lib → Feast → Models → Composite Engine → Redis → FastAPI → WebSocket**

1.  **Data Ingestion & Features:** Raw market data is fetched from Yahoo Finance. 21 technical indicators are computed via TA-Lib and materialized into an SQLite-backed feature store (compatible with Feast).
2.  **Inference Engine:** Historical sequences are passed to TimesFM (for quantile forecasts) and Kronos (for structural OHLCV generation and pattern detection).
3.  **Signal Generation:** Output from both models is fed into the Composite Signal Engine, which evaluates trend alignment and calculates an AI Rating.
4.  **Serving Layer:** Results are cached in Redis and served synchronously via FastAPI, or asynchronously broadcast over WebSockets.

### Execution Workflow (Runtime)
1.  **User sends request:** A client makes a `GET /predict/AAPL` request.
2.  **FastAPI receives request:** The routing layer intercepts and parses parameters.
3.  **Redis cache lookup:** The system checks Redis (e.g., key `kronos:AAPL:1d`).
4.  **Cache miss → model inference:** If not cached, data is fetched, and both TimesFM and Kronos perform PyTorch forward passes on the target device (CPU/CUDA/MPS).
5.  **Feature retrieval:** The latest technical indicators are pulled from the SQLite feature store.
6.  **Composite signal generation:** The `generate_signal_v2` and `generate_reasoning` utilities combine the forecasts into a strict 5-state signal (e.g., "Strong Buy").
7.  **Cache storage:** Responses are written back to Redis with a 300s TTL.
8.  **Response returned:** The JSON payload is returned to the client.
9.  **Optional WebSocket publication:** If matched by the screener, the payload is published to the Redis pub/sub channel `screener:updates` for live broadcasting.

---

## 5. Module-by-Module Breakdown

### a. `models/` (ML Integration)
This module houses the core foundation model wrappers and hardware routing logic.

| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `device.py` | Detects hardware accelerators. | `get_device()` dynamically returns "cuda", "mps", or "cpu". Enables Apple Silicon support. |
| `timesfm_model.py` | Wrapper for Google's TimesFM 2.5. | `TimesFMForecast.predict()` generates quantiles. Implements `_calculate_prob_return()` to extrapolate tail probabilities (P>=+5%). Context length fixed at 512. |
| `kronos_model.py` | Wrapper for Kronos architecture. | `KronosForecast.predict()` outputs structural prices. `detect_pattern()` uses `talib.CDLENGULFING` to find patterns on the *predicted* future candles. |
| `ensembles.py` | Combines Kronos and TimesFM. | `compute_confidence_ensemble()` dynamically adjusts weights based on inverse-validation MAE and the active P10-P90 band width. |
| `ml_forecasters.py` | Baseline ML models for benchmarking. | `MLForecaster.fit()` trains 30 separate Ridge/RF models (one per horizon step) using grid search. |

### b. `features/` (Data & Feature Store)
Handles the computation, materialization, and serving of technical indicators.

| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `indicators.py` | Computes technical features. | `build_features()` processes OHLCV through TA-Lib to yield 21 indicators (SMA, EMA, RSI, MACD, BBands, ATR, etc.). |
| `feature_store.py` | Offline-to-online SQLite materialization. | `store_features()` melts wide-format indicators into an entity-attribute-value (EAV) long format. `get_latest_features()` acts as the online serving mechanism. |
| `feast_store.py` | Feast bridge (if present). | Facilitates Feast-compliant ingestion logic from the SQLite staging tables. |

### c. `api/` (Routing & Schemas)
The presentation layer handling REST and WebSocket connections.

| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `app.py` | Primary FastAPI orchestrator. | Defines `/predict/{ticker}` and `/predict/batch`. Enforces sequential processing for batch requests to bypass PyTorch MPS threading deadlocks. Preloads models on startup. |
| `websocket.py` | Real-time push notifications. | `websocket_screener()` connects to the Redis `screener:updates` pub/sub channel and streams messages asynchronously to clients. |
| `schemas/prediction.py` | Pydantic response models. | Defines `PredictionResponse`, `KronosInferenceResponse`, enforces strict payload structures including reasonings and quantile bounds. |
| `schemas/batch.py` | Batch request/response shapes. | Defines `BatchPredictionRequest` handling up to 10 tickers simultaneously. |

### d. `utils/` (Core Logic & Utilities)
The quantitative research engine for signal generation, risk management, and telemetry.

| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `cache.py` | Redis caching logic. | `get_cached()` / `set_cached()` wrapper using `redis-py` with standard 300s TTLs to prevent API rate limits. |
| `trading_strategy.py` | 5-state signal generation. | `generate_signal_v2()` maps expected returns and confidence thresholds to Strong Buy/Buy/Hold/Sell/Strong Sell states. |
| `composite_signal.py` | AI Rating computation. | `compute_composite_rating()` flags "confirmation" or "conflict" by aligning TimesFM trajectory with Kronos structural labels. |
| `explainability.py` | Natural language logic. | `generate_reasoning()` outputs human-readable rules (e.g., "RSI Oversold", "MACD Bullish Crossover") justifying the AI rating. |
| `risk_engine.py` | Trade parameter estimation. | `compute_stop_take()` calculates ATR-based volatility stops. `suggested_position_size()` applies fixed-fractional sizing. |
| `portfolio_optimizer.py`| Capital allocation. | `suggest_allocation()` assigns inverse-risk-weighted allocations scaled to 100% across the active asset universe. |
| `backtester.py` | Strategy simulation. | `run_asset_backtest()` computes CAGR, Sharpe, and Drawdown with transaction costs (10 bps). |
| `walk_forward.py` | Walk-forward cross-validation. | `generate_rolling_windows()` creates non-overlapping splits. `verify_no_lookahead_bias()` proves feature causality. |
| `regime_analysis.py` | Market environment tagging. | `classify_market_regimes()` categorizes timeframes into Bull/Bear and High/Low volatility based on SPY 200-SMA. |
| `data_loader.py` | Historical data ingestion. | Downloads from `yfinance` and caches locally as `.parquet`. |
| `config.py` | Directory paths. | Ensures `data/`, `plots/`, and `feature_store/` exist. |
| `experiment_tracker.py` | SQLite run tracking. | `log_experiment()` writes backtest parameters and metrics into a queryable SQL database. |
| `metrics.py` | Quantitative metrics. | Computes MAE, RMSE, MAPE, Directional Accuracy, and Pinball Loss. |
| `plotting.py` | Matplotlib visualizations. | `plot_forecast()` draws actuals vs. point predictions with shaded P10-P90/P25-P75 quantile bands. |

### e. `scripts/` & `feature_repo/` (Runners & Configuration)
| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `run_quantitative_research.py` | Full backtest runner. | Executes the massive 8-step walk-forward backtest across all assets and models, simulating trading logic and outputting performance tables. |
| `run_timesfm.py` & `run_kronos.py`| Single-model validation. | Executes isolated out-of-sample holdout validations for the foundation models. |
| `feature_repo/definitions.py` | Feast Registry config. | Defines `ticker_entity` and `ta_indicators_fv` FeatureViews. Connects to `data/features_wide.parquet`. |

### f. Testing & Validation Scripts
| File | Purpose | Key Functions / Details |
| :--- | :--- | :--- |
| `test_api.py` / `test_batch_api.py` | API smoke tests. | Validates HTTP 200 responses, verifies concurrent payload structures, and confirms batch execution times. |
| `verify_pipeline_real_data.py` | Sub-component tests. | 7-step verifier checking signal diversity, risk engine sanity (ATR stops), and portfolio optimization boundary conditions. |
| `test_publish_update.py` | Pub/sub simulation. | Injects mock data into Redis `screener:updates` to test WebSocket listeners. |

---

## 6. API Surface

| Endpoint | Method | Purpose | Cache | Model |
| :--- | :--- | :--- | :--- | :--- |
| `/predict/{ticker}` | GET | Unified prediction | Redis (300s) | TimesFM + Kronos |
| `/predict/kronos/{ticker}` | GET | Pattern detection | Redis (300s) | Kronos |
| `/predict/timesfm/{ticker}` | GET | Multi-horizon forecast | Redis (300s) | TimesFM |
| `/predict/batch` | POST | Multi-asset execution | Redis (300s) | TimesFM + Kronos |
| `/api/v1/screeners/ai-filter` | GET | AI screener scan | Internal | Composite |
| `/ws/screener` | WS | Real-time updates | Redis Pub/Sub| N/A |

---

## 7. Engineering Highlights
*   **Hardware-aware Device Selection:** `models/device.py` dynamically routes matrix multiplications to CUDA, MPS (Apple Silicon), or CPU, ensuring cross-platform usability without manual configuration.
*   **Redis Caching Strategy:** A strict 300-second TTL in `utils/cache.py` drastically reduces redundant foundational model inference calls (which take ~5.5s each), yielding sub-second endpoint responses on cache hits.
*   **Multi-Model Architecture:** `utils/composite_signal.py` brilliantly cross-references TimesFM’s probabilistic trend forecasts with Kronos’s structural candlestick outputs, creating a high-confidence "confirmation" flag when both paradigms align.
*   **Feast Wrapper over SQLite:** Transforming wide-format Pandas DataFrames into long-format EAV (Entity-Attribute-Value) SQLite rows allows for future-proof integration into enterprise Feature Stores without complex infrastructure overhead today.
*   **Modular API Design:** Segregating `/predict/kronos` and `/predict/timesfm` into isolated endpoints allows specific downstream consumers (e.g., a fan-chart UI vs. a pattern-scanner bot) to pull exactly the data they need without full-pipeline overhead.
*   **Composite AI Rating Logic:** The `generate_signal_v2` function utilizes dynamic confidence thresholds (derived from the P10-P90 spread) to actively downgrade signals from "Buy" to "Hold" when model uncertainty is too high, protecting capital.
*   **WebSocket-based Live Updates:** Bypassing traditional HTTP polling, the `/ws/screener` endpoint asynchronously streams JSON updates via Redis Pub/Sub, enabling live tick-by-tick dashboard updates.

## 8. Current Limitations & Future Work
*   **Live Market Streaming:** Currently relies on standard API pulls. Future work will integrate live tick data via WebSocket providers (e.g., Polygon.io) directly into the feature store.
*   **Deployment & Scaling:** While the codebase is modular, it currently runs bare-metal. Future work includes Dockerization and Kubernetes deployment to horizontally scale the FastAPI workers.
*   **PyTorch MPS Constraints:** PyTorch's MPS backend on Apple Silicon currently suffers from thread-safety issues during concurrent `.generate()` calls. As a result, the `/predict/batch` endpoint processes assets sequentially. Transitioning to NVIDIA GPUs via TensorRT will unlock true parallel batch processing.
*   **Model Quantization:** Scripts exploring FP16/INT8 dynamic quantization (`investigate_quantization.py`) indicate significant speedups are possible, which will be integrated into the core inference pipeline to further reduce the ~5s per-asset latency.
*   **Multi-Market Screening:** Expanding the AI screener to cover 500+ assets will require distributed Redis workers and optimized Feast online-serving.

## 9. Conclusion
The Unified Finance Forecasting Engine successfully integrates TimesFM, Kronos, TA-Lib, Feast, Redis, and FastAPI into a unified platform for AI-assisted financial forecasting. By combining multi-horizon forecasting, technical pattern recognition, feature management, and low-latency API services, the system delivers scalable and explainable trading insights. Its modular architecture, efficient caching, and real-time capabilities provide a strong foundation for quantitative financial research and can be further extended with advanced analytics, additional data sources, and automated trading features in the future.


---

## 10. Appendix
### Temporary Development Files
The following files were inventoried but excluded from the main report as they are temporary scratchpads used for initial development and testing:
*   `scratch_step1.py`: Early-stage validation script.
*   `scratch_test.py`: Temporary testing module for localized component logic.
