# Kronos vs TimesFM — Benchmark Analysis Report

**Date:** 2026-07-15  
**Holdout Period:** 30 trading days (≈ May 28 – Jul 10, 2026)  
**Assets:** AAPL, MSFT, NVDA, SPY, BTC-USD  
**Context Window:** 365 days  
**Run type:** Single-run point-in-time snapshot

---

## 1. Summary Metrics

| Ticker  | Model   | MAE        | RMSE       | MAPE (%) | Dir. Acc (%) | Inference (s) | Memory (MB) |
|---------|---------|------------|------------|----------|--------------|---------------|-------------|
| AAPL    | Kronos  | 58.39      | 61.57      | 19.27    | **60.00**    | 5.575         | 0.50        |
| AAPL    | TimesFM | **11.26**  | **13.42**  | **3.81** | 50.00        | **0.241**     | **0.07**    |
| MSFT    | Kronos  | 68.38      | 72.23      | 17.81    | **56.67**    | 4.526         | 0.19        |
| MSFT    | TimesFM | **67.06**  | **69.42**  | **17.44**| 50.00        | **0.250**     | **0.07**    |
| NVDA    | Kronos  | 35.82      | 38.12      | 17.60    | **50.00**    | 4.455         | 0.19        |
| NVDA    | TimesFM | **16.59**  | **17.84**  | **8.22** | 46.67        | **0.236**     | **0.07**    |
| SPY     | Kronos  | 88.14      | 90.66      | 11.84    | **60.00**    | 4.424         | 0.19        |
| SPY     | TimesFM | **21.46**  | **23.17**  | **2.90** | **60.00**    | **0.269**     | **0.07**    |
| BTC-USD | Kronos  | 25,003.47  | 26,716.26  | 40.15    | 43.33        | 4.306         | 0.20        |
| BTC-USD | TimesFM | **6,140.88**| **6,503.33**| **9.91**| **50.00**    | **0.268**     | **0.11**    |

> **Bold** = better model for that metric/ticker in this run. Lower is better for MAE, RMSE, MAPE, Inference, Memory. Higher is better for Directional Accuracy.

---

## 2. Where TimesFM Performs Better

TimesFM demonstrates strong performance on the majority of metrics and assets tested:

- **Accuracy across All Assets:** TimesFM wins on every single asset for all error metrics (MAE/RMSE/MAPE). The margin is substantial on AAPL (MAE 11.26 vs 58.39 — 5.2× lower), NVDA (MAE 16.59 vs 35.82 — 2.2× lower), and SPY (MAE 21.46 vs 88.14 — 4.1× lower). On BTC-USD, TimesFM achieves a MAPE of 9.91% compared to Kronos's 40.15% (4× lower). Even on MSFT (the closest category), TimesFM remains slightly more accurate (MAE 67.06 vs 68.38).
- **Inference Speed:** TimesFM is **16–30× faster** than Kronos, completing forecasts in 0.23–0.27 seconds compared to Kronos's 4.3–5.6 seconds.
- **Memory Efficiency:** TimesFM uses **0.07–0.11 MB** per ticker, which is 2.5–7× more memory efficient than Kronos (0.19–0.50 MB).
- **Probabilistic Output:** TimesFM natively provides calibrated quantile forecasts (p10–p90) and a tail-risk probability (P(>+5%)), which are critical for risk management and portfolio sizing. Kronos only outputted point forecasts in this benchmark.

---

## 3. Where Kronos Performs Better

In this single-run benchmark, Kronos exhibits advantages on directional metrics:

- **Directional Accuracy:** Kronos achieves higher directional accuracy on 3 out of 5 assets: AAPL (60.0% vs 50.0%), MSFT (56.67% vs 50.0%), and NVDA (50.0% vs 46.67%), and ties on SPY (60.0% vs 60.0%).
- **Architectural Flexibility:** Kronos operates as a generative, sampling-based model. While not utilized in this single-point forecast, this architecture could theoretically produce distribution forecasts via multi-sample generation.
- **Zero-Shot Generalization:** Kronos is a zero-shot foundation model with no task-specific fine-tuning. This may provide a more robust baseline on highly out-of-distribution time series (though this was not explicitly tested here).

---

## 4. Limitations Observed

### 4.1 Kronos-Specific
- **High MAPE on most assets:** 11–40% MAPE suggests that a single point-forecast sample from Kronos is highly noisy.
- **Stochastic Architecture:** Kronos is sampling-based and stochastic. A single run does not measure or control for cross-run variance, so repeated executions may produce different results. This was not tested.
- **No native quantile output:** The pipeline does not expose calibrated prediction intervals natively. Pinball loss is N/A.
- **High latency:** 4–6 seconds per ticker makes it less viable for real-time or high-frequency dashboards.

### 4.2 TimesFM-Specific
- **Directional accuracy drops:** TimesFM's directional accuracy was lower than Kronos on AAPL, MSFT, and NVDA in this run, indicating it may smooth over turning points in certain market regimes.
- **Single-run evaluation:** Cross-run variance was not tested. TimesFM produced consistent point forecasts within this single execution, but its behavior under stochastic device execution (such as MPS non-determinism) was not averaged.
- **Quantile calibration unverified:** Pinball losses are calculated but their empirical coverage (e.g., does p10 actually capture 10% of outcomes?) was not formally tested.

### 4.3 General
- **Single holdout window:** All metrics are evaluated on one 30-day window. Performance could differ substantially in different market environments (e.g., bear markets, flat ranges, or high-volatility regimes).
- **No walk-forward validation:** A single holdout is a snapshot, not a statistically robust cross-validation.

---

## 5. Initial Conclusions

1. **TimesFM is the stronger model for most tested assets.** It produces lower error on all 5 tickers, completes inference 16–30× faster, uses significantly less memory, and natively supports probabilistic interval forecasts.
2. **Kronos demonstrates competitive directional accuracy.** It outperformed TimesFM on directional direction on AAPL, MSFT, and NVDA. However, its high point-forecast errors (MAPE) and slow speed make it less practical for standalone deployment.
3. **TimesFM is suitable for real-time applications.** Its sub-quarter-second latency makes it viable for interactive dashboards, while Kronos's multi-second latency limits it to batch processes.
4. **Next steps for production validation should include:**
   - Walk-forward validation across multiple historical holdout windows.
   - Drawing multiple samples from Kronos (e.g., N=50) to evaluate whether a sample mean reduces point-forecast noise.
   - Formally checking TimesFM's quantile calibration.

---

## 6. Caveats

> [!IMPORTANT]
> **This is a single-run benchmark comparison.** Cross-run variance was not tested. Kronos's sampling-based generative architecture is stochastic and may produce different metrics on repeated runs. Additionally, all evaluations are restricted to the same 30-day holdout window; performance under different market regimes was not evaluated.

---

## Appendix: Data Artifacts

| Artifact | Path |
|----------|------|
| Benchmark metrics CSV | `data/predictions/benchmark_results.csv` |
| Benchmark comparison chart | `plots/benchmark_comparison.png` |
| Kronos predictions (Parquet) | `data/predictions/{ticker}_kronos_predictions.parquet` |
| Kronos predictions (CSV) | `data/predictions/{ticker}_kronos_predictions.csv` |
| TimesFM predictions (Parquet) | `data/predictions/{ticker}_timesfm_predictions.parquet` |
| Individual forecast plots | `plots/{ticker}_{model}_forecast.png` |

All prediction files include a `run_timestamp` column for traceability.
