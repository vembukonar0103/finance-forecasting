"""
scripts/profile_latency.py
--------------------------
Quantitative Latency, Cache Performance, and Memory Profiler for Finance Forecasting Engine.

Phase 1 (Week 1, Day 1) Deliverable:
- Measures p50, p95, p99 latency across core API endpoints (10 iterations each).
- Evaluates Cold-Start (cache miss) vs Warm-Start (cache hit) latencies.
- Measures memory footprint per inference using tracemalloc (and torch.cuda if available).
- Hooks cProfile to predict_kronos_ticker() and predict_timesfm_ticker() to identify bottlenecks.
- Generates structured console tables and exports reports to reports/latency_baseline.json.
"""

import sys
import os
import time
import json
import argparse
import statistics
import tracemalloc
import cProfile
import pstats
import io
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import redis
import torch

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TICKER = "AAPL"
DEFAULT_ITERATIONS = 10


def get_redis_client():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        print(f"[Warning] Could not connect to Redis at localhost:6379: {e}")
        return None


def clear_cache_for_ticker(r: Optional[redis.Redis], ticker: str):
    """Deletes cached keys for a ticker to ensure clean cold-start testing."""
    if not r:
        return
    try:
        keys = r.keys(f"kronos:{ticker}:*") + r.keys(f"timesfm:{ticker}:*") + r.keys(f"indicators:{ticker}:*")
        if keys:
            r.delete(*keys)
            print(f"  [Cache Cleared] Flushed {len(keys)} test keys for ticker '{ticker}' from Redis.")
    except Exception as e:
        print(f"  [Cache Warning] Failed to clear keys for {ticker}: {e}")


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculates min, max, mean, p50, p95, and p99 from a list of latencies in ms."""
    if not latencies:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
    
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    
    def percentile(p: float) -> float:
        idx = (n - 1) * p
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_lats[lower] * (1 - weight) + sorted_lats[upper] * weight

    return {
        "min": round(sorted_lats[0], 2),
        "max": round(sorted_lats[-1], 2),
        "mean": round(statistics.mean(latencies), 2),
        "p50": round(percentile(0.50), 2),
        "p95": round(percentile(0.95), 2),
        "p99": round(percentile(0.99), 2),
    }


def profile_endpoint_http(
    base_url: str,
    method: str,
    path: str,
    name: str,
    iterations: int = 10,
    r: Optional[redis.Redis] = None,
    ticker: str = "AAPL",
    clear_cache_before_start: bool = True
) -> Dict[str, Any]:
    """
    Profiles an endpoint over HTTP:
    - Iteration 1 is cold-start (with cache cleared if specified).
    - Iterations 2..N are warm-starts (leveraging cache when available).
    """
    url = f"{base_url}{path}"
    print(f"\nProfiling [{name}] -> {url} ({iterations} iterations)...")
    
    if clear_cache_before_start:
        clear_cache_for_ticker(r, ticker)

    latencies_ms: List[float] = []
    cache_hits: List[bool] = []
    status_codes: List[int] = []

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        try:
            if method.upper() == "GET":
                resp = requests.get(url, timeout=120)
            elif method.upper() == "POST":
                resp = requests.post(url, timeout=120)
            else:
                raise ValueError(f"Unsupported method: {method}")
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            status_codes.append(resp.status_code)

            # Check if response payload indicates cache hit
            is_hit = False
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        if payload.get("cache_hit") is True:
                            is_hit = True
                        elif "horizons" in payload:
                            # TimesFM fan chart: check any horizon hit
                            for h_data in payload["horizons"].values():
                                if h_data.get("cache_hit") is True:
                                    is_hit = True
                                    break
                except Exception:
                    pass
            cache_hits.append(is_hit)

            tag = "Cold Start" if i == 1 else ("Warm Hit" if is_hit else "Warm Miss")
            print(f"  Req {i:02d}: {elapsed_ms:8.2f} ms | Status: {resp.status_code} | [{tag}]")

        except Exception as e:
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            status_codes.append(500)
            cache_hits.append(False)
            print(f"  Req {i:02d}: FAILED ({elapsed_ms:8.2f} ms) | Error: {e}")

    stats = calculate_percentiles(latencies_ms)
    cold_start_ms = round(latencies_ms[0], 2) if latencies_ms else 0.0
    warm_latencies = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
    warm_stats = calculate_percentiles(warm_latencies)

    return {
        "name": name,
        "endpoint": path,
        "method": method,
        "iterations": iterations,
        "status_success_rate": round(sum(1 for s in status_codes if s == 200) / max(1, len(status_codes)) * 100, 1),
        "cold_start_ms": cold_start_ms,
        "warm_mean_ms": warm_stats["mean"],
        "warm_p50_ms": warm_stats["p50"],
        "warm_p95_ms": warm_stats["p95"],
        "p50_ms": stats["p50"],
        "p95_ms": stats["p95"],
        "p99_ms": stats["p99"],
        "min_ms": stats["min"],
        "max_ms": stats["max"],
        "mean_ms": stats["mean"],
        "latencies_ms": [round(x, 2) for x in latencies_ms],
        "cache_hits": cache_hits
    }


def profile_functions_in_process(ticker: str = "AAPL") -> Dict[str, Any]:
    """
    Uses cProfile and tracemalloc directly on predict_kronos_ticker() and predict_timesfm_ticker()
    to measure memory footprint and isolate function bottlenecks.
    """
    print("\n" + "=" * 80)
    print(f"Deep In-Process Profiling (cProfile & tracemalloc) for Ticker: {ticker}")
    print("=" * 80)

    try:
        from api.app import predict_kronos_ticker, predict_timesfm_ticker, startup_load_models
        # Ensure models loaded
        startup_load_models()
    except Exception as e:
        print(f"[Error] Failed to import/initialize models for in-process profiling: {e}")
        return {}

    results = {}

    # -------------------------------------------------------------
    # 1. Profile predict_kronos_ticker()
    # -------------------------------------------------------------
    print(f"\n--- Profiling predict_kronos_ticker('{ticker}') ---")
    tracemalloc.start()
    tracemalloc.reset_peak()
    cuda_mem_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    profiler_k = cProfile.Profile()
    profiler_k.enable()
    t0 = time.perf_counter()
    k_res = predict_kronos_ticker(ticker=ticker, timeframe="1d")
    t1 = time.perf_counter()
    profiler_k.disable()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    s_k = io.StringIO()
    ps_k = pstats.Stats(profiler_k, stream=s_k).sort_stats("cumulative")
    ps_k.print_stats(15)  # Top 15 bottlenecks
    bottleneck_k = s_k.getvalue()

    k_profile_data = {
        "duration_ms": round((t1 - t0) * 1000.0, 2),
        "peak_ram_mb": round(peak_mem / (1024 * 1024), 2),
        "current_ram_mb": round(current_mem / (1024 * 1024), 2),
        "cuda_peak_mb": round(cuda_peak / (1024 * 1024), 2) if torch.cuda.is_available() else None,
        "tokenization_ms": getattr(k_res, "tokenization_ms", None),
        "top_bottlenecks": "\n".join(bottleneck_k.strip().split("\n")[:20])
    }
    results["kronos_function"] = k_profile_data
    print(f"Duration: {k_profile_data['duration_ms']} ms | Peak RAM: {k_profile_data['peak_ram_mb']} MB")
    print("\nTop 15 Cumulative Time Bottlenecks (Kronos):")
    print(k_profile_data["top_bottlenecks"])

    # -------------------------------------------------------------
    # 2. Profile predict_timesfm_ticker()
    # -------------------------------------------------------------
    print(f"\n--- Profiling predict_timesfm_ticker('{ticker}') ---")
    tracemalloc.start()
    tracemalloc.reset_peak()

    profiler_t = cProfile.Profile()
    profiler_t.enable()
    t0 = time.perf_counter()
    t_res = predict_timesfm_ticker(ticker=ticker, horizons="30")
    t1 = time.perf_counter()
    profiler_t.disable()

    current_mem_t, peak_mem_t = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_peak_t = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    s_t = io.StringIO()
    ps_t = pstats.Stats(profiler_t, stream=s_t).sort_stats("cumulative")
    ps_t.print_stats(15)
    bottleneck_t = s_t.getvalue()

    t_profile_data = {
        "duration_ms": round((t1 - t0) * 1000.0, 2),
        "peak_ram_mb": round(peak_mem_t / (1024 * 1024), 2),
        "current_ram_mb": round(current_mem_t / (1024 * 1024), 2),
        "cuda_peak_mb": round(cuda_peak_t / (1024 * 1024), 2) if torch.cuda.is_available() else None,
        "top_bottlenecks": "\n".join(bottleneck_t.strip().split("\n")[:20])
    }
    results["timesfm_function"] = t_profile_data
    print(f"Duration: {t_profile_data['duration_ms']} ms | Peak RAM: {t_profile_data['peak_ram_mb']} MB")
    print("\nTop 15 Cumulative Time Bottlenecks (TimesFM):")
    print(t_profile_data["top_bottlenecks"])

    return results


def print_summary_table(endpoint_results: List[Dict[str, Any]]):
    """Prints a clean ASCII summary table of the benchmark results."""
    print("\n" + "=" * 105)
    print(" " * 32 + "WEEK 1 BASELINE LATENCY PROFILING REPORT")
    print("=" * 105)
    header = (
        f"{'Endpoint Name':<28} | {'Cold (ms)':<10} | {'Warm Mean':<10} | {'p50 (ms)':<9} | "
        f"{'p95 (ms)':<9} | {'p99 (ms)':<9} | {'Min (ms)':<8} | {'Max (ms)':<8}"
    )
    print(header)
    print("-" * 105)
    for res in endpoint_results:
        print(
            f"{res['name']:<28} | "
            f"{res['cold_start_ms']:<10.1f} | "
            f"{res['warm_mean_ms']:<10.1f} | "
            f"{res['p50_ms']:<9.1f} | "
            f"{res['p95_ms']:<9.1f} | "
            f"{res['p99_ms']:<9.1f} | "
            f"{res['min_ms']:<8.1f} | "
            f"{res['max_ms']:<8.1f}"
        )
    print("=" * 105)


def export_markdown_report(
    endpoint_results: List[Dict[str, Any]],
    function_results: Dict[str, Any],
    out_path: str
):
    """Saves formatted Markdown report for Phase 1 documentation."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Benchmark & Latency Profiling Report (Week 1)\n\n")
        f.write(f"**Generated Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Hardware Backend:** {'CUDA' if torch.cuda.is_available() else ('MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU')}\n\n")
        f.write("## 1. HTTP Endpoint Latency Benchmark (10 Iterations per Endpoint)\n\n")
        f.write("| Endpoint Name | Method | Path | Cold-Start (ms) | Warm Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Cache Hit Benefit |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for res in endpoint_results:
            cold = res['cold_start_ms']
            warm = res['warm_mean_ms']
            speedup = f"{round(cold / max(0.1, warm), 1)}x" if warm > 0 else "N/A"
            f.write(f"| **{res['name']}** | `{res['method']}` | `{res['endpoint']}` | {cold:.1f} | {warm:.1f} | {res['p50_ms']:.1f} | {res['p95_ms']:.1f} | {res['p99_ms']:.1f} | {speedup} |\n")

        f.write("\n## 2. Key Takeaways & Bottlenecks\n\n")
        for res in endpoint_results:
            if "Screener" in res["name"]:
                f.write(f"- **Screener SLA (<2.0s):** Current response is **{res['cold_start_ms'] / 1000.0:.2f}s** cold / **{res['warm_mean_ms'] / 1000.0:.2f}s** warm. {'❌ FAILS SLA' if res['cold_start_ms'] > 2000 else '✅ MEETS SLA'}\n")
            elif "Unified" in res["name"]:
                f.write(f"- **Unified Prediction (/predict):** Takes **{res['cold_start_ms']:.1f}ms** without request-level caching. TimesFM + Kronos run sequentially on CPU.\n")

        if function_results:
            f.write("\n## 3. In-Process Memory Footprint & Bottleneck Analysis\n\n")
            if "kronos_function" in function_results:
                k = function_results["kronos_function"]
                f.write(f"### Kronos Function (`predict_kronos_ticker`)\n")
                f.write(f"- **Total Duration:** {k['duration_ms']} ms\n")
                f.write(f"- **Peak RAM Footprint:** {k['peak_ram_mb']} MB\n")
                if k.get("tokenization_ms"):
                    f.write(f"- **BSQ Tokenization Time:** {k['tokenization_ms']} ms\n")
            if "timesfm_function" in function_results:
                t = function_results["timesfm_function"]
                f.write(f"\n### TimesFM Function (`predict_timesfm_ticker`)\n")
                f.write(f"- **Total Duration:** {t['duration_ms']} ms\n")
                f.write(f"- **Peak RAM Footprint:** {t['peak_ram_mb']} MB\n")

    print(f"\n[Report Exported] Saved Markdown benchmark to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile latency and memory footprint of the forecasting platform.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL of running API server")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Ticker to test (default AAPL)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Number of benchmark iterations (default 10)")
    parser.add_argument("--endpoint", type=str, default="all", choices=["all", "health", "timesfm", "kronos", "unified", "screener"], help="Specific endpoint to profile or 'all'")
    parser.add_argument("--profile-functions", action="store_true", help="Run deep in-process cProfile and memory inspection (loads models locally)")
    args = parser.parse_args()

    r = get_redis_client()

    print("=" * 80)
    print(" STARTING PHASE 1 LATENCY & PERFORMANCE PROFILING")
    print(f" Target API Base: {args.base_url}")
    print(f" Target Ticker:   {args.ticker}")
    print(f" Iterations:      {args.iterations}")
    print(f" Endpoint Scope:  {args.endpoint}")
    print("=" * 80)

    # 1. Health check to ensure server is ready
    try:
        h = requests.get(f"{args.base_url}/health", timeout=5)
        if h.status_code != 200:
            print(f"[Error] API /health returned status {h.status_code}. Server may not be ready.")
            sys.exit(1)
        print(f"[API Status] Health Check: {h.json()}")
    except Exception as e:
        print(f"[Fatal Error] Cannot connect to API server at {args.base_url}: {e}")
        print("Please ensure the FastAPI server is running: python -m uvicorn api.app:app --port 8000")
        sys.exit(1)

    # Endpoints to profile
    all_endpoints = [
        ("health", "GET", "/health", "Health Check", False),
        ("timesfm", "GET", f"/predict/timesfm/{args.ticker}?horizons=30", "TimesFM Fan Chart (30d)", True),
        ("kronos", "GET", f"/predict/kronos/{args.ticker}", "Kronos Pattern Detection", True),
        ("unified", "GET", f"/predict/{args.ticker}", "Unified Prediction (/predict)", False),
        ("screener", "GET", "/api/v1/screeners/ai-filter", "AI Screener (5 Tickers)", False)
    ]

    endpoints = [
        (method, path, name, clear_cache)
        for key, method, path, name, clear_cache in all_endpoints
        if args.endpoint == "all" or args.endpoint == key
    ]

    endpoint_results = []
    for method, path, name, clear_cache in endpoints:
        res = profile_endpoint_http(
            base_url=args.base_url,
            method=method,
            path=path,
            name=name,
            iterations=args.iterations,
            r=r,
            ticker=args.ticker,
            clear_cache_before_start=clear_cache
        )
        endpoint_results.append(res)

    print_summary_table(endpoint_results)

    # Function-level profiling with cProfile & tracemalloc (if enabled)
    function_results = {}
    if args.profile_functions:
        function_results = profile_functions_in_process(ticker=args.ticker)

    # Export structured JSON data
    output_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "latency_baseline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "ticker": args.ticker,
            "endpoints": endpoint_results,
            "functions": {k: {k2: v2 for k2, v2 in v.items() if k2 != "top_bottlenecks"} for k, v in function_results.items()}
        }, f, indent=2)
    print(f"\n[Data Exported] Saved raw metrics to: {json_path}")

    # Export docs/benchmark_report_week1.md
    docs_path = os.path.join(PROJECT_ROOT, "docs", "benchmark_report_week1.md")
    export_markdown_report(endpoint_results, function_results, docs_path)


if __name__ == "__main__":
    main()
