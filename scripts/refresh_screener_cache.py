"""
scripts/refresh_screener_cache.py
---------------------------------
Background Pre-computation Worker for AI Screener.

Phase 4 Deliverable:
- Loads asset universe from config/universe.yaml.
- Pre-computes Kronos pattern detection + TimesFM forecasts + composite ratings.
- Writes latest scan results to Redis key 'screener:latest_results'.
- Publishes real-time event updates to Redis Pub/Sub channel 'screener:updates'.
- Enables /api/v1/screeners/ai-filter to achieve sub-200ms responses (meeting <2s SLA).
"""

import os
import sys
import time
import json
import yaml
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import redis
from utils.cache import r as redis_client, set_screener_cache
from utils.composite_signal import compute_composite_rating


def load_universe(universe_path: str = None) -> List[str]:
    """Loads tickers from config/universe.yaml."""
    if not universe_path:
        universe_path = os.path.join(PROJECT_ROOT, "config", "universe.yaml")

    default_tickers = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD"]
    if not os.path.exists(universe_path):
        print(f"[Warning] Universe config not found at {universe_path}. Using fallback tickers: {default_tickers}")
        return default_tickers

    try:
        with open(universe_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tickers = []
        if isinstance(data, dict):
            for group, t_list in data.items():
                if isinstance(t_list, list):
                    tickers.extend(t_list)
        elif isinstance(data, list):
            tickers = data
        
        # Deduplicate while preserving order
        seen = set()
        unique_tickers = [t for t in tickers if not (t in seen or seen.add(t))]
        return unique_tickers or default_tickers
    except Exception as e:
        print(f"[Error] Failed parsing universe.yaml: {e}. Falling back to default list.")
        return default_tickers


def refresh_screener(tickers: List[str], api_base_url: str = None) -> List[Dict[str, Any]]:
    """
    Computes screener results for the given ticker universe.
    Can query the running API or directly invoke local model functions.
    """
    import requests
    base_url = api_base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"\n[{datetime.utcnow().isoformat()}] Starting screener pre-computation for {len(tickers)} tickers...")
    
    results = []
    start_time = time.perf_counter()

    for idx, t in enumerate(tickers, 1):
        t_start = time.perf_counter()
        try:
            # Query Kronos endpoint
            k_resp = requests.get(f"{base_url}/predict/kronos/{t}", timeout=60)
            if k_resp.status_code != 200:
                print(f"  [{idx}/{len(tickers)}] {t}: Kronos error ({k_resp.status_code})")
                continue
            k_data = k_resp.json()

            # Query TimesFM endpoint
            tfm_resp = requests.get(f"{base_url}/predict/timesfm/{t}?horizons=30", timeout=60)
            if tfm_resp.status_code != 200:
                print(f"  [{idx}/{len(tickers)}] {t}: TimesFM error ({tfm_resp.status_code})")
                continue
            tfm_data = tfm_resp.json()

            composite = compute_composite_rating(k_data, tfm_data)
            
            entry = {
                "ticker": t,
                "kronos": {
                    "pattern": k_data.get("pattern", "CDLENGULFING"),
                    "label": k_data.get("label", "No Pattern"),
                    "confidence": float(k_data.get("confidence", 0.0)),
                    "forecast_step": k_data.get("forecast_step"),
                    "attention_heatmap": k_data.get("attention_heatmap")
                },
                "timesfm_summary": {
                    "direction": composite.get("timesfm_direction"),
                    "prob_plus_5": composite.get("timesfm_prob_plus_5")
                },
                "composite_rating": composite,
                "cached_at": datetime.utcnow().isoformat() + "Z"
            }
            results.append(entry)
            elapsed = time.perf_counter() - t_start
            print(f"  [{idx}/{len(tickers)}] {t}: SUCCESS ({elapsed:.2f}s) | Pattern: {entry['kronos']['label']} | Rating: {composite.get('rating')}")

        except Exception as e:
            print(f"  [{idx}/{len(tickers)}] {t}: FAILED with error: {e}")

    total_duration = time.perf_counter() - start_time
    print(f"\nScreener pre-computation finished in {total_duration:.2f}s. Total processed: {len(results)}/{len(tickers)}")

    # 1. Save results to Redis Key
    set_screener_cache(results, ttl_seconds=900)
    print(f"[Redis Cache] Saved {len(results)} items to 'screener:latest_results' with 900s TTL.")

    # 2. Publish structured event to Redis Pub/Sub channel
    top_matches = [
        {
            "ticker": r["ticker"],
            "composite_rating": r["composite_rating"].get("rating"),
            "confidence": r["kronos"]["confidence"]
        }
        for r in sorted(results, key=lambda x: x["kronos"]["confidence"], reverse=True)[:5]
    ]

    event_payload = {
        "event": "screener_refresh",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_scanned": len(tickers),
        "total_matched": len(results),
        "duration_seconds": round(total_duration, 2),
        "top_matches": top_matches
    }

    try:
        redis_client.publish("screener:updates", json.dumps(event_payload))
        print(f"[Redis Pub/Sub] Published 'screener_refresh' event to channel 'screener:updates'.")
    except Exception as e:
        print(f"[Warning] Failed publishing update to Redis pubsub: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Refresh screener pre-computed cache in Redis.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers from universe")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a background loop")
    parser.add_argument("--interval", type=int, default=300, help="Loop interval in seconds (default 300s)")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    tickers = load_universe()
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"Loaded {len(tickers)} tickers from universe: {tickers[:10]}{'...' if len(tickers)>10 else ''}")

    if args.loop:
        print(f"Starting continuous pre-computation loop every {args.interval}s. Press Ctrl+C to stop.")
        while True:
            try:
                refresh_screener(tickers, api_base_url=args.api_url)
                print(f"Sleeping for {args.interval}s until next refresh...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"[Error in loop] {e}. Retrying in 10 seconds...")
                time.sleep(10)
    else:
        refresh_screener(tickers, api_base_url=args.api_url)


if __name__ == "__main__":
    main()
