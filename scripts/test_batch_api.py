import sys
import json
import time
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

print("Running batch smoke test...")
payload = {
    "tickers": ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD", "NOTAREALTICKER"]
}

with TestClient(app) as client:
    t0 = time.perf_counter()
    response = client.post("/predict/batch", json=payload)
    t1 = time.perf_counter()
    
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    results = data.get("results", [])
    failed = data.get("failed", [])
    duration_ms = data.get("duration_ms", 0.0)
    
    print("\n--- Batch Prediction Results ---")
    print(f"Successful Tickers: {len(results)}")
    print(f"Failed Tickers: {len(failed)}")
    print(f"Batch Request Duration: {duration_ms / 1000.0:.2f} seconds")
    
    print("\n--- Failures ---")
    for f in failed:
        print(f"Ticker: {f['ticker']}, Error: {f['error']}")
        
    print("\n--- Reasoning Lists ---")
    has_kronos_pattern = False
    for res in results:
        ticker = res["ticker"]
        reasoning = res.get("reasoning", [])
        print(f"[{ticker}] Reasoning:")
        for r in reasoning:
            print(f"  - {r}")
            if "Kronos trend confirmation" in r or "Kronos pattern conflicts" in r:
                has_kronos_pattern = True
                
    print("\n--- Summary ---")
    if has_kronos_pattern:
        print("YES! Found at least one Kronos confirmation/conflict line in real data.")
    else:
        print("NO Kronos pattern lines found across all 5 tickers. It's a quiet week.")
        
    # Print timing comparison
    # From earlier tests, single ticker takes ~5-6 seconds.
    est_sequential = 6 * 5.5
    actual = duration_ms / 1000.0
    print(f"\nTiming Comparison:")
    print(f"Estimated Sequential (6 calls): ~{est_sequential:.1f}s")
    print(f"Actual Concurrent (6 calls): {actual:.1f}s")
    print(f"Speedup Factor: {est_sequential / actual:.2f}x")
else:
    print("Error:", response.text)
