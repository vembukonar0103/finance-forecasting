import sys
import json
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

print("Running smoke test for /predict/AAPL...")
with TestClient(app) as client:
    response = client.get("/predict/AAPL")
    
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("Response JSON Keys:", data.keys())
    print("\n--- Rich Prediction Payload (Partial) ---")
    print(f"Ticker: {data.get('ticker')}")
    print(f"Model: {data.get('model_used')}")
    print(f"Signal: {data.get('signal')}")
    print(f"Expected Return: {data.get('expected_return_pct')}")
    print(f"Confidence Score: {data.get('confidence_score')}")
    print("Reasoning:")
    for r in data.get('reasoning', []):
        print(f"  - {r}")
    print("-----------------------------------------")
else:
    print("Error:", response.text)
