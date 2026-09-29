# Production Deployment & Operations Guide

Unified Financial Forecasting & High-Performance Inference Platform  
**Target Environments:** Docker Compose, Kubernetes, AWS ECS, GCP Cloud Run  
**Author:** ML Platform Engineering Team  
**Date:** September 2026

---

## 1. System Requirements

### Hardware Prerequisites
* **CPU:** 4+ Cores (x86_64).
* **RAM:** 8 GB minimum (16 GB recommended for large context windows).
* **GPU (Optional but Recommended):** NVIDIA Turing or newer (RTX 3090, A10, T4, A100) with 8+ GB VRAM.
* **Storage:** 20 GB SSD space (for model weights and historical Parquet cache).

### Software Prerequisites
* Docker Engine 24.0+ & Docker Compose v2.20+.
* NVIDIA Container Toolkit (if deploying with GPU acceleration).
* Python 3.11 (if deploying bare-metal).
* Redis 7.0+ (standalone, clustered, or managed).

---

## 2. Docker Compose Deployment (Recommended for Single-Node)

The repository provides a production-hardened `docker-compose.yml` orchestrating the FastAPI inference engine, Redis cache layer, and asynchronous screener worker.

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/shravani-gangan/finance-forecasting.git
cd finance-forecasting

# Optional: configure GCP project for BigQuery fallback
cp .env.example .env 2>/dev/null || :
```

### Step 2: Build and Launch Containers
```bash
# Build multi-stage Docker image and start services in detached mode
docker compose up -d --build
```

### Step 3: Verify Container Health
```bash
docker compose ps
```
All three containers should report `healthy`:
* `finance_redis` (healthy on port `6379`)
* `finance_api` (healthy on port `8000`)
* `finance_screener_worker` (background pre-computation daemon)

### Step 4: Monitor Production Logs
```bash
# View structured JSON logs in real time
docker compose logs -f api
docker compose logs -f screener-worker
```

---

## 3. Bare-Metal / Local Production Deployment

If deploying directly onto bare-metal Linux servers:

### 1. Install TA-Lib C Library
```bash
sudo apt-get update && sudo apt-get install -y build-essential wget
wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download -O ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make -j$(nproc)
sudo make install
cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

### 2. Set Up Virtual Environment
```bash
python3.11 -m venv fin-env
source fin-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Services via Systemd or Supervisord
```bash
# 1. Start Redis
sudo systemctl start redis

# 2. Launch FastAPI API server (Workers: 1 per GPU node to prevent MPS deadlocks)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Launch Screener background cache worker
python scripts/refresh_screener_cache.py --daemon --interval 300
```

---

## 4. Kubernetes Production Deployment

For enterprise high-availability Kubernetes deployments (EKS, GKE, AKS):

### Recommended Kubernetes Architecture
* **Inference API Deployment:** Horizontally autoscaled based on `inference_latency_seconds` and CPU utilization.
* **Worker Deployment:** Singleton (`replicas: 1`) background daemon populating the shared Redis cluster.
* **Redis:** Managed Redis (AWS ElastiCache, GCP Memorystore, or Redis Operator with Sentinel).

### Sample API Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finance-forecasting-api
  namespace: ml-inference
  labels:
    app: finance-forecasting
spec:
  replicas: 2
  selector:
    matchLabels:
      app: finance-forecasting
  template:
    metadata:
      labels:
        app: finance-forecasting
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: api
        image: <your-registry>/finance-forecasting:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-cluster.cache.amazonaws.com"
        - name: REDIS_PORT
          value: "6379"
        - name: TIMESFM_MAX_CONTEXT
          value: "4096"
        resources:
          requests:
            cpu: "2000m"
            memory: "4Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
            nvidia.com/gpu: "1"  # Optional: For GPU acceleration
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 45
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## 5. Observability Setup (Prometheus & Grafana)

The platform includes full telemetry instrumentation via `/metrics` and a pre-configured Grafana dashboard (`observability/grafana_dashboard.json`).

### 1. Prometheus Configuration (`prometheus.yml`)
Add the finance forecasting scrape job:
```yaml
scrape_configs:
  - job_name: 'finance-forecasting'
    scrape_interval: 15s
    static_configs:
      - targets: ['api:8000']
```

### 2. Grafana Dashboard Import
1. Open Grafana UI (`http://localhost:3000`).
2. Navigate to **Dashboards** -> **Import**.
3. Upload `observability/grafana_dashboard.json` from the repository.
4. Select your Prometheus datasource and click **Import**.

The dashboard renders:
* **Inference Request Rate (RPS)** categorized by HTTP status code.
* **Inference Latency Percentiles (p50, p95, p99)** per model (TimesFM, Kronos, Fallback).
* **Redis Cache Hit Ratio (%)**.
* **AI Screener Match Counter**.
* **Dependency Health Status**.

---

## 6. Operational Runbook & Incident Troubleshooting

| Symptom | Probable Cause | Action |
|---|---|---|
| **Health endpoint reports Redis down** | Network partition, Redis crash, or port blocked | The API automatically degrades to in-memory caching with a 10s cooldown. Check Redis logs via `docker compose logs redis` and restart container. |
| **TimesFM inference falls back to Statistical/BigQuery** | CUDA Out-Of-Memory (OOM) or Circuit Breaker tripped | Check `docker compose logs api`. Ensure GPU has sufficient VRAM. The system handles fallback gracefully; breaker auto-recovers after 60s cooldown. |
| **Yahoo Finance rate limits (HTTP 429)** | High request frequency downloading uncached historical data | The circuit breaker (`yfinance_breaker`) trips OPEN and routes to cached Parquet files in `data/raw/`. Increase background refresh interval if needed. |
| **Screener endpoint latency >2s** | Redis `screener:latest_results` expired or worker stopped | Verify `finance_screener_worker` is active: `docker compose restart screener-worker`. The worker will regenerate the cache within 15 seconds. |
