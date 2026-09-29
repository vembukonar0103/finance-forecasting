# ==============================================================================
# Multi-Stage Production Dockerfile for Unified Finance Forecasting Platform
# ==============================================================================
# Stage 1: Build C dependencies (TA-Lib) and wheels
# Stage 2: Minimal hardened runtime image (<1GB target)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download, compile, and install TA-Lib 0.4.0 C Library (single-threaded -j1 avoids gen_code race condition)
RUN wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download -O ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install Python requirements into user site directory
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --user -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install minimal runtime dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled TA-Lib C shared libraries and headers from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta_lib* /usr/include/

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure local packages and TA-Lib C library are in linker/executable search paths
ENV PATH=/root/.local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Application configuration defaults
ENV REDIS_HOST=redis \
    REDIS_PORT=6379 \
    TIMESFM_MAX_CONTEXT=4096 \
    PORT=8000

# Copy application source code
COPY . .

# Expose API port
EXPOSE 8000

# Container Healthcheck hitting enterprise dependency health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production uvicorn server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
