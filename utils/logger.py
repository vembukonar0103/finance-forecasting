"""
utils/logger.py
---------------
Structured JSON Logger for Production Observability.

Phase 6 Deliverable:
- Emits structured JSON logs containing timestamp, log level, endpoint, ticker,
  latency_ms, cache_hit status, model, and message.
- Compatible with modern log shippers (Datadog, Loki, Fluentd, CloudWatch).
"""

import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any


class JsonFormatter(logging.Formatter):
    """Formats log records into compact, structured JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Inject standard contextual fields if present
        for field in ["endpoint", "ticker", "latency_ms", "cache_hit", "model", "status_code"]:
            if hasattr(record, field):
                log_obj[field] = getattr(record, field)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(name: str = "finance_forecasting", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()
