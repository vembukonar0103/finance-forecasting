"""
tests/unit/test_feature_store.py
---------------------------------
Unit tests for feature store operations:
- Legacy SQLite point-in-time storage (features/feature_store.py)
- Feast feature store adapter (features/feast_wrapper.py)
"""

import os
import pytest
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from features.feature_store import init_db, store_features, get_latest_features
from features.feast_wrapper import ensure_offline_stores, get_online_features, INDICATORS_25, VOLATILITY_FEATURES


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database path for isolated testing."""
    db_file = tmp_path / "test_features.db"
    return str(db_file)


def test_sqlite_feature_store_roundtrip(temp_db):
    """Verifies storing and retrieving features in SQLite."""
    with patch("features.feature_store.DB_PATH", temp_db):
        init_db()

        # Build dummy features dataframe
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        feats_df = pd.DataFrame({
            "timestamp": dates.strftime("%Y-%m-%d"),
            "rsi": [45.0, 50.0, 55.0, 60.0, 65.0],
            "macd": [0.1, 0.2, 0.3, 0.4, 0.5]
        })

        # Store features
        store_features("AAPL", feats_df)

        # Retrieve latest features
        latest = get_latest_features("AAPL")
        assert latest is not None
        assert "rsi" in latest
        assert latest["rsi"] == 65.0
        assert latest["macd"] == 0.5


def test_sqlite_feature_store_empty_ticker(temp_db):
    """Querying an unknown ticker returns empty dict."""
    with patch("features.feature_store.DB_PATH", temp_db):
        init_db()
        latest = get_latest_features("UNKNOWN")
        assert latest == {}


def test_feast_ensure_offline_stores(tmp_path):
    """Verifies that offline Parquet feature files are populated with valid schemas."""
    mock_data_dir = tmp_path / "feature_repo_data"
    mock_data_dir.mkdir(parents=True, exist_ok=True)

    with patch("features.feast_wrapper.DATA_DIR", str(mock_data_dir)):
        ensure_offline_stores()
        tech_path = os.path.join(mock_data_dir, "technical_indicators.parquet")
        vol_path = os.path.join(mock_data_dir, "volatility_regime.parquet")
        assert os.path.exists(tech_path)
        assert os.path.exists(vol_path)

        df_tech = pd.read_parquet(tech_path)
        assert "ticker" in df_tech.columns
        assert "event_timestamp" in df_tech.columns
        for ind in INDICATORS_25:
            assert ind in df_tech.columns


def test_feast_online_features_fallback_to_sqlite():
    """When Feast online store has no data or is uninitialized, falls back to SQLite store."""
    with patch("features.feast_wrapper.get_feature_store", return_value=None), \
         patch("features.feast_wrapper.sqlite_get_latest_features", return_value={"rsi": 58.2, "macd": 1.1}):
        feats = get_online_features("AAPL", feature_view="all")
        assert feats.get("rsi") == 58.2
        assert feats.get("macd") == 1.1
