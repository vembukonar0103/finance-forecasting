"""
tests/unit/test_data_loader.py
-------------------------------
Unit tests for data ingestion, caching, and cleaning in utils/data_loader.py.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from utils.data_loader import load_data


def test_load_data_from_cached_parquet(tmp_path):
    """Verifies load_data reads cached Parquet file without calling yfinance."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dummy_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="B"),
        "Open": [100.0] * 10,
        "High": [105.0] * 10,
        "Low": [95.0] * 10,
        "Close": [102.0] * 10,
        "Volume": [1000] * 10
    })
    target_parquet = cache_dir / "AAPL_2y_1d_raw.parquet"
    dummy_df.to_parquet(str(target_parquet), index=False)

    with patch("utils.data_loader.DATA_RAW_DIR", str(cache_dir)), \
         patch("yfinance.download") as mock_yf:
        res = load_data("AAPL", period="2y", interval="1d")
        mock_yf.assert_not_called()
        assert len(res) == 10
        assert "timestamps" in res.columns
        assert "close" in res.columns


def test_load_data_downloads_and_cleans_multiindex(tmp_path):
    """Verifies load_data calls yfinance when uncached and normalizes column headers."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_download = pd.DataFrame({
        ("Date", ""): pd.date_range("2024-01-01", periods=5, freq="B"),
        ("Close", "MSFT"): [300.0, 305.0, 310.0, 312.0, 315.0],
        ("Volume", "MSFT"): [5000, 5200, 4800, 5100, 5300]
    })

    with patch("utils.data_loader.DATA_RAW_DIR", str(cache_dir)), \
         patch("yfinance.download", return_value=raw_download):
        res = load_data("MSFT", period="1y", interval="1d", force_download=True)
        assert len(res) == 5
        assert "timestamps" in res.columns
        assert "close" in res.columns
        assert "volume" in res.columns
