import os

# Project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sub-directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATA_PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store")
DB_PATH = os.path.join(FEATURE_STORE_DIR, "finance_features.db")

# Create directories if they do not exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_PREDICTIONS_DIR, PLOTS_DIR, FEATURE_STORE_DIR]:
    os.makedirs(path, exist_ok=True)
