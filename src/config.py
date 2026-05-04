# ============================================================
#  config.py  –  Central configuration for the entire project
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_FINAL      = os.path.join(BASE_DIR, "data", "final")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# Create directories if missing
for _dir in [DATA_RAW, DATA_PROCESSED, DATA_FINAL,
             MODELS_DIR, PLOTS_DIR, METRICS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Stocks ───────────────────────────────────────────────────
# NSE tickers recognised by yfinance  (append .NS)
STOCKS = {
    "RELIANCE":   "RELIANCE.NS",
    "INFOSYS":    "INFY.NS",
    "AXISBANK":  "AXISBANK.NS",
    "ITC":        "ITC.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "SBIN":       "SBIN.NS",
    "ICICIBANK":      "ICICIBANK.NS",
}

STOCK_TICKERS   = list(STOCKS.values())   # ['RELIANCE.NS', ...]
STOCK_NAMES     = list(STOCKS.keys())     # ['RELIANCE', ...]

# ── Date range ───────────────────────────────────────────────
START_DATE = "2016-01-01"
END_DATE   = "2019-12-31"

# ── Target label ────────────────────────────────────────────
# Binary: 1 = stock went UP, 0 = stock went DOWN or stayed flat
# A return > THRESHOLD is labelled as UP
RETURN_THRESHOLD = 0.0   # 0.0 means any positive return = UP

# ── Sentiment model ──────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"   # HuggingFace model id
SENTIMENT_BATCH_SIZE = 16            # reduce to 8 if RAM is tight
MAX_TOKEN_LENGTH     = 512

# ── Feature engineering ──────────────────────────────────────
SMA_WINDOWS   = [5, 10, 20]          # Simple Moving Average windows
ROLLING_SENT  = [3, 7]               # Rolling sentiment windows
LAG_DAYS      = [1, 2, 3]            # Lag features for return & sentiment

# ── Train / Val / Test split ────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  = 0.15

# ── Random seed ──────────────────────────────────────────────
SEED = 42

# ── Kaggle news dataset ──────────────────────────────────────
# We will use:  https://www.kaggle.com/datasets/hkapoor/indian-financial-news
# Place the downloaded CSV here (or the download script will put it here):
KAGGLE_DATASET   = "hkapoor/indian-financial-news"
NEWS_RAW_CSV     = os.path.join(DATA_RAW, "indian_financial_news.csv")
NEWS_PROC_CSV    = os.path.join(DATA_PROCESSED, "news_with_sentiment.csv")

# ── Model file names ─────────────────────────────────────────
MODEL_FILES = {
    "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "random_forest":       os.path.join(MODELS_DIR, "random_forest.pkl"),
    "xgboost":             os.path.join(MODELS_DIR, "xgboost.pkl"),
    "lightgbm":            os.path.join(MODELS_DIR, "lightgbm.pkl"),
}
