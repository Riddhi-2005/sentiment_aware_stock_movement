# ============================================================
#  data_collection.py  –  Step 1: Download stock prices & news
# ============================================================
#
#  Run:  python src/data_collection.py
#
#  What it does:
#    1. Downloads historical OHLCV data for all 10 NSE stocks
#       using yfinance  →  data/raw/<TICKER>_stock.csv
#    2. Guides you to download the Kaggle news CSV and places
#       it in  data/raw/indian_financial_news.csv
# ============================================================

import os
import sys
import logging
import pandas as pd
import yfinance as yf

# ── Add project root to path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (STOCKS, START_DATE, END_DATE,
                        DATA_RAW, NEWS_RAW_CSV, KAGGLE_DATASET, LOGS_DIR)

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "data_collection.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Stock price download
# ─────────────────────────────────────────────────────────────
def download_stock_data() -> None:
    """
    Downloads daily OHLCV data for each stock in STOCKS dict
    and saves it as  data/raw/<NAME>_stock.csv
    """
    log.info("=" * 60)
    log.info("STEP 1: Downloading stock price data from Yahoo Finance")
    log.info("=" * 60)

    for name, ticker in STOCKS.items():
        save_path = os.path.join(DATA_RAW, f"{name}_stock.csv")

        if os.path.exists(save_path):
            log.info(f"  [SKIP] {name} already downloaded → {save_path}")
            continue

        log.info(f"  Downloading {name} ({ticker}) …")
        try:
            df = yf.download(
                ticker,
                start="2016-01-01",
                end="2019-12-31",
                auto_adjust=True,   # adjusts for splits & dividends
                actions= True
            )

            if df.empty:
                log.warning(f"  [WARN] No data returned for {name}. Check ticker.")
                continue

            # Flatten MultiIndex columns if present (yfinance ≥ 0.2.35)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.index.name = "Date"
            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            df["Name"]   = name

            df.to_csv(save_path, index=False)
            log.info(f"  [OK]   {name}: {len(df)} rows saved → {save_path}")

        except Exception as exc:
            log.error(f"  [ERROR] Failed to download {name}: {exc}")

    log.info("Stock download complete.\n")


# ─────────────────────────────────────────────────────────────
# 2. Kaggle news dataset
# ─────────────────────────────────────────────────────────────
def download_news_data() -> None:
    """
    Tries to download the Kaggle dataset automatically.
    Falls back to manual instructions if kaggle API key is missing.
    """
    log.info("=" * 60)
    log.info("STEP 2: Downloading Indian Financial News from Kaggle")
    log.info("=" * 60)

    if os.path.exists(NEWS_RAW_CSV):
        log.info(f"  [SKIP] News CSV already exists → {NEWS_RAW_CSV}")
        return

    try:
        import kaggle  # noqa: F401 – check it's installed

        log.info(f"  Using Kaggle API to download: {KAGGLE_DATASET}")
        os.system(
            f"kaggle datasets download -d {KAGGLE_DATASET} "
            f"--unzip -p {DATA_RAW}"
        )

        # Rename / identify the downloaded file
        _rename_kaggle_file()

    except ImportError:
        log.warning("  [WARN] kaggle library not found. Install with: pip install kaggle")
        # _print_manual_instructions()

    except Exception as exc:
        log.error(f"  [ERROR] Kaggle download failed: {exc}")
        # _print_manual_instructions()


def _rename_kaggle_file() -> None:
    """Searches for the downloaded CSV and renames it to the expected path."""
    for fname in os.listdir(DATA_RAW):
        if fname.endswith(".csv") and "news" in fname.lower():
            src = os.path.join(DATA_RAW, fname)
            os.rename(src, NEWS_RAW_CSV)
            log.info(f"  [OK] Renamed '{fname}' → {NEWS_RAW_CSV}")
            return
    log.warning("  [WARN] Could not identify downloaded CSV. "
                "Please rename it manually to: " + NEWS_RAW_CSV)


# def _print_manual_instructions() -> None:
#     log.info("""
#     ─── MANUAL DOWNLOAD INSTRUCTIONS ───────────────────────────
#     1. Go to: https://www.kaggle.com/datasets/hkapoor/indian-financial-news
#     2. Click "Download" (you need a free Kaggle account)
#     3. Unzip the downloaded file
#     4. Rename / copy the CSV to:
#        data/raw/indian_financial_news.csv
#     ALTERNATIVE DATASET (if above is unavailable):
#     https://www.kaggle.com/datasets/shivkumarshetgar/indian-financial-news-dataset
#     OR
#     https://www.kaggle.com/datasets/nitsanb/india-financial-news

#     Expected columns (at minimum):
#       date / published_at / Date  →  publication date
#       headline / title / text     →  news headline or body
#     ─────────────────────────────────────────────────────────────""")


# ─────────────────────────────────────────────────────────────
# 3. Validate downloads
# ─────────────────────────────────────────────────────────────
def validate_downloads() -> None:
    log.info("=" * 60)
    log.info("STEP 3: Validating downloaded files")
    log.info("=" * 60)

    # Stock files
    missing_stocks = []
    for name in STOCKS:
        path = os.path.join(DATA_RAW, f"{name}_stock.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            log.info(f"  [OK] {name}: {len(df)} rows, columns: {list(df.columns)}")
        else:
            missing_stocks.append(name)
            log.warning(f"  [MISS] {name} stock file not found!")

    if missing_stocks:
        log.warning(f"\n  Missing stock files: {missing_stocks}")
        log.warning("  Re-run download_stock_data() or check your internet connection.\n")

    # News file
    if os.path.exists(NEWS_RAW_CSV):
        df = pd.read_csv(NEWS_RAW_CSV, nrows=5)
        log.info(f"\n  [OK] News CSV found. Sample columns: {list(df.columns)}")
        log.info(f"       First row:\n{df.iloc[0].to_string()}")
    else:
        log.warning(f"\n  [MISS] News CSV not found at: {NEWS_RAW_CSV}")
        log.warning("  Follow the manual instructions above.\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_stock_data()
    #download_news_data()
    validate_downloads()
    log.info("\nData collection complete! Next step: python src/preprocessing.py")
