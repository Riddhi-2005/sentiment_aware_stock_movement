# ============================================================
#  preprocessing.py  –  Step 2: Clean & prepare data
# ============================================================
#
#  Run:  python src/preprocessing.py
#
#  What it does:
#    1. Cleans stock price data (missing values, returns, labels)
#    2. Cleans news text (lowercase, punctuation, stopwords)
#    3. Aligns news dates with stock trading dates
#    4. Saves processed files to data/processed/
# ============================================================

import os
import sys
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (STOCKS, START_DATE, END_DATE, RETURN_THRESHOLD,
                        DATA_RAW, DATA_PROCESSED, NEWS_RAW_CSV, LOGS_DIR)

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "preprocessing.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── NLTK downloads ───────────────────────────────────────────
def download_nltk_resources():
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            log.warning(f"Could not download NLTK resource {resource}: {e}")

download_nltk_resources()

# ─────────────────────────────────────────────────────────────
# 1. Stock Data Cleaning
# ─────────────────────────────────────────────────────────────
def clean_stock_data(name: str) -> pd.DataFrame | None:
    """
    Reads raw stock CSV, cleans it, calculates daily return,
    and creates binary target label.

    Returns cleaned DataFrame or None if file missing.
    """
    raw_path = os.path.join(DATA_RAW, f"{name}_stock.csv")
    if not os.path.exists(raw_path):
        log.error(f"  [ERROR] Raw file missing: {raw_path}. Run data_collection.py first.")
        return None

    df = pd.read_csv(raw_path, parse_dates=["Date"])

    # Filter date range
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()

    # Drop rows where Close is missing
    df.dropna(subset=["Close"], inplace=True)

    # Sort by date
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing OHLV values using forward fill
    for col in ["Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col].ffill(inplace=True)

    # ── Daily return ─────────────────────────────────────────
    df["Daily_Return"] = df["Close"].pct_change()

    # ── Binary label: 1 = UP, 0 = DOWN/FLAT ─────────────────
    # Label is based on NEXT day's movement (what we want to predict)
    df["Target"] = (df["Daily_Return"].shift(-1) > RETURN_THRESHOLD).astype(int)

    # Drop first row (no return) and last row (no next-day label)
    df.dropna(subset=["Daily_Return", "Target"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Keep relevant columns
    df = df[["Date", "Name", "Ticker", "Open", "High", "Low",
             "Close", "Volume", "Daily_Return", "Target"]]

    log.info(f"  [OK] {name}: {len(df)} rows after cleaning. "
             f"UP labels: {df['Target'].sum()} ({df['Target'].mean()*100:.1f}%)")
    return df


def process_all_stocks() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1: Cleaning stock data")
    log.info("=" * 60)

    all_dfs = []
    for name in STOCKS:
        df = clean_stock_data(name)
        if df is not None:
            save_path = os.path.join(DATA_PROCESSED, f"{name}_stock_clean.csv")
            df.to_csv(save_path, index=False)
            all_dfs.append(df)

    if not all_dfs:
        log.error("No stock data found. Run data_collection.py first!")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(DATA_PROCESSED, "all_stocks_clean.csv")
    combined.to_csv(combined_path, index=False)
    log.info(f"\n  Combined stock data saved: {combined_path} ({len(combined)} rows)\n")
    return combined


# ─────────────────────────────────────────────────────────────
# 2. News Data Cleaning
# ─────────────────────────────────────────────────────────────
def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    For Kaggle Indian Financial News dataset.
    Returns (date_col, text_col).
    """
    # Exact column names from the Excel file
    date_col = "Date"
    text_col = "Title"  # Use headlines for sentiment
    
    # Verify columns exist
    if date_col not in df.columns or text_col not in df.columns:
        log.error(f"Expected columns 'Date' and 'Title' not found!")
        log.error(f"Available columns: {list(df.columns)}")
        raise ValueError("Column mismatch")
    
    return date_col, text_col


def clean_text(text: str, lemmatizer, stop_words: set) -> str:
    """Cleans a single news headline/article string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]

    return " ".join(tokens)


def process_news_data() -> pd.DataFrame | None:
    log.info("=" * 60)
    log.info("STEP 2: Cleaning news data")
    log.info("=" * 60)

    if not os.path.exists(NEWS_RAW_CSV):
        log.error(f"  [ERROR] News CSV not found: {NEWS_RAW_CSV}")
        log.error("  Please follow the Kaggle download instructions in data_collection.py")
        return None

    df = pd.read_csv(NEWS_RAW_CSV, low_memory=False)
    log.info(f"  Raw news: {len(df)} rows, columns: {list(df.columns)}")

    date_col, text_col = _detect_columns(df)
    log.info(f"  Using date_col='{date_col}', text_col='{text_col}'")
    
    # Parse dates - handles format like "May 22, 2020, Friday"
    log.info(f"  Parsing dates from column '{date_col}'...")
    
    # Try specific format first
    df["Date"] = pd.to_datetime(df[date_col], format="%B %d, %Y, %A", errors="coerce")
    
    # If that didn't work, try automatic parsing
    if df["Date"].isna().sum() > len(df) * 0.5:  # If more than 50% failed
        log.info("  Specific format failed, trying automatic parsing...")
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Remove timezone if present
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    
    # Keep only date part (remove time)
    df["Date"] = df["Date"].dt.normalize()
    
    # Drop rows where date parsing failed
    before = len(df)
    df.dropna(subset=["Date"], inplace=True)
    after = len(df)
    if before - after > 0:
        log.warning(f"  Dropped {before - after} rows with invalid dates")

    # Parse dates
    #df["Date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    #df["Date"] = df["Date"].dt.tz_localize(None)   # remove timezone
    #df["Date"] = df["Date"].dt.normalize()          # keep date only (midnight)
    #df.dropna(subset=["Date"], inplace=True)

    # Filter date range
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()
    log.info(f"  After date filter: {len(df)} rows")

    if df.empty:
        log.warning("  [WARN] No news found in the given date range. "
                    "Check your dataset date range.")

    # Clean text
    lemmatizer = WordNetLemmatizer()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    log.info("  Cleaning text (this may take a moment) …")
    df["text_raw"]   = df[text_col].astype(str)
    df["text_clean"] = df["text_raw"].apply(
        lambda t: clean_text(t, lemmatizer, stop_words)
    )

    # Remove empty text rows
    df = df[df["text_clean"].str.strip().ne("")]
    df.dropna(subset=["text_clean"], inplace=True)

    # Remove duplicates
    df.drop_duplicates(subset=["Date", "text_raw"], inplace=True)

    df = df[["Date", "text_raw", "text_clean"]].reset_index(drop=True)

    save_path = os.path.join(DATA_PROCESSED, "news_clean.csv")
    df.to_csv(save_path, index=False)
    log.info(f"  [OK] News data saved: {save_path} ({len(df)} rows)\n")
    return df

def tag_news_to_stocks(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags each news article to relevant stock(s) based on keyword matching.
    Returns a new DataFrame with (Date, Stock, text) rows.
    """
    log.info("  Tagging news to specific stocks...")
    
    # Define keywords for each stock
    stock_keywords = {
      "RELIANCE": [
          "reliance industries",
          "reliance industries ltd",
          "ril",
          "ril share",
          "ril stock",
          "mukesh ambani",
          "reliance group",
          "reliance jio",
          "jio",
          "jio telecom",
          "jio 4g",
          "jio fiber",
          "jio gigafiber",
          "reliance retail",
          "reliance petroleum",
          "jamnagar refinery",
          "petchem",
          "petrochemical business"
        ],
      "INFOSYS": [
          "infosys",
          "infosys ltd",
          "infy",
          "infy stock",
          "narayana murthy",
          "salil parekh",
          "vishal sikka",
          "infosys ceo",
          "infosys guidance",
          "infosys earnings",
          "infosys q1",
          "infosys q2",
          "infosys q3",
          "infosys q4"
        ],
      "ICICIBANK": [
          "icici bank",
          "icici bank ltd",
          "icici",
          "private lender icici",
          "chanda kochhar",
          "sandeep bakshi",
          "icici bank npa",
          "icici bank results",
          "repo rate"
        ],
      "ITC": [
          "itc ltd",
          "itc limited",
          "itc stock",
          "itc share",
          "itc cigarettes",
          "gold flake",
          "wills navy cut",
          "bingo snacks",
          "classmate notebooks",
          "itc hotels",
          "itc fmcg",
          "itc agri business",
          "itc paperboards"
        ],
      "HDFCBANK": [
          "hdfc bank",
          "hdfc bank ltd",
          "hdfcbank",
          "hdfc bank stock",
          "hdfc bank share",
          "aditya puri",
          "sashidhar jagdishan",
          "hdfc bank results",
          "hdfc bank npa",
          "hdfc bank loan book"
        ],
      "AXISBANK": [
          "axis bank",
          "axis bank ltd",
          "private lender axis",
          "axis bank npa",
          "axis bank results",
          "shikha sharma",
          "amitabh chaudhry"
        ],
      "TCS": [
          "tata consultancy services",
          "tcs",
          "tcs ltd",
          "india's largest it firm tcs",
          "tcs results",
          "tcs earnings",
          "raj eshwaran",
          "it sector"
        ],
      "BHARTIARTL": [
          "bharti airtel",
          "airtel",
          "airtel india",
          "airtel 4g",
          "airtel broadband",
          "sunil mittal",
          "airtel africa",
          "telecom operator airtel",
          "airtel tariff",
          "airtel arpu"
        ],
      "SBIN": [
          "state bank of india",
          "sbi",
          "sbi bank",
          "sbi stock",
          "sbi share",
          "sbi chairman",
          "sbi results",
          "sbi npa",
          "sbi loan",
          "sbi recapitalisation"
        ],
      "LT": [
          "larsen & toubro",
          "larsen and toubro",
          "l&t",
          "lt ltd",
          "infrastructure major l&t",
          "lt order book",
          "lt results"
        ]
    }
    
    tagged_rows = []
    general_market_news = 0
    stock_specific_news = 0
    
    for _, row in news_df.iterrows():
        text_lower = row["text_raw"].lower()
        matched_stocks = []
        
        # Check which stocks are mentioned using regex whole-word matching
        for stock, patterns in stock_keywords.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                matched_stocks.append(stock)
        
        if matched_stocks:
            # Stock-specific news - create one row per matched stock
            stock_specific_news += 1
            for stock in matched_stocks:
                tagged_rows.append({
                    "Date": row["Date"],
                    "Stock": stock,
                    "text_raw": row["text_raw"],
                    "text_clean": row["text_clean"],
                    "news_type": "stock_specific",
                })
        else:
            # General market news - apply to ALL stocks
            general_market_news += 1
            for stock in STOCKS.keys():
                tagged_rows.append({
                    "Date": row["Date"],
                    "Stock": stock,
                    "text_raw": row["text_raw"],
                    "text_clean": row["text_clean"],
                    "news_type": "general_market",
                })
    
    tagged_df = pd.DataFrame(tagged_rows)
    
    # Log statistics
    log.info(f"  Original articles: {len(news_df)}")
    log.info(f"  Stock-specific news: {stock_specific_news}")
    log.info(f"  General market news: {general_market_news}")
    log.info(f"  Total tagged rows: {len(tagged_df)}")
    log.info(f"\n  Per-stock breakdown:")
    
    for stock in STOCKS.keys():
        stock_news = tagged_df[
            (tagged_df["Stock"] == stock) & 
            (tagged_df["news_type"] == "stock_specific")
        ]
        count = len(stock_news)
        log.info(f"    {stock:<12} {count:>5} specific news articles")
    
    return tagged_df

# ─────────────────────────────────────────────────────────────
# 3. Align news + stock trading dates
# ─────────────────────────────────────────────────────────────
def align_news_to_trading_days(
        stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Date, Stock) pair, aggregates relevant news from the
    previous trading day.
    
    Returns DataFrame with columns: Date, Stock, news_text_raw, news_text_clean, news_count
    """
    log.info("=" * 60)
    log.info("STEP 3: Aligning news to trading days (STOCK-SPECIFIC)")
    log.info("=" * 60)
    
    # Tag news to stocks first
    tagged_news = tag_news_to_stocks(news_df)
    
    trading_dates = sorted(stock_df["Date"].unique())
    stocks = stock_df["Name"].unique()
    
    rows = []
    for stock in stocks:
        for trade_date in trading_dates:
            pd_date = pd.Timestamp(trade_date)
            
            # Look-back window
            if pd_date.weekday() == 0:  # Monday
                lookback_start = pd_date - timedelta(days=3)
            else:
                lookback_start = pd_date - timedelta(days=1)
            
            # Get news for THIS stock only
            stock_news = tagged_news[
                (tagged_news["Stock"] == stock) & 
                (tagged_news["Date"] >= lookback_start) & 
                (tagged_news["Date"] < pd_date)
            ]
            
            # Separate stock-specific vs general market news
            specific = stock_news[stock_news["news_type"] == "stock_specific"]
            general = stock_news[stock_news["news_type"] == "general_market"]
            
            rows.append({
                "Date": trade_date,
                "Stock": stock,
                "news_text_raw": " ".join(stock_news["text_raw"].tolist()),
                "news_text_clean": " ".join(stock_news["text_clean"].tolist()),
                "news_count": len(stock_news),
                "specific_news_count": len(specific),
                "general_news_count": len(general),
            })
    
    aligned = pd.DataFrame(rows)
    save_path = os.path.join(DATA_PROCESSED, "news_aligned_per_stock.csv")
    aligned.to_csv(save_path, index=False)
    log.info(f"  [OK] Saved: {save_path}")
    
    return aligned


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stock_df = process_all_stocks()
    news_df  = process_news_data()

    if news_df is not None:
        align_news_to_trading_days(stock_df, news_df)
        log.info("Preprocessing complete! Next step: python src/sentiment_analysis.py")
    else:
        log.warning("News data missing. Add your news CSV and re-run.")
        log.info("   Stock preprocessing is done. You can still proceed to "
                 "feature_engineering.py with technical indicators only.")
