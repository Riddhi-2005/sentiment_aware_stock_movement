# ============================================================
#  feature_engineering.py  –  Step 4: Build ML feature matrix
# ============================================================
#
#  Run:  python src/feature_engineering.py
#
#  What it does:
#    - Merges cleaned stock data with FinBERT sentiment scores
#    - Adds technical indicators (SMA, RSI, MACD, Bollinger Bands)
#    - Adds lag features (previous day returns & sentiments)
#    - Adds rolling sentiment features
#    - Splits into train/val/test sets (time-based split)
#    - Saves:  data/final/features_train.csv
#              data/final/features_val.csv
#              data/final/features_test.csv
#              data/final/features_all.csv
# ============================================================

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (STOCKS, SMA_WINDOWS, ROLLING_SENT, LAG_DAYS,
                        TRAIN_RATIO, VAL_RATIO, DATA_PROCESSED,
                        DATA_FINAL, NEWS_PROC_CSV, LOGS_DIR, SEED)

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "feature_engineering.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Technical Indicators
# ─────────────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to a single-stock DataFrame sorted by Date.
    """
    close = df["Close"]
    volume = df["Volume"]

    # ── Simple Moving Averages ───────────────────────────────
    for w in SMA_WINDOWS:
        df[f"SMA_{w}"] = close.rolling(window=w).mean()
        df[f"SMA_{w}_ratio"] = close / df[f"SMA_{w}"]   # price relative to SMA

    # ── Exponential Moving Average (EMA 12, 26) ──────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    # ── MACD ─────────────────────────────────────────────────
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── RSI (14-period) ──────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (20-period, 2 std) ───────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (bb_mid + 1e-9)
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── Volume features ──────────────────────────────────────
    df["Volume_change"] = volume.pct_change()
    df["Volume_SMA5"]   = volume.rolling(5).mean()
    df["Volume_ratio"]  = volume / (df["Volume_SMA5"] + 1e-9)

    # ── Volatility (rolling std of returns) ─────────────────
    df["Volatility_5"]  = df["Daily_Return"].rolling(5).std()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    # ── High-Low range ───────────────────────────────────────
    df["HL_range"]  = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["OC_change"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)

    return df


# ─────────────────────────────────────────────────────────────
# 2. Lag & Rolling Sentiment Features
# ─────────────────────────────────────────────────────────────
def add_lag_and_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features for returns and sentiment, plus rolling sentiment.
    """
    # ── Lag: daily return ────────────────────────────────────
    for lag in LAG_DAYS:
        df[f"Return_lag{lag}"] = df["Daily_Return"].shift(lag)

    # ── Lag: sentiment score ─────────────────────────────────
    if "sentiment_score" in df.columns:
        for lag in LAG_DAYS:
            df[f"Sent_lag{lag}"] = df["sentiment_score"].shift(lag)

        # ── Rolling sentiment averages ───────────────────────
        for w in ROLLING_SENT:
            df[f"Sent_roll{w}"] = df["sentiment_score"].rolling(w).mean()

        # ── Sentiment change (momentum) ──────────────────────
        df["Sent_change"] = df["sentiment_score"].diff()
        df["Sent_pos"]    = df["sentiment_positive"].rolling(3).mean() \
                            if "sentiment_positive" in df.columns else 0.0
        df["Sent_neg"]    = df["sentiment_negative"].rolling(3).mean() \
                            if "sentiment_negative" in df.columns else 0.0

        # ── News count feature ───────────────────────────────
        if "news_count" in df.columns:
            df["News_count_log"] = np.log1p(df["news_count"])

    # ── Day of week (market patterns) ────────────────────────
    df["DayOfWeek"]  = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Month"]      = pd.to_datetime(df["Date"]).dt.month

    return df


# ─────────────────────────────────────────────────────────────
# 3. Main feature builder
# ─────────────────────────────────────────────────────────────
def build_features() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 4: Feature Engineering (STOCK-SPECIFIC SENTIMENT)")
    log.info("=" * 60)

    # Load stock data
    stock_path = os.path.join(DATA_PROCESSED, "all_stocks_clean.csv")
    stock_df = pd.read_csv(stock_path, parse_dates=["Date"])

    # Load stock-specific sentiment
    sent_path = os.path.join(DATA_PROCESSED, "news_with_sentiment_per_stock.csv")
    has_sentiment = os.path.exists(sent_path)
    
    if has_sentiment:
        sent_df = pd.read_csv(sent_path, parse_dates=["Date"])
        log.info(f"  Loaded stock-specific sentiment: {len(sent_df)} rows")
    else:
        log.warning("  No sentiment data found.")
        sent_df = None

    # Process each stock
    all_features = []
    for name in STOCKS:
        sdf = stock_df[stock_df["Name"] == name].copy()
        sdf.sort_values("Date", inplace=True)
        sdf.reset_index(drop=True, inplace=True)

        # Merge sentiment for THIS stock only
        if sent_df is not None:
            stock_sent = sent_df[sent_df["Stock"] == name][
                ["Date", "sentiment_score", "sentiment_positive",
                 "sentiment_negative", "sentiment_neutral", "news_count"]
            ]
            sdf = pd.merge(sdf, stock_sent, on="Date", how="left")
            
            # Forward-fill missing sentiment
            sent_cols = ["sentiment_score", "sentiment_positive",
                         "sentiment_negative", "sentiment_neutral", "news_count"]
            sdf[sent_cols] = sdf[sent_cols].ffill().fillna(0)

        # Add technical indicators
        sdf = add_technical_indicators(sdf)
        sdf = add_lag_and_sentiment_features(sdf)
        sdf["Stock_id"] = list(STOCKS.keys()).index(name)

        all_features.append(sdf)
        log.info(f"  [OK] {name}: {len(sdf)} rows, {sdf.shape[1]} features")

    combined = pd.concat(all_features, ignore_index=True)
    combined.sort_values(["Name", "Date"], inplace=True)

    # ── Drop NaN rows created by rolling windows ─────────────
    pre_drop = len(combined)
    combined.dropna(inplace=True)
    log.info(f"\n  Dropped {pre_drop - len(combined)} NaN rows (from rolling windows)")
    log.info(f"  Final feature matrix: {combined.shape}")

    # ── Save full feature matrix ──────────────────────────────
    all_path = os.path.join(DATA_FINAL, "features_all.csv")
    combined.to_csv(all_path, index=False)
    log.info(f"  Saved: {all_path}")

    return combined


# ─────────────────────────────────────────────────────────────
# 4. Train / Val / Test split (TIME-BASED)
# ─────────────────────────────────────────────────────────────
def time_based_split(df: pd.DataFrame) -> tuple:
    """
    Splits by date to avoid data leakage.
    Uses global TRAIN_RATIO / VAL_RATIO from config.
    """
    log.info("=" * 60)
    log.info("STEP 5: Train / Val / Test Split (time-based)")
    log.info("=" * 60)

    dates = sorted(df["Date"].unique())
    n = len(dates)

    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_cutoff = dates[train_end_idx]
    val_cutoff   = dates[val_end_idx]

    log.info(f"  Training   : {dates[0].date()} → {train_cutoff.date()} "
             f"({train_end_idx} dates)")
    log.info(f"  Validation : {train_cutoff.date()} → {val_cutoff.date()}")
    log.info(f"  Test       : {val_cutoff.date()} → {dates[-1].date()}")

    train_df = df[df["Date"] <= train_cutoff]
    val_df   = df[(df["Date"] > train_cutoff) & (df["Date"] <= val_cutoff)]
    test_df  = df[df["Date"] > val_cutoff]

    log.info(f"\n  Train size: {len(train_df)} rows  "
             f"| Val: {len(val_df)} | Test: {len(test_df)}")

    # Save splits
    for split_df, fname in [(train_df, "features_train.csv"),
                             (val_df,   "features_val.csv"),
                             (test_df,  "features_test.csv")]:
        path = os.path.join(DATA_FINAL, fname)
        split_df.to_csv(path, index=False)
        log.info(f"  Saved: {path}")

    # Class distribution
    log.info("\n  Target distribution (1=UP, 0=DOWN):")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pct = split["Target"].mean() * 100
        log.info(f"    {name}: UP={pct:.1f}%  DOWN={(100-pct):.1f}%")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# Helper: Feature column list (used by modeling.py)
# ─────────────────────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the list of feature columns (excludes metadata & target).
    """
    exclude = {"Date", "Name", "Ticker", "Target",
               "text_raw", "text_clean", "news_text_raw",
               "news_text_clean", "sentiment_label"}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feature_df = build_features()
    train_df, val_df, test_df = time_based_split(feature_df)

    feature_cols = get_feature_columns(train_df)
    log.info(f"\n  Total features for modelling: {len(feature_cols)}")
    log.info(f"  Feature list:\n    {feature_cols}\n")

    log.info("Feature engineering complete! Next step: python src/modeling.py")
