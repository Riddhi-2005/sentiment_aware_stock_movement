# ============================================================
#  sentiment_analysis.py  –  Step 3: FinBERT sentiment scoring
# ============================================================
#
#  Run:  python src/sentiment_analysis.py
#
#  What it does:
#    - Loads the aligned news CSV
#    - Scores each day's aggregated text using FinBERT
#      (ProsusAI/finbert: POSITIVE / NEGATIVE / NEUTRAL)
#    - Outputs a numerical sentiment score per (Date)
#    - Saves:  data/processed/news_with_sentiment.csv
#
#  FinBERT outputs:
#    positive → +1
#    neutral  →  0
#    negative → -1
#    sentiment_score = weighted sum of softmax probabilities
# ============================================================

import os
import sys
import logging
import warnings
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (FINBERT_MODEL, SENTIMENT_BATCH_SIZE,
                        MAX_TOKEN_LENGTH, DATA_PROCESSED,
                        NEWS_PROC_CSV, LOGS_DIR)

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "sentiment_analysis.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FinBERT loader
# ─────────────────────────────────────────────────────────────
def load_finbert():
    """
    Loads FinBERT tokeniser and model from HuggingFace.
    Downloads on first run (~500 MB); uses cache on subsequent runs.
    """
    log.info(f"  Loading FinBERT model: {FINBERT_MODEL} …")
    log.info("  (First run downloads ~500 MB — please wait)")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    log.info(f"  FinBERT loaded on device: {device}")
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, tokenizer, max_tokens: int = 500) -> list[str]:
    """
    Splits long text into chunks that fit within FinBERT's token limit.
    We use word-level splits for simplicity.
    """
    words = text.split()
    # Approximate: 1 word ≈ 1.3 tokens for financial text
    chunk_size = int(max_tokens / 1.3)
    return [" ".join(words[i: i + chunk_size])
            for i in range(0, max(len(words), 1), chunk_size)]


def score_batch(texts: list[str], tokenizer, model, device) -> list[dict]:
    """
    Runs FinBERT on a list of texts.
    Returns list of dicts with keys: positive, neutral, negative, sentiment_score.
    """
    import torch
    import torch.nn.functional as F

    results = []

    for text in texts:
        if not text or not text.strip():
            results.append({
                "positive": 0.0, "neutral": 1.0,
                "negative": 0.0, "sentiment_score": 0.0,
            })
            continue

        chunks = _chunk_text(text, tokenizer)
        chunk_scores = []

        for chunk in chunks:
            try:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKEN_LENGTH,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits

                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # FinBERT label order: positive=0, negative=1, neutral=2
                # (verify with model.config.id2label)
                label_map = model.config.id2label  # e.g. {0:'positive', 1:'negative', 2:'neutral'}
                score_dict = {label_map[i].lower(): float(probs[i])
                              for i in range(len(probs))}
                chunk_scores.append(score_dict)

            except Exception as exc:
                log.debug(f"    Chunk scoring error (skipping): {exc}")
                continue

        if not chunk_scores:
            results.append({
                "positive": 0.0, "neutral": 1.0,
                "negative": 0.0, "sentiment_score": 0.0,
            })
            continue

        # Average across chunks
        avg = {
            "positive": np.mean([s.get("positive", 0) for s in chunk_scores]),
            "neutral":  np.mean([s.get("neutral",  1) for s in chunk_scores]),
            "negative": np.mean([s.get("negative", 0) for s in chunk_scores]),
        }
        # Composite sentiment score in [-1, +1]
        avg["sentiment_score"] = avg["positive"] - avg["negative"]
        results.append(avg)

    return results


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def run_sentiment_analysis() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 3: Running FinBERT Sentiment Analysis (STOCK-SPECIFIC)")
    log.info("=" * 60)

    aligned_path = os.path.join(DATA_PROCESSED, "news_aligned_per_stock.csv")
    if not os.path.exists(aligned_path):
        log.error(f"  [ERROR] Aligned news not found: {aligned_path}")
        log.error("  Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(aligned_path, parse_dates=["Date"])
    log.info(f"  Loaded {len(df)} (date, stock) pairs for sentiment scoring.")

    # Load FinBERT
    tokenizer, model, device = load_finbert()

    # Score in batches
    log.info(f"  Scoring sentiment (batch size={SENTIMENT_BATCH_SIZE}) …")
    texts = df["news_text_raw"].fillna("").tolist()

    all_results = []
    n_batches = math.ceil(len(texts) / SENTIMENT_BATCH_SIZE)

    for i in tqdm(range(n_batches), desc="  FinBERT batches"):
        batch = texts[i * SENTIMENT_BATCH_SIZE: (i + 1) * SENTIMENT_BATCH_SIZE]
        batch_results = score_batch(batch, tokenizer, model, device)
        all_results.extend(batch_results)

    # Attach results
    df["sentiment_positive"] = [r["positive"] for r in all_results]
    df["sentiment_neutral"]  = [r["neutral"] for r in all_results]
    df["sentiment_negative"] = [r["negative"] for r in all_results]
    df["sentiment_score"]    = [r["sentiment_score"] for r in all_results]
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
    )

    save_path = os.path.join(DATA_PROCESSED, "news_with_sentiment_per_stock.csv")
    df.to_csv(save_path, index=False)
    log.info(f"  [OK] Stock-specific sentiment saved: {save_path}")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_sentiment_analysis()
    log.info("Sentiment analysis complete! Next step: python src/feature_engineering.py")
