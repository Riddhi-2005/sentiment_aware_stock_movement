# 📈 Sentiment-Aware Stock Movement Prediction for Indian Equity Market

> A Data Science college project that uses **FinBERT** (Financial NLP) and **Machine Learning** to predict whether an Indian stock will move **UP or DOWN** the next day, based on financial news sentiment and technical indicators.

---

## 🗂️ Project Structure

```
stock_sentiment_project/
├── data/
│   ├── raw/              ← Original downloaded data
│   ├── processed/        ← Cleaned & transformed data
│   └── final/            ← ML-ready features (train/val/test CSVs)
├── src/
│   ├── config.py              ← All settings (tickers, dates, params)
│   ├── data_collection.py     ← Step 1: Download stock & news data
│   ├── preprocessing.py       ← Step 2: Clean data, create labels
│   ├── sentiment_analysis.py  ← Step 3: Run FinBERT on news
│   ├── feature_engineering.py ← Step 4: Build ML feature matrix
│   ├── modeling.py            ← Step 5: Train & tune models
│   ├── evaluation.py          ← Step 6: Metrics, plots, reports
│   └── dashboard.py           ← Optional: Streamlit demo app
├── models/               ← Saved trained models (.pkl)
├── results/
│   ├── plots/            ← All charts (PNG)
│   └── metrics/          ← Accuracy tables (CSV + TXT)
├── logs/                 ← Step-by-step log files
├── requirements.txt      ← All Python libraries
├── run_pipeline.py       ← Run everything end-to-end
└── README.md
```

---

## 📦 Stocks Covered

| Stock | Ticker | Sector |
|-------|--------|--------|
| Reliance Industries | RELIANCE.NS | Energy |
| Infosys | INFY.NS | IT Services |
| Eicher Motors | EICHERMOT.NS | Automobile |
| ITC | ITC.NS | FMCG |
| HDFC Bank | HDFCBANK.NS | Banking |
| Asian Paints | ASIANPAINT.NS | Chemicals |
| Bajaj Finance | BAJFINANCE.NS | NBFC |
| Bharti Airtel | BHARTIARTL.NS | Telecom |
| SBI | SBIN.NS | Banking |
| Trent | TRENT.NS | Retail |

**Date Range:** 2021-01-01 → 2024-12-31  
**Task:** Binary classification — will stock go UP (1) or DOWN (0) next day?

---

## ⚙️ Setup Instructions

### Step 1: Clone / Download the project
```bash
cd stock_sentiment_project
```

### Step 2: Create a Python virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install all libraries
```bash
pip install -r requirements.txt
```

> ⚠️ This installs PyTorch + HuggingFace Transformers (for FinBERT).  
> Takes 5–10 minutes on first install. FinBERT model (~500 MB) downloads on first run.

### Step 4: Kaggle API Setup (for news data)

1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → API → "Create New Token"
2. Download `kaggle.json`
3. Place it at:
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
4. Set permissions (Mac/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

**If you don't want to use the API:**
- Go to: https://www.kaggle.com/datasets/hkapoor/indian-financial-news
- Click Download → unzip
- Copy the CSV file to: `data/raw/indian_financial_news.csv`

---

## 🚀 How to Run

### Option A: Run Everything at Once
```bash
python run_pipeline.py
```

### Option B: Run Step by Step (Recommended for debugging)
```bash
# Step 1: Download data
python src/data_collection.py

# Step 2: Clean and preprocess
python src/preprocessing.py

# Step 3: Run FinBERT sentiment analysis
python src/sentiment_analysis.py

# Step 4: Build features
python src/feature_engineering.py

# Step 5: Train models
python src/modeling.py

# Step 6: Evaluate and generate plots
python src/evaluation.py

# Optional: Launch demo dashboard
streamlit run src/dashboard.py
```

---

## 🔍 Debugging Guide

### ❌ "Module not found" Error
```bash
# Make sure virtual environment is active first!
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

### ❌ "No data returned for RELIANCE" (yfinance error)
```bash
# Check your internet connection
# Try running this in Python to test:
import yfinance as yf
df = yf.download("RELIANCE.NS", start="2021-01-01", end="2021-06-01")
print(df.head())
```

### ❌ Kaggle download fails
- Check `kaggle.json` is placed correctly
- Try manual download (see Setup Step 4)
- Rename the CSV to exactly: `data/raw/indian_financial_news.csv`

### ❌ "CUDA out of memory" (FinBERT error)
```python
# In src/config.py, reduce batch size:
SENTIMENT_BATCH_SIZE = 4   # default is 16
```

### ❌ "Columns not detected" in news CSV
- Open `data/raw/indian_financial_news.csv` in Excel
- Find the column names for the date and headline
- Edit `src/preprocessing.py` → `_detect_columns()` function:
  ```python
  date_candidates = ["your_actual_date_column_name", ...]
  text_candidates = ["your_actual_headline_column_name", ...]
  ```

### ❌ FinBERT is too slow
- FinBERT runs on CPU by default if no GPU
- Reduce `SENTIMENT_BATCH_SIZE = 4` in `config.py`
- Alternatively: Use a sample of news data first to test

### ❌ Low accuracy (below 50%)
- Check class distribution: `df['Target'].value_counts()`
- Check for data leakage in feature engineering
- Make sure news dates are BEFORE the stock date being predicted
- Try increasing the `RETURN_THRESHOLD` in `config.py` to `0.005` (0.5%)

### ✅ Checking log files
```bash
# Each step saves a detailed log in logs/
cat logs/data_collection.log
cat logs/preprocessing.log
cat logs/sentiment_analysis.log
cat logs/feature_engineering.log
cat logs/modeling.log
cat logs/evaluation.log
```

---

## 📊 Expected Outputs

### After `evaluation.py`:
| File | Description |
|------|-------------|
| `results/plots/confusion_matrices.png` | Confusion matrices for all 4 models |
| `results/plots/roc_curves.png` | ROC curves with AUC scores |
| `results/plots/feature_importance.png` | Top 15 features per model |
| `results/plots/model_comparison.png` | Bar chart: accuracy/F1/AUC |
| `results/plots/per_stock_accuracy.png` | Which stocks are easier to predict |
| `results/plots/sentiment_analysis.png` | Sentiment distribution & trend |
| `results/metrics/model_metrics.csv` | All metrics in a table |
| `results/metrics/classification_reports.txt` | Detailed per-class metrics |

### Expected Accuracy Range:
- 🟡 **50-55%** = Decent (beats random)
- 🟢 **55-62%** = Good for stock prediction
- 🔵 **62-68%** = Very good

> Note: Stock prediction is inherently difficult. 55%+ is acceptable for an academic project.

---

## 🧠 Models Used

| Model | Why Used |
|-------|----------|
| Logistic Regression | Baseline, interpretable |
| Random Forest | Handles non-linearity, robust |
| XGBoost | Best for tabular financial data |
| LightGBM | Faster XGBoost, often better accuracy |

**Best model** is automatically identified and saved to `models/best_model_info.txt`

---

## 📐 Features Used

### Technical Indicators
- SMA (5, 10, 20 day)
- RSI (14-period)
- MACD & Signal line
- Bollinger Bands (width, %B)
- Volume ratio
- Volatility (5, 10 day)

### Sentiment Features (FinBERT)
- Daily sentiment score (-1 to +1)
- Positive / Negative / Neutral probabilities
- Rolling sentiment (3-day, 7-day)
- Sentiment lag (1, 2, 3 days)
- News count

### Lag Features
- Previous day/2-day/3-day return
- Previous day/2-day/3-day sentiment

---

## 📅 Data Split

| Split | Dates | Purpose |
|-------|-------|---------|
| Train | Jan 2021 – ~Jun 2023 (70%) | Model training |
| Validation | ~Jun 2023 – ~Mar 2024 (15%) | Hyperparameter tuning |
| Test | ~Mar 2024 – Dec 2024 (15%) | Final evaluation |

> ⚠️ Split is **time-based** (not random) to prevent data leakage.

---

## 🎯 Project Methodology

```
Financial News (Kaggle)
        ↓
   Text Cleaning (NLTK)
        ↓
  FinBERT Scoring → sentiment_score [-1, +1]
        ↓
Feature Engineering ←── Technical Indicators (RSI, MACD, etc.)
        ↓
   Binary Classifier → Predict: UP (1) or DOWN (0)
        ↓
 Evaluation: Accuracy, F1, ROC-AUC, Confusion Matrix
```

---

## 🙋 Presentation Tips

- Show the **model comparison chart** first
- Explain why stock prediction is hard (efficient market hypothesis)
- Highlight the **improvement** sentiment adds over technical-only model
- Show the **per-stock accuracy chart** — some sectors are more predictable
- Mention **FinBERT** specifically: it's trained on financial text, unlike general BERT
- Talk about **data leakage prevention** (time-based split)
- Future work: real-time API, LSTM/Transformer models, sector-specific models

---

## 📚 References

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [Yahoo Finance API (yfinance)](https://pypi.org/project/yfinance/)
- [Kaggle Indian Financial News Dataset](https://www.kaggle.com/datasets/hkapoor/indian-financial-news)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
