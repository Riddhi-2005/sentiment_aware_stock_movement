# ============================================================
#  dashboard.py  –  Optional: Streamlit Demo for Presentation
# ============================================================
#
#  Run:  streamlit run src/dashboard.py
#
#  Shows:
#    - Project overview
#    - Stock price trend with labels
#    - Sentiment over time
#    - Model comparison
#    - Make a prediction (select stock + date range)
# ============================================================

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Streamlit import ─────────────────────────────────────────
try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import STOCKS, DATA_FINAL, MODELS_DIR, PLOTS_DIR, METRICS_DIR
from src.feature_engineering import get_feature_columns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment-Aware Stock Movement Prediction for Indian Equity Market | Indian Equity Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: bold;
        background: linear-gradient(90deg, #1565C0, #42A5F5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #F5F5F5; padding: 12px; border-radius: 8px;
        border-left: 4px solid #1565C0; margin-bottom: 10px;
    }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #1565C0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load data helper
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = {
        "test":  os.path.join(DATA_FINAL, "features_test.csv"),
        "train": os.path.join(DATA_FINAL, "features_train.csv"),
        "all":   os.path.join(DATA_FINAL, "features_all.csv"),
    }
    data = {}
    for k, p in paths.items():
        if os.path.exists(p):
            data[k] = pd.read_csv(p, parse_dates=["Date"])
    return data


@st.cache_resource
def load_models():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
        "LightGBM":            "lightgbm.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_metrics():
    path = os.path.join(METRICS_DIR, "model_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/BSE_logo.svg/320px-BSE_logo.svg.png",
             width=150)
    st.markdown("## 📊 Navigation")
    page = st.radio("Go to", [
        "🏠 Overview",
        "📈 Stock Explorer",
        "💬 Sentiment Analysis",
        "🤖 Model Results",
        "🔮 Make Prediction",
    ])
    st.markdown("---")
    st.markdown("**Project Details**")
    st.markdown("- 10 NSE Stocks")
    st.markdown("- Jan 2016 – Dec 2019")
    st.markdown("- FinBERT Sentiment")
    st.markdown("- 4 ML Models")
    st.markdown("---")
    st.caption("College Data Science Project\nSentiment-Aware Stock Movement Prediction for Indian Equity Market")


# ─────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────
data    = load_data()
models  = load_models()
metrics = load_metrics()

all_df = data.get("all", pd.DataFrame())


# ─────────────────────────────────────────────────────────────
# Page 1: Overview
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<p class="main-header">📈 Sentiment-Aware Stock Movement Prediction for Indian Equity Market</p>',
                unsafe_allow_html=True)
    st.markdown("**Indian Equity Market | NSE Stocks | 2016–2019**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📌 Stocks Analysed", "10")
    with col2:
        st.metric("📰 Sentiment Model", "FinBERT")
    with col3:
        st.metric("📅 Date Range", "2016 – 2019")
    with col4:
        if metrics is not None:
            best_acc = metrics["Accuracy"].max()
            st.metric("🏆 Best Accuracy", f"{best_acc*100:.1f}%")
        else:
            st.metric("🏆 Models Trained", str(len(models)))

    st.markdown("---")
    st.markdown("### 🔬 Project Pipeline")
    cols = st.columns(5)
    steps = [
        ("1️⃣", "Data Collection", "Stock prices (yfinance) + Financial news (Kaggle)"),
        ("2️⃣", "Preprocessing",   "Text cleaning, return calculation, date alignment"),
        ("3️⃣", "Sentiment",       "FinBERT scores each day's news articles"),
        ("4️⃣", "Features",        "Technical indicators + sentiment + lag features"),
        ("5️⃣", "Modelling",       "LR, Random Forest, XGBoost, LightGBM + evaluation"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"#### {icon} {title}")
            st.caption(desc)

    st.markdown("---")
    st.markdown("### 📋 Stocks in Study")
    stock_info = {
        "RELIANCE":   "Energy & Petrochemicals",
        "INFOSYS":    "IT Services",
        "AXISBANK":   "Banking",
        "ITC":        "FMCG & Tobacco",
        "HDFCBANK":   "Banking",
        "TCS":        "IT Services",
        "LT":         "Construction/Infrastructure",
        "BHARTIARTL": "Telecom",
        "SBIN":       "Banking",
        "ICICIBANK":  "Banking",
    }
    cols = st.columns(5)
    for i, (name, sector) in enumerate(stock_info.items()):
        with cols[i % 5]:
            st.markdown(f"**{name}**\n\n*{sector}*")


# ─────────────────────────────────────────────────────────────
# Page 2: Stock Explorer
# ─────────────────────────────────────────────────────────────
elif page == "📈 Stock Explorer":
    st.markdown('<p class="section-header">📈 Stock Price Explorer</p>',
                unsafe_allow_html=True)

    if all_df.empty:
        st.warning("Run the full pipeline first. No data found.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("Select Stock", list(STOCKS.keys()))
        with col2:
            show_preds = st.checkbox("Show Predicted Labels", value=True)

        sdf = all_df[all_df["Name"] == selected].sort_values("Date")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Price & Signals", "Volume", "RSI"],
                            row_heights=[0.6, 0.2, 0.2])

        # Candlestick
        fig.add_trace(go.Scatter(x=sdf["Date"], y=sdf["Close"],
                                  name="Close", line=dict(color="#1565C0")), row=1, col=1)
        if "SMA_20" in sdf.columns:
            fig.add_trace(go.Scatter(x=sdf["Date"], y=sdf["SMA_20"],
                                      name="SMA20", line=dict(color="orange", dash="dash"),
                                      opacity=0.7), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=sdf["Date"], y=sdf["Volume"],
                              name="Volume", marker_color="#90CAF9"), row=2, col=1)

        # RSI
        if "RSI" in sdf.columns:
            fig.add_trace(go.Scatter(x=sdf["Date"], y=sdf["RSI"],
                                      name="RSI", line=dict(color="#7B1FA2")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red",   row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

        fig.update_layout(height=700, showlegend=True,
                          title_text=f"{selected} — Price, Volume & RSI")
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(sdf))
        with col2:
            up_pct = sdf["Target"].mean() * 100 if "Target" in sdf.columns else 0
            st.metric("UP Days", f"{up_pct:.1f}%")
        with col3:
            avg_ret = sdf["Daily_Return"].mean() * 100 if "Daily_Return" in sdf.columns else 0
            st.metric("Avg Daily Return", f"{avg_ret:.3f}%")


# ─────────────────────────────────────────────────────────────
# Page 3: Sentiment Analysis
# ─────────────────────────────────────────────────────────────
elif page == "💬 Sentiment Analysis":
    st.markdown('<p class="section-header">💬 FinBERT Sentiment Analysis</p>',
                unsafe_allow_html=True)

    if all_df.empty or "sentiment_score" not in all_df.columns:
        st.warning("Sentiment data not found. Run sentiment_analysis.py first.")
    else:
        # Daily sentiment
        daily = all_df.groupby("Date")["sentiment_score"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["Date"], y=daily["sentiment_score"],
            mode="lines", name="Sentiment",
            line=dict(color="#1565C0", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(21, 101, 192, 0.1)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(title="Daily Market Sentiment Score (FinBERT)",
                          xaxis_title="Date",
                          yaxis_title="Sentiment Score [-1 to +1]",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution
        col1, col2 = st.columns(2)
        with col1:
            label_counts = all_df.groupby(
                "Date")["sentiment_score"].mean().apply(
                lambda s: "Positive" if s > 0.05 else (
                    "Negative" if s < -0.05 else "Neutral")
            ).value_counts()
            fig2 = px.pie(values=label_counts.values, names=label_counts.index,
                          title="Sentiment Distribution",
                          color_discrete_map={"Positive": "#43A047",
                                              "Negative": "#E53935",
                                              "Neutral":  "#FB8C00"})
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            sent_by_target = all_df.groupby("Target")["sentiment_score"].mean()
            fig3 = px.bar(
                x=["DOWN (0)", "UP (1)"],
                y=[sent_by_target.get(0, 0), sent_by_target.get(1, 0)],
                color=["DOWN", "UP"],
                color_discrete_map={"DOWN": "#E53935", "UP": "#43A047"},
                title="Avg Sentiment: UP vs DOWN Days",
                labels={"x": "Market Movement", "y": "Avg Sentiment Score"},
            )
            st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Page 4: Model Results
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.markdown('<p class="section-header">🤖 Model Performance Results</p>',
                unsafe_allow_html=True)

    if metrics is None:
        st.warning("Metrics not found. Run evaluation.py first.")
    else:
        # Formatted metrics table
        styled = metrics.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1_Weighted", "ROC_AUC"]:
            if col in styled.columns:
                styled[col] = styled[col].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(styled, use_container_width=True)

        # Bar chart
        fig = go.Figure()
        metric_cols = ["Accuracy", "F1_Weighted", "ROC_AUC"]
        colors_bar  = ["#1565C0", "#43A047", "#FB8C00"]

        for metric, color in zip(metric_cols, colors_bar):
            if metric in metrics.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics["Model"],
                    y=metrics[metric],
                    marker_color=color,
                    opacity=0.85,
                    text=[f"{v:.3f}" for v in metrics[metric]],
                    textposition="outside",
                ))

        fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="Random Baseline (50%)")
        fig.update_layout(title="Model Comparison",
                          barmode="group",
                          yaxis=dict(range=[0, 1.1], title="Score"),
                          height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Show saved plots
        for plot_name, title in [
            ("confusion_matrices.png", "Confusion Matrices"),
            ("roc_curves.png",         "ROC Curves"),
            ("feature_importance.png", "Feature Importance"),
            ("per_stock_accuracy.png", "Per-Stock Accuracy"),
        ]:
            path = os.path.join(PLOTS_DIR, plot_name)
            if os.path.exists(path):
                st.markdown(f"#### {title}")
                try:
                    st.image(path, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load {plot_name}: {e}")


# ─────────────────────────────────────────────────────────────
# Page 5: Make Prediction
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Make Prediction":
    st.markdown('<p class="section-header">🔮 Make a Prediction</p>',
                unsafe_allow_html=True)
    st.markdown("Select a stock and a date from the **test set** to see the model's prediction.")

    if not models or all_df.empty:
        st.warning("Models or data not found. Run the full pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_stock = st.selectbox("Stock", list(STOCKS.keys()))
        with col2:
            model_name = st.selectbox("Model", list(models.keys()))
        with col3:
            test_df = data.get("test", pd.DataFrame())
            if not test_df.empty:
                stock_test = test_df[test_df["Name"] == sel_stock]
                dates = sorted(stock_test["Date"].dt.date.unique())
                if dates:
                    sel_date = st.selectbox("Date (Test Set)", dates)
                else:
                    st.warning("No test data for this stock.")
                    st.stop()
            else:
                st.warning("Test data not found.")
                st.stop()

        if st.button("🔮 Predict", type="primary"):
            row = stock_test[stock_test["Date"].dt.date == sel_date]
            if row.empty:
                st.error("No data for selected date.")
            else:
                feature_cols = get_feature_columns(test_df)
                X = row[feature_cols].values.astype(np.float32)
                mdata  = models[model_name]
                scaler = mdata.get("scaler")
                Xp = scaler.transform(X) if scaler is not None else X

                pred  = mdata["model"].predict(Xp)[0]
                probs = mdata["model"].predict_proba(Xp)[0] \
                        if hasattr(mdata["model"], "predict_proba") else None

                actual = row["Target"].values[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    pred_label = "⬆️ UP" if pred == 1 else "⬇️ DOWN"
                    color = "green" if pred == 1 else "red"
                    st.markdown(f"### Prediction: :{color}[{pred_label}]")
                with col2:
                    act_label = "⬆️ UP" if actual == 1 else "⬇️ DOWN"
                    match = "✅ Correct!" if pred == actual else "❌ Incorrect"
                    st.markdown(f"### Actual: {act_label}")
                    st.markdown(f"**{match}**")
                with col3:
                    if probs is not None:
                        st.markdown(f"### Confidence")
                        st.progress(float(max(probs)))
                        st.caption(f"DOWN: {probs[0]*100:.1f}% | UP: {probs[1]*100:.1f}%")

                # Context
                with st.expander("📊 Market Context for this Date"):
                    display_cols = ["Date", "Close", "Daily_Return",
                                    "RSI", "MACD", "sentiment_score"]
                    available = [c for c in display_cols if c in row.columns]
                    st.dataframe(row[available].T)
