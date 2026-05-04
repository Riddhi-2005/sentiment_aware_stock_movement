# ============================================================
#  evaluation.py  –  Step 6: Full evaluation & visualizations
# ============================================================
#
#  Run:  python src/evaluation.py
#
#  Generates:
#    - Accuracy, Precision, Recall, F1-Score for all models
#    - Confusion matrices
#    - ROC curves
#    - Feature importance plots
#    - Sentiment vs stock movement analysis
#    - Model comparison chart
#    - Per-stock accuracy breakdown
#    - All plots saved to results/plots/
#    - Metrics CSV saved to results/metrics/
# ============================================================

import os
import sys
import logging
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config  import DATA_FINAL, MODELS_DIR, PLOTS_DIR, METRICS_DIR, LOGS_DIR, STOCKS
from src.feature_engineering import get_feature_columns

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "evaluation.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Load models & data
# ─────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
    "LightGBM":            "lightgbm.pkl",
}

def load_models_and_data():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            log.info(f"  [OK] Loaded: {name}")
        else:
            log.warning(f"  [MISS] {name} not found. Run modeling.py first.")

    test_df  = pd.read_csv(os.path.join(DATA_FINAL, "features_test.csv"),
                           parse_dates=["Date"])
    train_df = pd.read_csv(os.path.join(DATA_FINAL, "features_train.csv"),
                           parse_dates=["Date"])
    return models, train_df, test_df


def get_preds(model_data: dict, X: np.ndarray):
    m = model_data["model"]
    scaler = model_data.get("scaler")
    Xp = scaler.transform(X) if scaler is not None else X
    preds = m.predict(Xp)
    probs = m.predict_proba(Xp)[:, 1] if hasattr(m, "predict_proba") else None
    return preds, probs


# ─────────────────────────────────────────────────────────────
# 1. Metrics table
# ─────────────────────────────────────────────────────────────
def compute_all_metrics(models: dict, X_test: np.ndarray,
                         y_test: np.ndarray) -> pd.DataFrame:
    log.info("\n  -- Metrics (Test Set) --")
    rows = []
    for name, mdata in models.items():
        preds, probs = get_preds(mdata, X_test)
        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, zero_division=0)
        rec   = recall_score(y_test, preds, zero_division=0)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        auc   = roc_auc_score(y_test, probs) if probs is not None else float("nan")

        rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1_Weighted": f1, "ROC_AUC": auc})
        log.info(f"  {name:<25} Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    df = pd.DataFrame(rows)
    path = os.path.join(METRICS_DIR, "model_metrics.csv")
    df.to_csv(path, index=False)
    log.info(f"  Metrics saved: {path}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Confusion Matrices
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, mdata), color in zip(axes, models.items(), COLORS):
        preds, _ = get_preds(mdata, X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["DOWN", "UP"],
                    yticklabels=["DOWN", "UP"],
                    linewidths=0.5)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual",    fontsize=9)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Confusion matrices saved: {path}")


# ─────────────────────────────────────────────────────────────
# 3. ROC Curves
# ─────────────────────────────────────────────────────────────
def plot_roc_curves(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for (name, mdata), color in zip(models.items(), COLORS):
        _, probs = get_preds(mdata, X_test)
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  ROC curves saved: {path}")


# ─────────────────────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(models: dict, feature_cols: list):
    tree_models = {k: v for k, v in models.items()
                   if k in ("Random Forest", "XGBoost", "LightGBM")}
    if not tree_models:
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, mdata), color in zip(axes, tree_models.items(), COLORS[1:]):
        importances = pd.Series(
            mdata["model"].feature_importances_, index=feature_cols
        ).nlargest(15).sort_values()

        importances.plot(kind="barh", ax=ax, color=color, alpha=0.8)
        ax.set_title(f"{name}\nTop 15 Features", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Feature importance saved: {path}")


# ─────────────────────────────────────────────────────────────
# 5. Model comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_model_comparison(metrics_df: pd.DataFrame):
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Weighted", "ROC_AUC"]
    metrics_df_clean = metrics_df.dropna(subset=metrics_to_plot[:3])

    x = np.arange(len(metrics_df_clean))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics_to_plot):
        vals = metrics_df_clean[metric].fillna(0).values
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=COLORS[i % len(COLORS)], alpha=0.85)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison - Test Set Metrics",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df_clean["Model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="red", linestyle="--", lw=1, alpha=0.5,
               label="Random baseline")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Model comparison chart saved: {path}")


# ─────────────────────────────────────────────────────────────
# 6. Per-stock accuracy (best model)
# ─────────────────────────────────────────────────────────────
def plot_per_stock_accuracy(models: dict, test_df: pd.DataFrame, feature_cols: list):
    # Find best model
    best_name = None
    best_f1   = -1
    for name, mdata in models.items():
        X = test_df[feature_cols].values.astype(np.float32)
        y = test_df["Target"].values
        preds, _ = get_preds(mdata, X)
        f1 = f1_score(y, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1   = f1
            best_name = name

    if best_name is None:
        return

    best_model = models[best_name]
    stock_rows = []

    for stock in STOCKS:
        s_df = test_df[test_df["Name"] == stock]
        if len(s_df) < 5:
            continue
        X = s_df[feature_cols].values.astype(np.float32)
        y = s_df["Target"].values
        preds, _ = get_preds(best_model, X)
        acc = accuracy_score(y, preds)
        stock_rows.append({"Stock": stock, "Accuracy": acc, "Samples": len(s_df)})

    if not stock_rows:
        return

    df = pd.DataFrame(stock_rows).sort_values("Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = ["#E53935" if a < 0.5 else "#43A047" for a in df["Accuracy"]]
    bars = ax.barh(df["Stock"], df["Accuracy"], color=colors_bar, alpha=0.85)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row.Accuracy:.2f}", va="center", fontsize=9)

    ax.axvline(0.5, color="red", linestyle="--", lw=1.5, label="50% baseline")
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title(f"Per-Stock Accuracy — {best_name} (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "per_stock_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Per-stock accuracy saved: {path}")
    log.info(f"  Best model for this chart: {best_name}")


# ─────────────────────────────────────────────────────────────
# 7. Sentiment distribution vs target
# ─────────────────────────────────────────────────────────────
def plot_sentiment_vs_target(test_df: pd.DataFrame):
    if "sentiment_score" not in test_df.columns:
        log.info("  Sentiment column not found – skipping sentiment plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution of sentiment by UP/DOWN
    ax = axes[0]
    for label, color, ls in [(1, "#43A047", "-"), (0, "#E53935", "--")]:
        subset = test_df[test_df["Target"] == label]["sentiment_score"]
        subset.hist(bins=40, ax=ax, alpha=0.6, color=color, linestyle=ls,
                    label="UP (1)" if label == 1 else "DOWN (0)", density=True)
    ax.set_title("Sentiment Score Distribution by Target",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sentiment Score [-1, +1]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.axvline(0, color="k", linestyle=":", lw=1)

    # Mean sentiment over time
    ax2 = axes[1]
    daily_sent = test_df.groupby("Date")["sentiment_score"].mean().reset_index()
    ax2.plot(daily_sent["Date"], daily_sent["sentiment_score"],
             color="#2196F3", lw=1.5, alpha=0.8)
    ax2.axhline(0, color="k", linestyle="--", lw=1)
    ax2.fill_between(daily_sent["Date"], daily_sent["sentiment_score"],
                     where=daily_sent["sentiment_score"] >= 0,
                     color="#43A047", alpha=0.3, label="Positive")
    ax2.fill_between(daily_sent["Date"], daily_sent["sentiment_score"],
                     where=daily_sent["sentiment_score"] < 0,
                     color="#E53935", alpha=0.3, label="Negative")
    ax2.set_title("Daily Sentiment Trend (Test Period)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Mean Sentiment Score")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sentiment_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Sentiment plots saved: {path}")


# ─────────────────────────────────────────────────────────────
# 8. Classification report (best model, saved as text)
# ─────────────────────────────────────────────────────────────
def save_classification_reports(models: dict, X_test: np.ndarray, y_test: np.ndarray):
    report_path = os.path.join(METRICS_DIR, "classification_reports.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  CLASSIFICATION REPORTS — TEST SET\n")
        f.write("=" * 60 + "\n\n")

        for name, mdata in models.items():
            preds, _ = get_preds(mdata, X_test)
            report = classification_report(
                y_test, preds, target_names=["DOWN (0)", "UP (1)"], zero_division=0
            )
            f.write(f"\n-- {name} --\n")
            f.write(report)
            f.write("\n")

    log.info(f"  Classification reports saved: {report_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STEP 6: Evaluation & Visualization")
    log.info("=" * 60)

    models, train_df, test_df = load_models_and_data()

    if not models:
        log.error("No models found. Run modeling.py first.")
        sys.exit(1)

    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["Target"].values

    # ── Run all evaluations ──────────────────────────────────
    metrics_df = compute_all_metrics(models, X_test, y_test)

    log.info("\n  Generating plots..")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_feature_importance(models, feature_cols)
    plot_model_comparison(metrics_df)
    plot_per_stock_accuracy(models, test_df, feature_cols)
    plot_sentiment_vs_target(test_df)
    save_classification_reports(models, X_test, y_test)

    log.info("\n" + "=" * 60)
    log.info("  FINAL SUMMARY")
    log.info("=" * 60)
    best_row = metrics_df.loc[metrics_df["F1_Weighted"].idxmax()]
    log.info(f"     Best Model  : {best_row['Model']}")
    log.info(f"     Accuracy    : {best_row['Accuracy']:.4f}  ({best_row['Accuracy']*100:.1f}%)")
    log.info(f"     F1 Score    : {best_row['F1_Weighted']:.4f}")
    log.info(f"     ROC-AUC     : {best_row['ROC_AUC']:.4f}")
    log.info(f"\n   Plots   -> {PLOTS_DIR}")
    log.info(f"     Metrics -> {METRICS_DIR}")
    log.info("\n    Evaluation complete!")
    log.info("   Optional next step: streamlit run src/dashboard.py")
