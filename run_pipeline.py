# ============================================================
#  run_pipeline.py  –  Run entire project pipeline end-to-end
# ============================================================
#
#  Usage:  python run_pipeline.py
#
#  Runs each step in order:
#    1. data_collection.py
#    2. preprocessing.py
#    3. sentiment_analysis.py
#    4. feature_engineering.py
#    5. modeling.py
#    6. evaluation.py
#
#  You can skip steps by commenting them out below.
#  Check logs/ folder for detailed output of each step.
# ============================================================

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run_step(name: str, func):
    log.info("\n" + "=" * 70)
    log.info(f"  RUNNING: {name}")
    log.info("=" * 70)
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        log.info(f"      {name} completed in {elapsed:.1f}s")
        return True
    except SystemExit:
        log.warning(f"      {name} exited early (possibly missing data). Check logs.")
        return False
    except Exception as exc:
        log.error(f"  X  {name} FAILED: {exc}")
        import traceback
        log.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    log.info("    Starting Full Pipeline: Sentiment-Aware Stock Prediction")
    log.info("=" * 70)
    total_start = time.time()

    # ── Import all steps ─────────────────────────────────────
    from src.data_collection    import download_stock_data, download_news_data, validate_downloads
    from src.preprocessing      import process_all_stocks, process_news_data, align_news_to_trading_days
    from src.sentiment_analysis import run_sentiment_analysis
    from src.feature_engineering import build_features, time_based_split, get_feature_columns
    from src.modeling            import (load_splits, get_Xy, compare_models,
                                         train_logistic_regression, train_random_forest,
                                         train_xgboost, train_lightgbm)
    from src.evaluation          import (load_models_and_data, compute_all_metrics,
                                         plot_confusion_matrices, plot_roc_curves,
                                         plot_feature_importance, plot_model_comparison,
                                         plot_per_stock_accuracy, plot_sentiment_vs_target,
                                         save_classification_reports)
    import numpy as np

    # ── Step 1: Data Collection ───────────────────────────────
    def step1():
        download_stock_data()
        download_news_data()
        validate_downloads()
    #run_step("Step 1: Data Collection", step1)

    # ── Step 2: Preprocessing ────────────────────────────────
    def step2():
        stock_df = process_all_stocks()
        news_df  = process_news_data()
        if news_df is not None:
            align_news_to_trading_days(stock_df, news_df)
    run_step("Step 2: Preprocessing", step2)

    # ── Step 3: Sentiment Analysis ────────────────────────────
    run_step("Step 3: FinBERT Sentiment Analysis", run_sentiment_analysis)

    # ── Step 4: Feature Engineering ───────────────────────────
    def step4():
        feature_df = build_features()
        train_df, val_df, test_df = time_based_split(feature_df)
        feature_cols = get_feature_columns(train_df)
        log.info(f"  Features built: {len(feature_cols)} columns")
    run_step("Step 4: Feature Engineering", step4)

    # ── Step 5: Modelling ────────────────────────────────────
    def step5():
        train_df, val_df, test_df = load_splits()
        feature_cols = get_feature_columns(train_df)
        X_train, y_train = get_Xy(train_df, feature_cols)
        X_val,   y_val   = get_Xy(val_df,   feature_cols)
        X_test,  y_test  = get_Xy(test_df,  feature_cols)

        results = []
        results.append(train_logistic_regression(X_train, y_train, X_val, y_val,
                                                  X_test,  y_test,  feature_cols))
        results.append(train_random_forest(X_train, y_train, X_val, y_val,
                                            X_test,  y_test,  feature_cols))
        results.append(train_xgboost(X_train, y_train, X_val, y_val,
                                      X_test,  y_test,  feature_cols))
        results.append(train_lightgbm(X_train, y_train, X_val, y_val,
                                       X_test,  y_test,  feature_cols))
        compare_models(results)
    run_step("Step 5: Model Training", step5)

    # ── Step 6: Evaluation ────────────────────────────────────
    def step6():
        models, train_df, test_df = load_models_and_data()
        feature_cols = get_feature_columns(test_df)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["Target"].values

        metrics_df = compute_all_metrics(models, X_test, y_test)
        plot_confusion_matrices(models, X_test, y_test)
        plot_roc_curves(models, X_test, y_test)
        plot_feature_importance(models, feature_cols)
        plot_model_comparison(metrics_df)
        plot_per_stock_accuracy(models, test_df, feature_cols)
        plot_sentiment_vs_target(test_df)
        save_classification_reports(models, X_test, y_test)
    run_step("Step 6: Evaluation & Visualization", step6)

    # ── Done ─────────────────────────────────────────────────
    total = time.time() - total_start
    log.info("\n" + "=" * 70)
    log.info(f"    PIPELINE COMPLETE in {total:.1f}s ({total/60:.1f} min)")
    log.info("=" * 70)
    log.info("\n  Next steps:")
    log.info("     View results/plots/     for all charts")
    log.info("     View results/metrics/   for CSV metrics")
    log.info("     View logs/              for detailed logs")
    log.info("     streamlit run src/dashboard.py   for demo\n")
