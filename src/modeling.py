# ============================================================
#  modeling.py  –  Step 5: Train, tune & save ML models
# ============================================================
#
#  Run:  python src/modeling.py
#
#  Models trained:
#    1. Logistic Regression  (baseline)
#    2. Random Forest
#    3. XGBoost
#    4. LightGBM
#
#  For each model:
#    - GridSearch / RandomSearch on validation set
#    - Best model saved to models/
#    - Training metrics logged
# ============================================================

import os
import sys
import logging
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report)
import xgboost  as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config  import DATA_FINAL, MODELS_DIR, LOGS_DIR, SEED
from src.feature_engineering import get_feature_columns

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "modeling.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
def load_splits():
    paths = {
        "train": os.path.join(DATA_FINAL, "features_train.csv"),
        "val":   os.path.join(DATA_FINAL, "features_val.csv"),
        "test":  os.path.join(DATA_FINAL, "features_test.csv"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            log.error(f"Missing: {path}. Run feature_engineering.py first.")
            sys.exit(1)

    train_df = pd.read_csv(paths["train"], parse_dates=["Date"])
    val_df   = pd.read_csv(paths["val"],   parse_dates=["Date"])
    test_df  = pd.read_csv(paths["test"],  parse_dates=["Date"])
    return train_df, val_df, test_df


def get_Xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values.astype(np.float32)
    y = df["Target"].values.astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────
# Hyperparameter search (lightweight grid search on val set)
# ─────────────────────────────────────────────────────────────
def grid_search_val(model_fn, param_grid: dict,
                    X_train, y_train, X_val, y_val,
                    model_name: str):
    """
    Trains model for every param combination,
    evaluates on val set, returns best model + params.
    Uses F1-score (weighted) as selection metric.
    """
    best_score  = -1
    best_params = None
    best_model  = None

    grid = list(ParameterGrid(param_grid))
    log.info(f"    Grid size: {len(grid)} combinations")

    for params in grid:
        try:
            m = model_fn(**params)
            m.fit(X_train, y_train)
            preds = m.predict(X_val)
            score = f1_score(y_val, preds, average="weighted", zero_division=0)

            if score > best_score:
                best_score  = score
                best_params = params
                best_model  = m
        except Exception as exc:
            log.debug(f"    Param error {params}: {exc}")
            continue

    log.info(f"    Best val F1: {best_score:.4f} | params: {best_params}")
    return best_model, best_params, best_score


# ─────────────────────────────────────────────────────────────
# Model 1: Logistic Regression
# ─────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, X_val, y_val,
                               X_test,  y_test,  feature_cols):
    log.info("\n  -- Logistic Regression --")

    param_grid = {
        "C":       [0.01, 0.1, 1.0, 10.0],
        "solver":  ["lbfgs", "liblinear"],
        "max_iter":[500],
        "random_state": [SEED],
    }

    # Scale features (LR is sensitive to scale)
    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_train)
    Xval_s  = scaler.transform(X_val)
    Xte_s   = scaler.transform(X_test)

    model, params, val_f1 = grid_search_val(
        LogisticRegression, param_grid,
        Xtr_s, y_train, Xval_s, y_val,
        "LogisticRegression"
    )

    # Wrap in pipeline for consistent predict interface
    pipe = Pipeline([("scaler", scaler), ("lr", model)])
    test_preds = model.predict(Xte_s)
    test_acc   = accuracy_score(y_test, test_preds)
    test_f1    = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    log.info(f"    Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # Save
    save_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols,
                 "params": params}, save_path)
    log.info(f"    Saved -> {save_path}")

    return {"name": "Logistic Regression", "model": model, "scaler": scaler,
            "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1}


# ─────────────────────────────────────────────────────────────
# Model 2: Random Forest
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val,
                         X_test,  y_test,  feature_cols):
    log.info("\n  -- Random Forest --")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [5, 10, None],
        "min_samples_split": [5, 10],
        "random_state": [SEED],
        "n_jobs":       [-1],
    }

    model, params, val_f1 = grid_search_val(
        RandomForestClassifier, param_grid,
        X_train, y_train, X_val, y_val,
        "RandomForest"
    )

    test_preds = model.predict(X_test)
    test_acc   = accuracy_score(y_test, test_preds)
    test_f1    = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    log.info(f"    Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    save_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    joblib.dump({"model": model, "scaler": None, "feature_cols": feature_cols,
                 "params": params}, save_path)
    log.info(f"    Saved -> {save_path}")

    # Feature importances (top 10)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    log.info(f"    Top 10 features:\n{top10.to_string()}")

    return {"name": "Random Forest", "model": model, "scaler": None,
            "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1,
            "feature_importances": importances}


# ─────────────────────────────────────────────────────────────
# Model 3: XGBoost
# ─────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val,
                   X_test,  y_test,  feature_cols):
    log.info("\n  -- XGBoost --")

    def xgb_fn(**kw):
        return xgb.XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            **kw,
        )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [4, 6],
        "learning_rate":[0.05, 0.1],
        "subsample":    [0.8, 1.0],
        "random_state": [SEED],
    }

    model, params, val_f1 = grid_search_val(
        xgb_fn, param_grid,
        X_train, y_train, X_val, y_val,
        "XGBoost"
    )

    test_preds = model.predict(X_test)
    test_acc   = accuracy_score(y_test, test_preds)
    test_f1    = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    log.info(f"    Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    save_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    joblib.dump({"model": model, "scaler": None, "feature_cols": feature_cols,
                 "params": params}, save_path)
    log.info(f"    Saved -> {save_path}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    return {"name": "XGBoost", "model": model, "scaler": None,
            "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1,
            "feature_importances": importances}


# ─────────────────────────────────────────────────────────────
# Model 4: LightGBM
# ─────────────────────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_val, y_val,
                    X_test,  y_test,  feature_cols):
    log.info("\n  -- LightGBM --")

    def lgb_fn(**kw):
        return lgb.LGBMClassifier(verbosity=-1, **kw)

    param_grid = {
        "n_estimators":   [100, 200],
        "max_depth":      [5, 10],
        "learning_rate":  [0.05, 0.1],
        "num_leaves":     [31, 63],
        "random_state":   [SEED],
    }

    model, params, val_f1 = grid_search_val(
        lgb_fn, param_grid,
        X_train, y_train, X_val, y_val,
        "LightGBM"
    )

    test_preds = model.predict(X_test)
    test_acc   = accuracy_score(y_test, test_preds)
    test_f1    = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    log.info(f"    Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    save_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    joblib.dump({"model": model, "scaler": None, "feature_cols": feature_cols,
                 "params": params}, save_path)
    log.info(f"    Saved -> {save_path}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    return {"name": "LightGBM", "model": model, "scaler": None,
            "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1,
            "feature_importances": importances}


# ─────────────────────────────────────────────────────────────
# Compare models & pick best
# ─────────────────────────────────────────────────────────────
def compare_models(results: list[dict]) -> dict:
    log.info("\n" + "=" * 60)
    log.info("  MODEL COMPARISON (Test Set)")
    log.info("=" * 60)
    log.info(f"  {'Model':<25} {'Test Acc':>10} {'Test F1':>10}")
    log.info("  " + "-" * 45)

    best = None
    for r in results:
        log.info(f"  {r['name']:<25} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")
        if best is None or r["test_f1"] > best["test_f1"]:
            best = r

    log.info("  " + "-" * 45)
    log.info(f"Best model: {best['name']} (F1={best['test_f1']:.4f})\n")

    # Save best model reference
    best_ref_path = os.path.join(MODELS_DIR, "best_model_info.txt")
    with open(best_ref_path, "w") as f:
        f.write(f"Best Model: {best['name']}\n")
        f.write(f"Test Accuracy: {best['test_acc']:.4f}\n")
        f.write(f"Test F1 Score: {best['test_f1']:.4f}\n")

    return best


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STEP 5: Model Training")
    log.info("=" * 60)

    train_df, val_df, test_df = load_splits()
    feature_cols = get_feature_columns(train_df)

    log.info(f"  Feature columns ({len(feature_cols)}): {feature_cols[:5]} …")
    log.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

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

    best = compare_models(results)

    log.info("Modeling complete! Next step: python src/evaluation.py")
