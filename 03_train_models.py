"""
STEP 3: MODEL TRAINING
========================
Trains three models:
  1. Logistic Regression (L2, baseline)
  2. XGBoost
  3. LightGBM
Each with Bayesian hyperparameter optimization on the validation set.

Run:  python 03_train_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     f1_score, precision_score, recall_score,
                                     brier_score_loss, roc_curve,
                                     precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import StandardScaler

import xgboost as xgb
import lightgbm as lgb

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUT_DIR    = "outputs/metrics"
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_val   = pd.read_csv(f"{DATA_DIR}/X_val.csv")
    X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{DATA_DIR}/y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

# ─── Metrics helper ──────────────────────────────────────────────────────────
def youden_threshold(y_true, y_proba):
    """Find threshold that maximises Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return thresholds[np.argmax(j)]

def compute_metrics(y_true, y_proba, threshold=None):
    if threshold is None:
        threshold = youden_threshold(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc":   round(float(roc_auc_score(y_true, y_proba)),   4),
        "auc_prc":   round(float(average_precision_score(y_true, y_proba)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)),         4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)),      4),
        "brier":     round(float(brier_score_loss(y_true, y_proba)), 4),
        "threshold": round(float(threshold), 4),
    }

def bootstrap_ci(y_true, y_proba, metric_fn, n_boot=1000, seed=RANDOM_SEED):
    """Bootstrap 95% CI for a scalar metric."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true.iloc[idx], y_proba[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return round(lo, 4), round(hi, 4)

# ─── 1. Logistic Regression ───────────────────────────────────────────────────
def train_logreg(X_train, y_train, X_val, y_val):
    print("\n[MODEL 1] Training Logistic Regression...")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    best_auc, best_C, best_model = 0, 1.0, None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_SEED,
                                 class_weight="balanced", solver="lbfgs")
        clf.fit(X_tr_sc, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_va_sc)[:,1])
        print(f"  C={C:.3f}  val_AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_C, best_model = auc, C, clf

    print(f"  Best C={best_C}  val_AUC={best_auc:.4f}")
    joblib.dump({"model": best_model, "scaler": scaler}, f"{MODEL_DIR}/logreg.pkl")
    return best_model, scaler

# ─── 2. XGBoost ───────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n[MODEL 2] Training XGBoost with grid search...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n,
         "subsample": ss, "colsample_bytree": cs}
        for d  in [4, 6]
        for lr in [0.05, 0.1]
        for n  in [300, 500]
        for ss in [0.8]
        for cs in [0.8]
    ]

    best_auc, best_params, best_model = 0, None, None
    for i, params in enumerate(param_grid):
        clf = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(param_grid)}] current best val_AUC={best_auc:.4f}")
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, clf

    print(f"  Best params: {best_params}")
    print(f"  Best val_AUC: {best_auc:.4f}")
    joblib.dump(best_model, f"{MODEL_DIR}/xgboost.pkl")
    return best_model

# ─── 3. LightGBM ──────────────────────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n[MODEL 3] Training LightGBM with grid search...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    param_grid = [
        {"num_leaves": nl, "learning_rate": lr, "n_estimators": n,
         "subsample": ss, "colsample_bytree": cs, "min_child_samples": mc}
        for nl in [31, 63]
        for lr in [0.05, 0.1]
        for n  in [300, 500]
        for ss in [0.8]
        for cs in [0.8]
        for mc in [20]
    ]

    best_auc, best_params, best_model = 0, None, None
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

    for i, params in enumerate(param_grid):
        clf = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(param_grid)}] current best val_AUC={best_auc:.4f}")
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, clf

    print(f"  Best params: {best_params}")
    print(f"  Best val_AUC: {best_auc:.4f}")
    joblib.dump(best_model, f"{MODEL_DIR}/lightgbm.pkl")
    return best_model

# ─── Evaluate all models on test set ────────────────────────────────────────
def evaluate_all(models_dict, X_test, y_test):
    print("\n[EVAL] Computing test set metrics with 95% bootstrap CIs...")
    results = {}
    for name, (model, extra) in models_dict.items():
        if extra is not None:  # logreg has scaler
            X_t = extra.transform(X_test)
        else:
            X_t = X_test
        proba = model.predict_proba(X_t)[:,1]
        metrics = compute_metrics(y_test, proba)

        # Bootstrap CI for AUC-ROC
        ci_lo, ci_hi = bootstrap_ci(y_test, proba, roc_auc_score)
        metrics["auc_roc_ci_lo"] = ci_lo
        metrics["auc_roc_ci_hi"] = ci_hi

        results[name] = metrics

        # Save probabilities for fairness + SHAP analysis
        pd.Series(proba, name="proba").to_csv(
            f"{OUT_DIR}/proba_{name}.csv", index=False)

        print(f"\n  {name.upper()}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    # Save combined results
    with open(f"{OUT_DIR}/model_performance.json", "w") as f:
        json.dump(results, f, indent=2)

    # Pretty table
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":     name,
            "AUC-ROC":   f"{m['auc_roc']:.4f} ({m['auc_roc_ci_lo']:.4f}–{m['auc_roc_ci_hi']:.4f})",
            "AUC-PRC":   f"{m['auc_prc']:.4f}",
            "F1":        f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "Brier":     f"{m['brier']:.4f}",
        })
    table = pd.DataFrame(rows)
    table.to_csv(f"{OUT_DIR}/performance_table.csv", index=False)
    print(f"\n[DONE] Performance table:\n{table.to_string(index=False)}")
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"[INFO] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    logreg, scaler = train_logreg(X_train, y_train, X_val, y_val)
    xgb_model      = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model      = train_lightgbm(X_train, y_train, X_val, y_val)

    models_dict = {
        "logistic_regression": (logreg, scaler),
        "xgboost":             (xgb_model, None),
        "lightgbm":            (lgb_model, None),
    }

    results = evaluate_all(models_dict, X_test, y_test)
    print(f"\n[DONE] All models saved to {MODEL_DIR}/")
    print(f"[DONE] Metrics saved to {OUT_DIR}/")
