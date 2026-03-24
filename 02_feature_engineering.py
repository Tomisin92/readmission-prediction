"""
STEP 2: FEATURE ENGINEERING & TRAIN/TEST SPLIT
================================================
Builds the full feature matrix from the cohort CSV, handles missing values,
and produces a chronological 70/15/15 train/val/test split.

Run:  python 02_feature_engineering.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "data"
OUT_DIR    = "data"
RANDOM_SEED = 42

# ─── Feature columns to use ───────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Demographics
    "age",
    # Clinical
    "charlson_index", "los_days", "n_diagnoses", "n_procedures",
    "prior_admissions_12mo",
    # Labs
    "creatinine", "egfr", "bnp", "albumin", "wbc",
    "hemoglobin", "sodium", "potassium", "hba1c", "inr",
    # Medications
    "n_medications",
]

BINARY_FEATURES = [
    "gender_enc",        # 1=Male
    "emergency_adm",     # 1=Emergency/Urgent
    "high_risk_med",     # 1=Has high-risk medication
    "polypharmacy",      # 1=>=5 medications
]

CATEGORICAL_FEATURES = [
    "race_simple",
    "insurance_simple",
    "age_group",
]

TARGET = "readmitted_30d"

# ─── Load cohort ──────────────────────────────────────────────────────────────
def load_cohort():
    path = os.path.join(DATA_DIR, "cohort.csv")
    df = pd.read_csv(path, low_memory=False, parse_dates=["admittime","dischtime"])
    print(f"[INFO] Loaded cohort: {len(df):,} admissions")
    return df

# ─── Missingness indicators ───────────────────────────────────────────────────
def add_missingness_flags(df, cols):
    """Add binary flag for each column indicating if value was imputed."""
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df

# ─── One-hot encode categoricals ─────────────────────────────────────────────
def encode_categoricals(df, cat_cols):
    dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=int)
    return pd.concat([df, dummies], axis=1)

# ─── Build feature matrix ────────────────────────────────────────────────────
def build_features(df):
    print("[INFO] Building feature matrix...")

    # Add missingness flags for lab columns
    lab_cols = ["creatinine","egfr","bnp","albumin","wbc",
                "hemoglobin","sodium","potassium","hba1c","inr"]
    df = add_missingness_flags(df, lab_cols)
    miss_cols = [f"{c}_missing" for c in lab_cols if f"{c}_missing" in df.columns]

    # One-hot encode categoricals
    df = encode_categoricals(df, CATEGORICAL_FEATURES)
    cat_dummies = [c for c in df.columns if any(
        c.startswith(cat + "_") for cat in CATEGORICAL_FEATURES)]

    all_feature_cols = NUMERIC_FEATURES + BINARY_FEATURES + miss_cols + cat_dummies

    # Keep only columns that exist
    feature_cols = [c for c in all_feature_cols if c in df.columns]
    missing = [c for c in all_feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing feature columns (will skip): {missing}")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Target distribution: {y.value_counts().to_dict()}")
    print(f"[INFO] Readmission rate: {y.mean()*100:.2f}%")

    return X, y, feature_cols, df

# ─── Chronological train/val/test split ──────────────────────────────────────
def temporal_split(df, X, y):
    """Split by admission time to prevent data leakage."""
    df_sorted = df.sort_values("admittime").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_idx = df_sorted.index[:train_end]
    val_idx   = df_sorted.index[train_end:val_end]
    test_idx  = df_sorted.index[val_end:]

    X_train = X.iloc[train_idx]
    X_val   = X.iloc[val_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_val   = y.iloc[val_idx]
    y_test  = y.iloc[test_idx]

    # Also store metadata for fairness analysis
    meta_cols = ["hadm_id","subject_id","age","age_group","gender",
                 "race_simple","insurance_simple","admittime","dischtime"]
    meta_cols = [c for c in meta_cols if c in df_sorted.columns]
    meta_test = df_sorted.iloc[test_idx][meta_cols].reset_index(drop=True)

    print(f"\n[INFO] Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"[INFO] Train readmission rate: {y_train.mean()*100:.2f}%")
    print(f"[INFO] Val readmission rate:   {y_val.mean()*100:.2f}%")
    print(f"[INFO] Test readmission rate:  {y_test.mean()*100:.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test, meta_test

# ─── Impute and scale ────────────────────────────────────────────────────────
def impute_splits(X_train, X_val, X_test, numeric_cols):
    """Fit median imputer on train, apply to all splits. Returns updated splits."""
    # Only use numeric cols that actually exist in the data
    cols = [c for c in numeric_cols if c in X_train.columns]
    if not cols:
        return X_train, X_val, X_test, None

    # Compute medians from train set only
    medians = X_train[cols].median()

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    for col in cols:
        fill_val = medians[col] if not pd.isna(medians[col]) else 0
        X_train[col] = X_train[col].fillna(fill_val)
        X_val[col]   = X_val[col].fillna(fill_val)
        X_test[col]  = X_test[col].fillna(fill_val)

    # Also save as sklearn imputer for joblib persistence
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[cols])
    return X_train, X_val, X_test, imputer

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_cohort()
    X, y, feature_cols, df_full = build_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test, meta_test = \
        temporal_split(df_full, X, y)

    # Impute numeric features (fit on train only)
    numeric_in_features = [c for c in NUMERIC_FEATURES if c in feature_cols]
    X_train, X_val, X_test, imputer = impute_splits(X_train, X_val, X_test, numeric_in_features)

    # Fill remaining NaNs (binary/categorical dummies) with 0
    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)
    X_test  = X_test.fillna(0)

    # Save
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(OUT_DIR,   "X_val.csv"),   index=False)
    X_test.to_csv(os.path.join(OUT_DIR,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(OUT_DIR,   "y_val.csv"),   index=False)
    y_test.to_csv(os.path.join(OUT_DIR,  "y_test.csv"),  index=False)
    meta_test.to_csv(os.path.join(OUT_DIR, "meta_test.csv"), index=False)
    if imputer is not None:
        joblib.dump(imputer, os.path.join(OUT_DIR, "imputer.pkl"))

    # Save feature list and cohort stats for the paper
    stats = {
        "n_total":        int(len(df)),
        "n_train":        int(len(X_train)),
        "n_val":          int(len(X_val)),
        "n_test":         int(len(X_test)),
        "readmission_rate_total": float(round(y.mean()*100, 2)),
        "readmission_rate_train": float(round(y_train.mean()*100, 2)),
        "readmission_rate_val":   float(round(y_val.mean()*100, 2)),
        "readmission_rate_test":  float(round(y_test.mean()*100, 2)),
        "n_features":     int(len(feature_cols)),
        "feature_cols":   feature_cols,
    }
    with open(os.path.join(OUT_DIR, "cohort_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DONE] All splits saved to {OUT_DIR}/")
    print(json.dumps({k: v for k, v in stats.items() if k != "feature_cols"}, indent=2))