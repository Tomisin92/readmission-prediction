"""
STEP 6: GENERATE PAPER TABLES & FINAL SUMMARY
================================================
Reads all metrics, fairness results, and SHAP outputs and produces:
  - paper_metrics.json   → all numbers to populate the LaTeX paper
  - cohort_table.csv     → Table 1
  - performance_table.csv → Table 2 (already generated in step 3)
  - fairness_table.csv   → Table 3 (already generated in step 5)
  - waterfall_table.csv  → Table 4
  - A clean printout of everything needed to fill the paper

Run:  python 06_generate_paper_tables.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "data"
MET_DIR   = "outputs/metrics"
FIG_DIR   = "outputs/figures"
OUT_DIR   = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})
NIW_BLUE = "#1F4E79"

# ─── Load everything ─────────────────────────────────────────────────────────
def load_all():
    cohort_stats  = json.load(open(f"{DATA_DIR}/cohort_stats.json"))
    perf          = json.load(open(f"{MET_DIR}/model_performance.json"))
    fairness_gaps = json.load(open(f"{MET_DIR}/fairness_gaps.json"))
    shap_imp      = pd.read_csv(f"{MET_DIR}/shap_global_importance.csv")
    fairness_tbl  = pd.read_csv(f"{MET_DIR}/fairness_table.csv")
    waterfall_tbl = pd.read_csv(f"{MET_DIR}/example_patient_waterfall.csv")
    perf_tbl      = pd.read_csv(f"{MET_DIR}/performance_table.csv")
    cohort        = pd.read_csv(f"{DATA_DIR}/cohort.csv", low_memory=False)
    y_test        = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    meta_test     = pd.read_csv(f"{DATA_DIR}/meta_test.csv")
    return (cohort_stats, perf, fairness_gaps, shap_imp, fairness_tbl,
            waterfall_tbl, perf_tbl, cohort, y_test, meta_test)

# ─── Cohort demographics table ───────────────────────────────────────────────
def build_cohort_table(cohort, y_test, meta_test):
    cohort_sorted = cohort.sort_values("admittime").reset_index(drop=True)
    n = len(cohort_sorted)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = cohort_sorted.iloc[:train_end]
    val_  = cohort_sorted.iloc[train_end:val_end]
    test  = cohort_sorted.iloc[val_end:]

    def stats(df):
        age = df["age"].dropna()
        return {
            "N admissions":          len(df),
            "Readmission rate (%)":  f"{df['readmitted_30d'].mean()*100:.1f}",
            "Median age (IQR)":      f"{age.median():.0f} ({age.quantile(0.25):.0f}–{age.quantile(0.75):.0f})",
            "Female (%)":            f"{(df['gender']=='F').mean()*100:.1f}" if "gender" in df else "N/A",
            "White (%)":             f"{(df['race_simple']=='White').mean()*100:.1f}" if "race_simple" in df else "N/A",
            "Black/AA (%)":          f"{(df['race_simple']=='Black/AA').mean()*100:.1f}" if "race_simple" in df else "N/A",
            "Hispanic (%)":          f"{(df['race_simple']=='Hispanic').mean()*100:.1f}" if "race_simple" in df else "N/A",
            "Medicare (%)":          f"{(df['insurance_simple']=='Medicare').mean()*100:.1f}" if "insurance_simple" in df else "N/A",
            "Median LOS days (IQR)": f"{df['los_days'].median():.1f} ({df['los_days'].quantile(0.25):.1f}–{df['los_days'].quantile(0.75):.1f})" if "los_days" in df else "N/A",
            "Mean Charlson (SD)":    f"{df['charlson_index'].mean():.1f} (±{df['charlson_index'].std():.1f})" if "charlson_index" in df else "N/A",
        }

    tbl = pd.DataFrame({
        "Characteristic": list(stats(train).keys()),
        "Train Set":      list(stats(train).values()),
        "Val Set":        list(stats(val_).values()),
        "Test Set":       list(stats(test).values()),
    })
    tbl.to_csv(f"{MET_DIR}/cohort_table.csv", index=False)
    print("\n[TABLE 1: Cohort Characteristics]")
    print(tbl.to_string(index=False))
    return tbl

# ─── Print everything for paper ──────────────────────────────────────────────
def print_paper_summary(cohort_stats, perf, fairness_gaps, shap_imp,
                         fairness_tbl, waterfall_tbl, perf_tbl):

    print("\n" + "="*70)
    print("  PAPER METRICS SUMMARY — PASTE THESE INTO YOUR LaTeX PAPER")
    print("="*70)

    print("\n── SECTION 3: DATASET ─────────────────────────────────────────────")
    print(f"  Total admissions:       {cohort_stats['n_total']:,}")
    print(f"  Train / Val / Test:     {cohort_stats['n_train']:,} / {cohort_stats['n_val']:,} / {cohort_stats['n_test']:,}")
    print(f"  Readmission rate:       {cohort_stats['readmission_rate_total']:.2f}%")
    print(f"  Train readmission rate: {cohort_stats['readmission_rate_train']:.2f}%")
    print(f"  Test readmission rate:  {cohort_stats['readmission_rate_test']:.2f}%")
    print(f"  Number of features:     {cohort_stats['n_features']}")

    print("\n── SECTION 4: MODEL PERFORMANCE (Table 2) ─────────────────────────")
    print(perf_tbl.to_string(index=False))

    print("\n── SECTION 4: BEST MODEL METRICS (LightGBM) ───────────────────────")
    lgb = perf.get("lightgbm", perf.get("lgb", {}))
    for k, v in lgb.items():
        print(f"  {k}: {v}")

    print("\n── SECTION 4: SHAP TOP FEATURES ───────────────────────────────────")
    print(shap_imp.head(10).to_string(index=False))

    print("\n── SECTION 4: FAIRNESS GAPS ────────────────────────────────────────")
    for group, gaps in fairness_gaps.items():
        flag_a = "⚠️  EXCEEDS THRESHOLD" if gaps["flag_auc"] else "✅ OK"
        flag_f = "⚠️  EXCEEDS THRESHOLD" if gaps["flag_fnr"] else "✅ OK"
        print(f"  {group}:")
        print(f"    Max AUC gap: {gaps['max_auc_gap']:.4f}  {flag_a}")
        print(f"    Max FNR gap: {gaps['max_fnr_gap']:.4f}  {flag_f}")

    print("\n── SECTION 4: EXAMPLE PATIENT WATERFALL ───────────────────────────")
    print(waterfall_tbl.to_string(index=False))

    print("\n── FIGURES GENERATED ───────────────────────────────────────────────")
    for fname in sorted(os.listdir("outputs/figures")):
        print(f"  outputs/figures/{fname}")

    print("\n" + "="*70)
    print("  NEXT STEP: Send this output back to Claude to populate the paper")
    print("="*70)

# ─── Confusion matrix summary figure ─────────────────────────────────────────
def plot_threshold_analysis(y_test, proba_lgb):
    from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
    thresholds = np.linspace(0.05, 0.95, 100)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        pred = (proba_lgb >= t).astype(int)
        f1s.append(f1_score(y_test, pred, zero_division=0))
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s,   color=NIW_BLUE,   lw=2, label="F1 Score")
    ax.plot(thresholds, precs, color="#E67E22",   lw=2, ls="--", label="Precision")
    ax.plot(thresholds, recs,  color="#27AE60",   lw=2, ls=":",  label="Recall")
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, color="red", ls="--", lw=1.5, alpha=0.8,
               label=f"Optimal threshold ({best_t:.2f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis — LightGBM (MIMIC-IV Test Set)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{FIG_DIR}/threshold_analysis.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Save complete paper_metrics.json ────────────────────────────────────────
def save_paper_metrics(cohort_stats, perf, fairness_gaps, shap_imp, fairness_tbl):
    lgb = perf.get("lightgbm", {})
    xgb = perf.get("xgboost", {})
    lr  = perf.get("logistic_regression", {})

    # Top 5 SHAP features
    top5 = shap_imp.head(5)[["feature","mean_abs_shap"]].to_dict("records")

    # Fairness: max gaps
    race_gaps = fairness_gaps.get("Race/Ethnicity", {})

    metrics = {
        "cohort": {
            "n_total":            cohort_stats["n_total"],
            "n_train":            cohort_stats["n_train"],
            "n_val":              cohort_stats["n_val"],
            "n_test":             cohort_stats["n_test"],
            "readmission_rate_pct": cohort_stats["readmission_rate_total"],
            "n_features":         cohort_stats["n_features"],
        },
        "model_performance": {
            "logistic_regression": lr,
            "xgboost":             xgb,
            "lightgbm":            lgb,
        },
        "shap_top5_features": top5,
        "fairness_gaps":       fairness_gaps,
        "paper_placeholders": {
            "[X.XX] LightGBM AUC-ROC":  lgb.get("auc_roc", "TBD"),
            "[X.XX] XGBoost AUC-ROC":   xgb.get("auc_roc", "TBD"),
            "[X.XX] LogReg AUC-ROC":    lr.get("auc_roc",  "TBD"),
            "[X.XX] LightGBM F1":       lgb.get("f1",      "TBD"),
            "[X.XX] LightGBM Precision":lgb.get("precision","TBD"),
            "[X.XX] LightGBM Recall":   lgb.get("recall",  "TBD"),
            "[X.XX] LightGBM Brier":    lgb.get("brier",   "TBD"),
            "[X.XX] AUC CI lo":         lgb.get("auc_roc_ci_lo","TBD"),
            "[X.XX] AUC CI hi":         lgb.get("auc_roc_ci_hi","TBD"),
            "[X] readmission rate %":   cohort_stats["readmission_rate_total"],
            "[N] total admissions":     cohort_stats["n_total"],
            "[N] test admissions":      cohort_stats["n_test"],
            "[X.XX] Max race AUC gap":  race_gaps.get("max_auc_gap","TBD"),
            "[X.XX] Max race FNR gap":  race_gaps.get("max_fnr_gap","TBD"),
        }
    }
    with open(f"{OUT_DIR}/paper_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Full paper metrics saved to {OUT_DIR}/paper_metrics.json")
    return metrics

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (cohort_stats, perf, fairness_gaps, shap_imp, fairness_tbl,
     waterfall_tbl, perf_tbl, cohort, y_test, meta_test) = load_all()

    build_cohort_table(cohort, y_test, meta_test)

    proba_lgb = pd.read_csv(f"{MET_DIR}/proba_lightgbm.csv").squeeze().values
    plot_threshold_analysis(y_test.values, proba_lgb)

    metrics = save_paper_metrics(cohort_stats, perf, fairness_gaps, shap_imp, fairness_tbl)

    print_paper_summary(cohort_stats, perf, fairness_gaps, shap_imp,
                         fairness_tbl, waterfall_tbl, perf_tbl)
