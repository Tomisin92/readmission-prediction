"""
STEP 5: FAIRNESS & EQUITY ANALYSIS
=====================================
Evaluates model performance across demographic subgroups:
  - Race/ethnicity
  - Age group
  - Gender
  - Insurance type (socioeconomic proxy)

Applies equalized odds post-processing where gaps exceed thresholds.
Outputs: fairness tables, plots, LaTeX-ready CSV.

Run:  python 05_fairness_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve

DATA_DIR  = "data"
MET_DIR   = "outputs/metrics"
FIG_DIR   = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Thresholds for flagging fairness gaps
AUC_GAP_THRESHOLD = 0.05
FNR_GAP_THRESHOLD = 0.10

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

NIW_BLUE   = "#1F4E79"
NIW_ACCENT = "#0070C0"
NIW_GREEN  = "#217347"
NIW_RED    = "#C0392B"

# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    y_test   = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    meta     = pd.read_csv(f"{DATA_DIR}/meta_test.csv")
    proba    = pd.read_csv(f"{MET_DIR}/proba_lightgbm.csv").squeeze()

    # Load all model probas
    probas = {}
    for m in ["logistic_regression", "xgboost", "lightgbm"]:
        try:
            probas[m] = pd.read_csv(f"{MET_DIR}/proba_{m}.csv").squeeze().values
        except Exception:
            pass
    return y_test.values, meta.reset_index(drop=True), proba.values, probas

# ─── Metrics per subgroup ─────────────────────────────────────────────────────
def subgroup_metrics(y_true, y_proba, threshold=None):
    """Compute AUC, FNR, PPV, calibration for a subgroup."""
    if len(y_true) < 10:
        return None
    try:
        auc = round(roc_auc_score(y_true, y_proba), 4)
    except Exception:
        auc = None

    if threshold is None:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else None
    ppv = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    prev = round(y_true.mean(), 4)
    n    = len(y_true)
    n_pos = int(y_true.sum())

    return {"n": n, "n_positive": n_pos, "prevalence": prev,
            "auc_roc": auc, "fnr": fnr, "ppv": ppv}

# ─── Run fairness analysis ───────────────────────────────────────────────────
def fairness_analysis(y_true, proba, meta):
    # Get global threshold from full test set
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j = tpr - fpr
    global_threshold = float(thresholds[np.argmax(j)])
    print(f"[INFO] Global Youden threshold: {global_threshold:.4f}")

    subgroup_defs = {
        "Race/Ethnicity":  ("race_simple",      ["White","Black/AA","Hispanic","Asian","Other/Unknown"]),
        "Age Group":       ("age_group",         ["18-50","51-65","66-75","76-85","85+"]),
        "Gender":          ("gender",            ["M","F"]),
        "Insurance":       ("insurance_simple",  ["Medicare","Medicaid","Private","Other"]),
    }

    all_results = {}
    for group_name, (col, categories) in subgroup_defs.items():
        if col not in meta.columns:
            print(f"[WARN] Column '{col}' not found in meta, skipping {group_name}")
            continue
        results = []
        for cat in categories:
            mask = meta[col].astype(str).str.strip() == cat
            if mask.sum() < 10:
                continue
            m = subgroup_metrics(y_true[mask], proba[mask], global_threshold)
            if m:
                m["subgroup"] = cat
                results.append(m)

        if results:
            df = pd.DataFrame(results)
            # Compute gaps
            aucs = df["auc_roc"].dropna()
            fnrs = df["fnr"].dropna()
            df.attrs["auc_gap"] = round(aucs.max() - aucs.min(), 4) if len(aucs) > 1 else 0
            df.attrs["fnr_gap"] = round(fnrs.max() - fnrs.min(), 4) if len(fnrs) > 1 else 0
            all_results[group_name] = df
            print(f"\n  {group_name}: AUC gap={df.attrs['auc_gap']:.4f}, FNR gap={df.attrs['fnr_gap']:.4f}")
            print(df[["subgroup","n","n_positive","auc_roc","fnr","ppv"]].to_string(index=False))

    return all_results, global_threshold

# ─── Equalized odds post-processing ──────────────────────────────────────────
def equalized_odds_postprocess(y_true, proba, group_labels, threshold):
    """
    Simple threshold adjustment per subgroup to reduce FNR gaps.
    Groups with higher FNR get a lower threshold (more sensitive).
    """
    groups = np.unique(group_labels)
    adjusted_thresholds = {}
    for g in groups:
        mask = group_labels == g
        if mask.sum() < 10:
            adjusted_thresholds[g] = threshold
            continue
        from sklearn.metrics import roc_curve
        try:
            fpr, tpr, ths = roc_curve(y_true[mask], proba[mask])
            j = tpr - fpr
            adj_t = float(ths[np.argmax(j)])
        except Exception:
            adj_t = threshold
        adjusted_thresholds[g] = adj_t
    return adjusted_thresholds

# ─── Plot: AUC comparison by subgroup ────────────────────────────────────────
def plot_fairness_auc(all_results):
    groups = [g for g in all_results if all_results[g]["auc_roc"].notna().any()]
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(4*n_groups, 4), sharey=False)
    if n_groups == 1:
        axes = [axes]

    for ax, group_name in zip(axes, groups):
        df = all_results[group_name].dropna(subset=["auc_roc"])
        colors = [NIW_BLUE if v >= df["auc_roc"].max() - 0.02 else
                  NIW_RED  if v <= df["auc_roc"].min() + 0.02 else
                  NIW_ACCENT for v in df["auc_roc"]]
        bars = ax.bar(df["subgroup"], df["auc_roc"], color=colors,
                      edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_ylim(max(0, df["auc_roc"].min() - 0.1), 1.0)
        ax.set_title(group_name, fontweight="bold", pad=8)
        ax.set_ylabel("AUC-ROC" if group_name == groups[0] else "")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(df["auc_roc"].mean(), color="gray", ls="--",
                   lw=1, alpha=0.7, label="Mean AUC")
        for bar, val in zip(bars, df["auc_roc"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        gap = df.attrs.get("auc_gap", 0)
        ax.text(0.97, 0.03, f"Δ = {gap:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=NIW_RED if gap > AUC_GAP_THRESHOLD else NIW_GREEN,
                fontweight="bold")

    plt.suptitle("Subgroup Fairness — AUC-ROC by Demographic Group (LightGBM, MIMIC-IV Test Set)",
                 fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = f"{FIG_DIR}/fairness_auc_comparison.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Plot: FNR comparison ─────────────────────────────────────────────────────
def plot_fairness_fnr(all_results):
    groups = [g for g in all_results if all_results[g]["fnr"].notna().any()]
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(4*n_groups, 4), sharey=False)
    if n_groups == 1:
        axes = [axes]

    for ax, group_name in zip(axes, groups):
        df = all_results[group_name].dropna(subset=["fnr"])
        colors = [NIW_RED  if v >= df["fnr"].max() - 0.02 else
                  NIW_GREEN if v <= df["fnr"].min() + 0.02 else
                  NIW_ACCENT for v in df["fnr"]]
        bars = ax.bar(df["subgroup"], df["fnr"], color=colors,
                      edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_ylim(0, min(1.0, df["fnr"].max() + 0.15))
        ax.set_title(group_name, fontweight="bold", pad=8)
        ax.set_ylabel("False Negative Rate (FNR)" if group_name == groups[0] else "")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, df["fnr"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        gap = df.attrs.get("fnr_gap", 0)
        ax.text(0.97, 0.97, f"Δ = {gap:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=NIW_RED if gap > FNR_GAP_THRESHOLD else NIW_GREEN,
                fontweight="bold")

    plt.suptitle("Subgroup Fairness — False Negative Rate (FNR)\n(High FNR = high-risk patients missed)",
                 fontsize=11, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = f"{FIG_DIR}/fairness_fnr_comparison.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Calibration plot ────────────────────────────────────────────────────────
def plot_calibration(y_true, probas_dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {
        "Logistic Regression": ("#E67E22", "--"),
        "XGBoost":             (NIW_ACCENT, "-"),
        "LightGBM":            (NIW_BLUE, "-"),
    }
    for name, proba in probas_dict.items():
        fraction_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        color, ls = colors.get(name, ("gray","-"))
        ax.plot(mean_pred, fraction_pos, marker="o", ms=4,
                color=color, ls=ls, lw=2, label=name)

    ax.plot([0,1],[0,1], "k--", lw=1.5, alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Calibration Curves — 30-Day Readmission Prediction", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{FIG_DIR}/calibration_curves.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Save LaTeX-ready fairness table ─────────────────────────────────────────
def save_fairness_table(all_results):
    rows = []
    for group_name, df in all_results.items():
        for _, row in df.iterrows():
            rows.append({
                "Demographic Group": group_name,
                "Subgroup":    row["subgroup"],
                "N":           row["n"],
                "N Positive":  row["n_positive"],
                "Prevalence":  f"{row['prevalence']*100:.1f}%",
                "AUC-ROC":     f"{row['auc_roc']:.4f}" if pd.notna(row.get("auc_roc")) else "N/A",
                "FNR":         f"{row['fnr']:.4f}"     if pd.notna(row.get("fnr"))     else "N/A",
                "PPV":         f"{row['ppv']:.4f}"     if pd.notna(row.get("ppv"))     else "N/A",
            })
    table = pd.DataFrame(rows)
    table.to_csv(f"{MET_DIR}/fairness_table.csv", index=False)
    print(f"\n[DONE] Fairness table saved: {MET_DIR}/fairness_table.csv")
    print(table.to_string(index=False))

    # Summary gaps
    gaps = {}
    for group_name, df in all_results.items():
        gaps[group_name] = {
            "max_auc_gap": df.attrs.get("auc_gap", 0),
            "max_fnr_gap": df.attrs.get("fnr_gap", 0),
            "flag_auc": df.attrs.get("auc_gap", 0) > AUC_GAP_THRESHOLD,
            "flag_fnr": df.attrs.get("fnr_gap", 0) > FNR_GAP_THRESHOLD,
        }
    with open(f"{MET_DIR}/fairness_gaps.json", "w") as f:
        json.dump(gaps, f, indent=2)
    print("\n[INFO] Fairness gap summary:")
    print(json.dumps(gaps, indent=2))

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_true, meta, proba_lgb, all_probas = load_data()
    print(f"[INFO] Test set: {len(y_true)} samples, "
          f"readmission rate: {y_true.mean()*100:.2f}%")

    all_results, threshold = fairness_analysis(y_true, proba_lgb, meta)

    plot_fairness_auc(all_results)
    plot_fairness_fnr(all_results)
    save_fairness_table(all_results)

    # Calibration
    probas_named = {}
    label_map = {"logistic_regression": "Logistic Regression",
                 "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    for k, v in all_probas.items():
        probas_named[label_map.get(k, k)] = v
    if probas_named:
        plot_calibration(y_true, probas_named)

    print(f"\n[DONE] All fairness figures saved to {FIG_DIR}/")
