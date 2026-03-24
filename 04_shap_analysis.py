"""
STEP 4: SHAP EXPLAINABILITY
=============================
Computes SHAP values for the best model (LightGBM) and generates:
  - Global feature importance table (mean |SHAP|)
  - Beeswarm summary plot
  - Waterfall plot for an example high-risk patient
  - All outputs saved for LaTeX paper

Run:  python 04_shap_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUT_DIR    = "outputs"
FIG_DIR    = "outputs/figures"
MET_DIR    = "outputs/metrics"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

NIW_BLUE   = "#1F4E79"
NIW_ACCENT = "#0070C0"
NIW_GREEN  = "#217347"
NIW_RED    = "#C0392B"

# ─── Load ────────────────────────────────────────────────────────────────────
def load_all():
    X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    model   = joblib.load(f"{MODEL_DIR}/lightgbm.pkl")
    proba   = pd.read_csv(f"{MET_DIR}/proba_lightgbm.csv").squeeze()
    return X_test, y_test, model, proba.values

# ─── Compute SHAP values ─────────────────────────────────────────────────────
def compute_shap(model, X_test):
    print("[INFO] Computing SHAP values with TreeExplainer...")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)

    # LightGBM may return list [class0, class1] or single array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class 1 = readmitted

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    print(f"[INFO] SHAP values shape: {shap_vals.shape}")
    print(f"[INFO] Base value (expected value): {base_value:.4f}")
    return shap_vals, float(base_value), explainer

# ─── Global importance ────────────────────────────────────────────────────────
def global_importance(shap_vals, X_test, top_k=15):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance = pd.DataFrame({
        "feature":   X_test.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).head(top_k)
    importance.to_csv(f"{MET_DIR}/shap_global_importance.csv", index=False)
    print(f"\n[INFO] Top {top_k} features by mean |SHAP|:")
    print(importance.to_string(index=False))
    return importance

# ─── Clean feature name ───────────────────────────────────────────────────────
def clean_name(name):
    return (name.replace("_", " ")
                .replace("charlson index", "Charlson Index")
                .replace("prior admissions 12mo", "Prior admissions (12 mo)")
                .replace("los days", "Length of stay (days)")
                .replace("n diagnoses", "# Diagnoses")
                .replace("n procedures", "# Procedures")
                .replace("n medications", "# Medications")
                .replace("emergency adm", "Emergency admission")
                .replace("high risk med", "High-risk medication")
                .replace("polypharmacy", "Polypharmacy (≥5 meds)")
                .replace("gender enc", "Male gender")
                .title())

# ─── Plot 1: Global SHAP bar chart ───────────────────────────────────────────
def plot_global_importance(importance):
    fig, ax = plt.subplots(figsize=(8, 5))
    feats  = [clean_name(f) for f in importance["feature"]]
    vals   = importance["mean_abs_shap"].values

    colors = [NIW_BLUE if i < 5 else NIW_ACCENT if i < 10 else "#7FB3D3"
              for i in range(len(feats))]

    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Mean |SHAP value| (impact on model output)", labelpad=8)
    ax.set_title("Global Feature Importance — LightGBM\n(Mean absolute SHAP across test set)",
                 fontweight="bold", pad=12)

    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color="#333333")

    patch1 = mpatches.Patch(color=NIW_BLUE,   label="Top 5 features")
    patch2 = mpatches.Patch(color=NIW_ACCENT, label="Features 6–10")
    patch3 = mpatches.Patch(color="#7FB3D3",  label="Features 11–15")
    ax.legend(handles=[patch1, patch2, patch3], loc="lower right", fontsize=9)

    ax.set_xlim(0, vals.max() * 1.18)
    plt.tight_layout()
    path = f"{FIG_DIR}/shap_global_importance.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Plot 2: Beeswarm summary ────────────────────────────────────────────────
def plot_beeswarm(shap_vals, X_test, top_k=15):
    top_features = (np.abs(shap_vals).mean(axis=0)
                    .argsort()[::-1][:top_k])

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_vals[:, top_features],
        X_test.iloc[:, top_features],
        feature_names=[clean_name(X_test.columns[i]) for i in top_features],
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
        alpha=0.6,
    )
    plt.title("SHAP Beeswarm Summary — LightGBM\n(Each dot = one patient, colored by feature value)",
              fontweight="bold", pad=12)
    plt.tight_layout()
    path = f"{FIG_DIR}/shap_beeswarm.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Plot 3: Waterfall for a high-risk patient ───────────────────────────────
def plot_waterfall(shap_vals, X_test, proba, base_value, top_k=10):
    # Pick a genuinely high-risk patient (predicted prob in 70-90th pctile)
    hi_idx  = np.where((proba >= np.percentile(proba, 70)) &
                        (proba <= np.percentile(proba, 90)))[0]
    pat_idx = hi_idx[len(hi_idx)//2]  # median of high-risk group

    sv = shap_vals[pat_idx]
    feat_names = np.array(X_test.columns)
    feat_vals  = X_test.iloc[pat_idx].values

    # Sort by abs SHAP
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    sorted_names  = [clean_name(feat_names[i]) for i in order]
    sorted_shap   = sv[order]
    sorted_fvals  = feat_vals[order]

    # Build waterfall
    fig, ax = plt.subplots(figsize=(9, 6))
    cumulative = base_value
    bar_data = []
    for i, (name, s, fv) in enumerate(zip(sorted_names, sorted_shap, sorted_fvals)):
        bar_data.append((name, cumulative, s, fv))
        cumulative += s

    # Plot bars
    for i, (name, start, delta, fv) in enumerate(reversed(bar_data)):
        color = NIW_BLUE if delta > 0 else NIW_GREEN
        ax.barh(i, delta, left=start, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.8, height=0.6)
        label = f"= {fv:.2f}" if isinstance(fv, float) and not np.isnan(fv) else f"= {fv}"
        ax.text(start + delta + (0.002 if delta > 0 else -0.002),
                i, f"{delta:+.4f}  [{label}]",
                va="center", ha="left" if delta > 0 else "right",
                fontsize=8.5, color="#222222")

    y_labels = [b[0] for b in reversed(bar_data)]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)

    pred_prob = float(np.clip(cumulative, 0, 1))
    ax.axvline(base_value, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("SHAP value (contribution to log-odds)", labelpad=8)
    ax.set_title(
        f"Patient-Level SHAP Waterfall\n"
        f"Predicted readmission risk: {pred_prob*100:.1f}%  "
        f"(baseline: {(1/(1+np.exp(-base_value)))*100:.1f}%)",
        fontweight="bold", pad=12)

    pos_patch = mpatches.Patch(color=NIW_BLUE,  label="Increases risk")
    neg_patch = mpatches.Patch(color=NIW_GREEN, label="Decreases risk")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    path = f"{FIG_DIR}/shap_waterfall.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

    # Save patient data for paper table
    waterfall_data = []
    for name, start, delta, fv in bar_data:
        waterfall_data.append({"Feature": name, "Value": fv, "SHAP": round(delta, 4)})
    waterfall_data.append({"Feature": "Baseline (expected value)", "Value": "",
                            "SHAP": round(base_value, 4)})
    waterfall_data.append({"Feature": "Predicted risk (probability)", "Value": "",
                            "SHAP": round(pred_prob, 4)})
    pd.DataFrame(waterfall_data).to_csv(f"{MET_DIR}/example_patient_waterfall.csv", index=False)

# ─── Plot 4: ROC + PRC curves ────────────────────────────────────────────────
def plot_roc_prc(y_test, models_probas):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "Logistic Regression": ("#E67E22", "--"),
        "XGBoost":             (NIW_ACCENT, "-"),
        "LightGBM":            (NIW_BLUE, "-"),
    }

    for name, proba in models_probas.items():
        color, ls = colors.get(name, ("gray", "-"))
        # ROC
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, ls=ls, lw=2,
                     label=f"{name} (AUC={roc_auc:.4f})")
        # PRC
        prec, rec, _ = precision_recall_curve(y_test, proba)
        prc_auc = auc(rec, prec)
        axes[1].plot(rec, prec, color=color, ls=ls, lw=2,
                     label=f"{name} (AUC={prc_auc:.4f})")

    # ROC formatting
    axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — 30-Day Readmission", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_xlim([-0.01, 1.01])
    axes[0].set_ylim([-0.01, 1.01])

    # PRC formatting
    baseline = y_test.mean()
    axes[1].axhline(baseline, color="gray", ls="--", lw=1,
                    label=f"Baseline (prevalence={baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves — 30-Day Readmission", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].set_xlim([-0.01, 1.01])
    axes[1].set_ylim([-0.01, 1.01])

    for ax in axes:
        ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.suptitle("Model Evaluation Curves (MIMIC-IV Test Set)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{FIG_DIR}/roc_prc_curves.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf",".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_test, y_test, model, proba_lgb = load_all()

    shap_vals, base_value, explainer = compute_shap(model, X_test)

    importance = global_importance(shap_vals, X_test, top_k=15)
    plot_global_importance(importance)
    plot_beeswarm(shap_vals, X_test, top_k=12)
    plot_waterfall(shap_vals, X_test, proba_lgb, base_value, top_k=10)

    # Load all model probas for ROC/PRC
    models_probas = {}
    for mname, label in [("logistic_regression","Logistic Regression"),
                          ("xgboost","XGBoost"),
                          ("lightgbm","LightGBM")]:
        p = pd.read_csv(f"{MET_DIR}/proba_{mname}.csv").squeeze().values
        models_probas[label] = p

    plot_roc_prc(y_test, models_probas)

    # Save SHAP values for downstream use
    pd.DataFrame(shap_vals, columns=X_test.columns).to_csv(
        f"{MET_DIR}/shap_values.csv", index=False)

    print(f"\n[DONE] All SHAP figures saved to {FIG_DIR}/")
    print(f"[DONE] SHAP metrics saved to {MET_DIR}/")
