"""
RUN ALL — Master Pipeline Script
==================================
Runs the complete readmission prediction pipeline in order:
  Step 1: Download MIMIC-IV demo + build cohort
  Step 2: Feature engineering + train/val/test split
  Step 3: Train models (LogReg, XGBoost, LightGBM)
  Step 4: SHAP explainability + plots
  Step 5: Fairness analysis + plots
  Step 6: Generate all paper tables + final summary

Usage:
  python run_all.py

After this completes, copy-paste the PAPER METRICS SUMMARY
output and send it to Claude to populate the LaTeX paper.
"""

import subprocess
import sys
import time
import os

STEPS = [
    ("01_data_acquisition.py",   "Data Acquisition & Cohort Building"),
    ("02_feature_engineering.py","Feature Engineering & Train/Val/Test Split"),
    ("03_train_models.py",       "Model Training (LogReg, XGBoost, LightGBM)"),
    ("04_shap_analysis.py",      "SHAP Explainability Analysis"),
    ("05_fairness_analysis.py",  "Fairness & Equity Analysis"),
    ("06_generate_paper_tables.py","Paper Tables & Final Summary"),
]

def run_step(script, name, step_num, total):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/{total}: {name}")
    print(f"  Script: {script}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} FAILED after {elapsed:.1f}s")
        print(f"  Return code: {result.returncode}")
        return False
    else:
        print(f"\n[OK] Step {step_num} completed in {elapsed:.1f}s")
        return True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  NIW PAPER — COMPLETE PREDICTION PIPELINE")
    print("  Transparent and Observable Predictive AI for Hospital Readmission")
    print("="*70)
    print(f"\n  Working directory: {os.getcwd()}")
    print(f"  Python: {sys.executable}")
    print(f"\n  Steps to run: {len(STEPS)}")

    # Check dependencies
    print("\n[INFO] Checking dependencies...")
    missing = []
    for pkg in ["pandas","numpy","sklearn","xgboost","lightgbm","shap",
                "matplotlib","scipy","joblib","tqdm","requests"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing packages: {missing}")
        print(f"  Install with:  pip install {' '.join(missing)}")
        print(f"  Or:            pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("[OK] All dependencies found.")

    t_start = time.time()
    for i, (script, name) in enumerate(STEPS, 1):
        ok = run_step(script, name, i, len(STEPS))
        if not ok:
            print(f"\n[FATAL] Pipeline stopped at step {i}. Fix the error and re-run.")
            sys.exit(1)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL {len(STEPS)} STEPS COMPLETED SUCCESSFULLY")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    print("""
  OUTPUT FILES:
    outputs/metrics/paper_metrics.json     ← All numbers for the paper
    outputs/metrics/cohort_table.csv       ← Table 1
    outputs/metrics/performance_table.csv  ← Table 2
    outputs/metrics/fairness_table.csv     ← Table 3
    outputs/metrics/example_patient_waterfall.csv ← Table 4
    outputs/metrics/shap_global_importance.csv    ← SHAP feature table
    outputs/figures/shap_global_importance.png    ← Figure 1
    outputs/figures/shap_beeswarm.png             ← Figure 2
    outputs/figures/shap_waterfall.png            ← Figure 3
    outputs/figures/roc_prc_curves.png            ← Figure 4
    outputs/figures/fairness_auc_comparison.png   ← Figure 5
    outputs/figures/fairness_fnr_comparison.png   ← Figure 6
    outputs/figures/calibration_curves.png        ← Figure 7
    outputs/figures/threshold_analysis.png        ← Figure 8

  NEXT STEP:
    Copy the PAPER METRICS SUMMARY printed above and paste it
    into Claude to automatically populate the LaTeX paper.
""")
