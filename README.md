# 🏥 Transparent & Observable Predictive AI for Hospital Readmission Risk

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0.696_AUC--ROC-FF6600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-SHAP_Explainability-2ECC71?style=for-the-badge)
![MIMIC-IV](https://img.shields.io/badge/MIMIC--IV-415K_Admissions-1F4E79?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Isaac Tosin Adisa** · Department of Statistics · Florida State University  
`ita24@fsu.edu`

[📄 Paper (arXiv)](#) · [📊 Results](#results) · [🚀 Quickstart](#quickstart) · [🗂️ Project Structure](#project-structure)

---

> *A production-grade, explainable, and demographically fair machine learning system for predicting 30-day hospital readmission risk — trained on 415,231 MIMIC-IV admissions with full SHAP explainability and a designed observability stack.*

</div>

---

## 📌 Overview

Hospital readmissions within 30 days of discharge cost the U.S. healthcare system over **$26 billion annually** and affect ~3.8 million Medicare beneficiaries each year. Despite decades of research, fewer than 15% of U.S. hospitals deploy any AI-based readmission tool in routine clinical workflows.

This project addresses the three main barriers to clinical translation:

| Barrier | This Project's Solution |
|---|---|
| 🔲 Black-box predictions | Per-patient SHAP waterfall explanations via LightGBM TreeExplainer |
| 🔲 No production reliability | Designed observability stack: Prometheus + Grafana + AKS + SLOs |
| 🔲 Ignored fairness | Full demographic audit across 16 subgroups — no post-processing needed |

### Alignment with U.S. National Policy
- ✅ **CMS Hospital Readmissions Reduction Program (HRRP)** — direct penalty-reduction tool
- ✅ **White House EO on Safe & Trustworthy AI (Oct 2023)** — transparent, auditable clinical AI
- ✅ **ONC TEFCA** — FHIR-compatible API design for national health exchange integration

---

## 📊 Results

### Model Performance (MIMIC-IV Test Set, n = 62,285)

| Model | AUC-ROC (95% CI) | AUC-PRC | F1 | Recall | Brier Score |
|---|---|---|---|---|---|
| Logistic Regression | 0.675 (0.669–0.680) | 0.326 | 0.381 | 0.599 | 0.224 |
| **XGBoost** ⭐ | **0.696 (0.691–0.701)** | **0.346** | **0.394** | **0.641** | 0.217 |
| LightGBM | 0.689 (0.684–0.695) | 0.333 | 0.390 | 0.612 | **0.146** |

> ⭐ XGBoost achieves best discrimination. LightGBM achieves best calibration (Brier 0.146) and is the **deployed model** for real-time SHAP explanations.

### Top SHAP Features (LightGBM, mean |φ| across test set)

```
Prior Admissions (12 mo)   ████████████████████  0.085
Number of Medications      █████                 0.020
Number of Diagnoses        ████                  0.018
Length of Stay (days)      ████                  0.014
Number of Procedures       ███                   0.011
Age                        ██                    0.007
Charlson Comorbidity Index ██                    0.005
Emergency Admission        █                     0.003
```

### Fairness Evaluation (Zero post-processing required ✅)

| Demographic Dimension | Max ΔAUC | Max ΔFNR | Flag |
|---|---|---|---|
| Race / Ethnicity | 0.011 | 0.034 | ✅ OK |
| Age Group | 0.012 | 0.016 | ✅ OK |
| Gender | 0.001 | 0.006 | ✅ OK |
| Insurance Type | 0.030 | 0.032 | ✅ OK |

All 16 subgroups pass equity thresholds (ΔAUC ≤ 0.05, ΔFNR ≤ 0.10) without threshold adjustment.

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Tomisin92/readmission-prediction.git
cd readmission-prediction
pip install -r requirements.txt
```

### 2. Download MIMIC-IV

Access requires free credentialing via [PhysioNet](https://physionet.org/content/mimiciv/). Once approved:

```bash
# Place downloaded MIMIC-IV files under:
data/mimic-iv/hosp
  ├── hosp/admissions.csv.gz
  ├── hosp/patients.csv.gz
  ├── hosp/diagnoses_icd.csv.gz
  ├── hosp/labevents.csv.gz
  └── hosp/prescriptions.csv.gz
```

### 3. Run the Full Pipeline

```bash
python run_all.py
```

This executes all 6 steps sequentially (~45 minutes on a standard laptop):

```
Step 1/6  Data Acquisition & Cohort Building      ~37 min
Step 2/6  Feature Engineering & Train/Val/Test     ~22 sec
Step 3/6  Model Training (LogReg, XGBoost, LGBM)   ~5 min
Step 4/6  SHAP Explainability Analysis             ~79 sec
Step 5/6  Fairness & Equity Analysis               ~12 sec
Step 6/6  Paper Tables & Final Summary             ~24 sec
```

### 4. Run Individual Steps

```bash
python 01_data_acquisition.py
python 02_feature_engineering.py
python 03_train_models.py
python 04_shap_analysis.py
python 05_fairness_analysis.py
python 06_generate_paper_tables.py
```

---

## 🗂️ Project Structure

```
readmission-prediction/
│
├── run_all.py                        # Master pipeline runner
│
├── 01_data_acquisition.py            # Cohort building from MIMIC-IV
├── 02_feature_engineering.py         # Feature matrix + train/val/test split
├── 03_train_models.py                # LogReg, XGBoost, LightGBM + tuning
├── 04_shap_analysis.py               # SHAP TreeExplainer + figures
├── 05_fairness_analysis.py           # Subgroup equity evaluation
├── 06_generate_paper_tables.py       # LaTeX-ready tables + metrics JSON
│
├── data/
│   ├── mimic-iv/                     # Raw MIMIC-IV (not tracked by git)
│   └── cohort.csv                    # Built cohort (415K admissions)
│
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── logistic_regression_model.pkl
│
├── outputs/
│   ├── figures/                      # All paper figures (PDF + PNG)
│   │   ├── shap_global_importance.*
│   │   ├── shap_beeswarm.*
│   │   ├── shap_waterfall.*
│   │   ├── roc_prc_curves.*
│   │   ├── fairness_auc_comparison.*
│   │   ├── fairness_fnr_comparison.*
│   │   ├── calibration_curves.*
│   │   └── threshold_analysis.*
│   └── metrics/
│       ├── paper_metrics.json
│       ├── cohort_table.csv
│       ├── performance_table.csv
│       ├── fairness_table.csv
│       └── shap_global_importance.csv
│
├── paper/
│   └── readmission_paper.tex         # Full LaTeX manuscript
│
└── requirements.txt
```

---

## 🏗️ Planned Deployment Architecture

The production deployment stack is fully specified and constitutes the next phase of this work. Target architecture:

```
                        ┌─────────────────────────────────┐
                        │     Azure Kubernetes Service      │
                        │  ┌──────────┐  ┌─────────────┐  │
   EHR / FHIR  ──────▶  │  │ FastAPI  │  │  LightGBM   │  │
   Input Data           │  │ /predict │  │  + SHAP     │  │
                        │  │ /explain │  │  TreeExp.   │  │
                        │  └────┬─────┘  └─────────────┘  │
                        └───────┼─────────────────────────-┘
                                │
                    ┌───────────▼────────────┐
                    │  Prometheus + Grafana   │
                    │  SLO Monitoring         │
                    │  Drift Detection        │
                    └────────────────────────┘
```

**Target SLOs:**
- System availability: ≥ 99.9%
- p99 prediction latency: ≤ 200 ms
- p99 SHAP latency: ≤ 200 ms
- Error rate: ≤ 0.1%
- Prediction drift: ≤ 2σ from 30-day baseline

---

## 📦 Requirements

```
python>=3.9
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
matplotlib
seaborn
hyperopt
imbalanced-learn
fastapi
uvicorn
prometheus-client
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📄 Citation

If you use this code or dataset pipeline, please cite:

```bibtex
@article{adisa2025readmission,
  title   = {Transparent and Observable Predictive AI for Hospital
             Readmission Risk in U.S. Healthcare Systems},
  author  = {Adisa, Isaac Tosin},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## 🔒 Data & Ethics

- **MIMIC-IV** data access requires credentialing via [PhysioNet](https://physionet.org). Raw data is **not included** in this repository.
- All analyses use de-identified data in compliance with PhysioNet Data Use Agreement.
- Fairness evaluation follows the equalized odds framework of [Hardt et al., NeurIPS 2016].

---

## 📬 Contact

**Isaac Tosin Adisa**  
Department of Statistics, Florida State University  
📧 ita24@fsu.edu

---

<div align="center">
<sub>Built with MIMIC-IV · XGBoost · LightGBM · SHAP · FastAPI · Prometheus · Grafana · Azure Kubernetes Service</sub>
</div>