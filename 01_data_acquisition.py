"""
STEP 1: DATA ACQUISITION & COHORT BUILDING
============================================
Downloads the MIMIC-IV Clinical Database Demo (freely available, no credentialing needed)
from PhysioNet and builds the 30-day readmission cohort.

For FULL MIMIC-IV (requires PhysioNet credentialing):
  - Sign up at https://physionet.org/register/
  - Request access to https://physionet.org/content/mimiciv/
  - Set USE_FULL_MIMIC = True and FULL_MIMIC_DIR to your hosp folder below

Run:  python 01_data_acquisition.py
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────
USE_FULL_MIMIC  = True
MIMIC_FULL_PATH = r"C:\Users\PALMPAY\Downloads\readmission_pipeline\physionet.org\files\mimiciv\3.1"
FULL_MIMIC_DIR  = r"C:\Users\PALMPAY\Downloads\readmission_pipeline\physionet.org\files\mimiciv\3.1\hosp"
DEMO_DIR        = "data/mimic_demo"
OUT_DIR         = "data"
RANDOM_SEED     = 42

DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"

os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ─── Download MIMIC-IV Demo ───────────────────────────────────────────────────
def download_demo():
    zip_path = os.path.join(DEMO_DIR, "mimic_demo.zip")
    if os.path.exists(os.path.join(DEMO_DIR, "hosp")):
        print("[INFO] MIMIC-IV demo already downloaded.")
        return
    print("[INFO] Downloading MIMIC-IV demo from PhysioNet...")
    r = requests.get(DEMO_URL, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print("[INFO] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEMO_DIR)
    print("[INFO] Done.")

# ─── Load tables ─────────────────────────────────────────────────────────────
def load_tables(base_dir):
    """
    Load core MIMIC-IV tables.
    - For full MIMIC-IV: base_dir is already the hosp folder
    - For demo:          base_dir is the extracted root, hosp is a subfolder
    """
    if USE_FULL_MIMIC:
        hosp = Path(base_dir)                    # already points to hosp/
        icu  = Path(base_dir).parent / "icu"
    else:
        hosp = Path(base_dir) / "hosp"
        icu  = Path(base_dir) / "icu"

    print("[INFO] Loading admissions...")
    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime"],
        low_memory=False
    )

    print("[INFO] Loading patients...")
    patients = pd.read_csv(hosp / "patients.csv.gz", low_memory=False)

    print("[INFO] Loading diagnoses...")
    diagnoses = pd.read_csv(hosp / "diagnoses_icd.csv.gz", low_memory=False)

    print("[INFO] Loading labevents (this may take a moment)...")
    try:
        labevents = pd.read_csv(
            hosp / "labevents.csv.gz",
            usecols=["subject_id", "hadm_id", "itemid", "valuenum", "flag"],
            low_memory=False
        )
    except Exception as e:
        print(f"[WARN] Could not load labevents: {e}")
        labevents = pd.DataFrame()

    print("[INFO] Loading procedures...")
    try:
        procedures = pd.read_csv(hosp / "procedures_icd.csv.gz", low_memory=False)
    except Exception as e:
        print(f"[WARN] Could not load procedures: {e}")
        procedures = pd.DataFrame()

    print("[INFO] Loading prescriptions...")
    try:
        prescriptions = pd.read_csv(
            hosp / "prescriptions.csv.gz",
            usecols=["subject_id", "hadm_id", "drug", "ndc"],
            low_memory=False
        )
    except Exception as e:
        print(f"[WARN] Could not load prescriptions: {e}")
        prescriptions = pd.DataFrame()

    return admissions, patients, diagnoses, labevents, procedures, prescriptions

# ─── Charlson Comorbidity Index ───────────────────────────────────────────────
CCI_ICD10 = {
    "ami":             (["I21","I22","I25.2"], 1),
    "chf":             (["I09.9","I11.0","I13.0","I13.2","I25.5","I42.0",
                         "I42.5","I42.6","I42.7","I42.8","I42.9","I43","I50",
                         "P29.0"], 1),
    "pvd":             (["I70","I71","I73.1","I73.8","I73.9","I77.1",
                         "I79.0","I79.2","K55.1","K55.8","K55.9","Z95.8","Z95.9"], 1),
    "cerebrovascular": (["G45","G46","H34.0","I60","I61","I62","I63",
                         "I64","I65","I66","I67","I68","I69"], 1),
    "dementia":        (["F00","F01","F02","F03","F05.1","G30","G31.1"], 1),
    "copd":            (["I27.8","I27.9","J40","J41","J42","J43","J44",
                         "J45","J46","J47","J60","J61","J62","J63","J64",
                         "J65","J66","J67","J68.4","J70.1","J70.3"], 1),
    "rheumatic":       (["M05","M06","M31.5","M32","M33","M34",
                         "M35.1","M35.3","M36.0"], 1),
    "peptic_ulcer":    (["K25","K26","K27","K28"], 1),
    "mild_liver":      (["B18","K70.0","K70.1","K70.2","K70.3","K70.9",
                         "K71.3","K71.4","K71.5","K71.7","K73","K74",
                         "K76.0","K76.2","K76.3","K76.4","K76.8","K76.9",
                         "Z94.4"], 1),
    "diabetes_no_cc":  (["E10.0","E10.1","E10.6","E10.8","E10.9",
                         "E11.0","E11.1","E11.6","E11.8","E11.9",
                         "E12.0","E12.1","E12.6","E12.8","E12.9",
                         "E13.0","E13.1","E13.6","E13.8","E13.9",
                         "E14.0","E14.1","E14.6","E14.8","E14.9"], 1),
    "diabetes_cc":     (["E10.2","E10.3","E10.4","E10.5","E10.7",
                         "E11.2","E11.3","E11.4","E11.5","E11.7",
                         "E12.2","E12.3","E12.4","E12.5","E12.7",
                         "E13.2","E13.3","E13.4","E13.5","E13.7",
                         "E14.2","E14.3","E14.4","E14.5","E14.7"], 2),
    "hemiplegia":      (["G04.1","G11.4","G80.1","G80.2","G81","G82",
                         "G83.0","G83.1","G83.2","G83.3","G83.9"], 2),
    "renal":           (["I12.0","I13.1","N03.2","N03.3","N03.4","N03.5",
                         "N03.6","N03.7","N05.2","N05.3","N05.4","N05.5",
                         "N05.6","N05.7","N18","N19","N25.0","Z49.0",
                         "Z49.1","Z49.2","Z94.0","Z99.2"], 2),
    "malignancy":      (["C00","C01","C02","C03","C04","C05","C06","C07",
                         "C08","C09","C10","C11","C12","C13","C14","C15",
                         "C16","C17","C18","C19","C20","C21","C22","C23",
                         "C24","C25","C26","C30","C31","C32","C33","C34",
                         "C37","C38","C39","C40","C41","C43","C45","C46",
                         "C47","C48","C49","C50","C51","C52","C53","C54",
                         "C55","C56","C57","C58","C60","C61","C62","C63",
                         "C64","C65","C66","C67","C68","C69","C70","C71",
                         "C72","C73","C74","C75","C76","C81","C82","C83",
                         "C84","C85","C88","C90","C91","C92","C93","C94",
                         "C95","C96","C97"], 2),
    "severe_liver":    (["I85.0","I85.9","I86.4","I98.2","K70.4","K71.1",
                         "K72.1","K72.9","K76.5","K76.6","K76.7"], 3),
    "metastatic":      (["C77","C78","C79","C80"], 6),
    "aids":            (["B20","B21","B22","B24"], 6),
}

def compute_cci(diag_df):
    """Compute Charlson Comorbidity Index per hadm_id."""
    cci = {}
    for hadm_id, group in diag_df.groupby("hadm_id"):
        codes = group["icd_code"].astype(str).tolist()
        score = 0
        for cond, (prefixes, weight) in CCI_ICD10.items():
            for code in codes:
                if any(
                    code.startswith(p.replace(".", "")) or
                    code.replace(".", "").startswith(p.replace(".", ""))
                    for p in prefixes
                ):
                    score += weight
                    break
        cci[hadm_id] = min(score, 24)
    return pd.Series(cci, name="charlson_index")

# ─── Key lab item IDs (MIMIC-IV) ─────────────────────────────────────────────
LAB_ITEMS = {
    "creatinine": [50912],
    "egfr":       [50920],
    "bnp":        [51006],
    "albumin":    [50862],
    "wbc":        [51301],
    "hemoglobin": [51222],
    "sodium":     [50983],
    "potassium":  [50971],
    "hba1c":      [50852],
    "inr":        [51237],
}

def get_last_lab(labevents_df, hadm_ids):
    """Get last value per lab item per admission."""
    if labevents_df.empty:
        return pd.DataFrame({"hadm_id": hadm_ids})
    results = {}
    for feat, item_ids in LAB_ITEMS.items():
        subset = labevents_df[labevents_df["itemid"].isin(item_ids)]
        last = (
            subset.groupby("hadm_id")["valuenum"]
            .last()
            .reindex(hadm_ids)
            .rename(feat)
        )
        results[feat] = last
    out = pd.DataFrame(results)
    out.index.name = "hadm_id"
    return out.reset_index()

# ─── High-risk medication flags ───────────────────────────────────────────────
HIGH_RISK_MEDS = [
    "warfarin","coumadin","heparin","insulin","opioid","morphine",
    "oxycodone","hydrocodone","fentanyl","digoxin","methotrexate",
    "lithium","vancomycin","gentamicin","amiodarone"
]

def get_med_features(prescriptions_df, hadm_ids):
    """Count medications and flag high-risk drugs per admission."""
    if prescriptions_df.empty:
        return pd.DataFrame({
            "hadm_id": hadm_ids,
            "n_medications": 0,
            "high_risk_med": 0,
            "polypharmacy": 0
        })
    n_meds = (
        prescriptions_df.groupby("hadm_id")["drug"]
        .nunique()
        .reindex(hadm_ids)
        .fillna(0)
        .rename("n_medications")
    )
    rx = prescriptions_df.copy()
    rx["drug_lower"] = rx["drug"].str.lower().fillna("")
    rx["is_high_risk"] = rx["drug_lower"].apply(
        lambda d: int(any(m in d for m in HIGH_RISK_MEDS))
    )
    hr = (
        rx.groupby("hadm_id")["is_high_risk"]
        .max()
        .reindex(hadm_ids)
        .fillna(0)
        .rename("high_risk_med")
    )
    out = pd.DataFrame({
        "n_medications": n_meds,
        "high_risk_med": hr,
        "polypharmacy":  (n_meds >= 5).astype(int)
    })
    out.index.name = "hadm_id"
    return out.reset_index()

# ─── Build cohort ─────────────────────────────────────────────────────────────
def build_cohort(admissions, patients, diagnoses, labevents, procedures, prescriptions):
    print("[INFO] Building 30-day readmission cohort...")

    # Merge patient demographics
    df = admissions.merge(
        patients[["subject_id","gender","anchor_age","anchor_year","anchor_year_group"]],
        on="subject_id", how="left"
    )

    # Approximate age at admission
    df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    df["age"] = df["age"].clip(18, 120)

    # Exclusions: minors, missing discharge, in-hospital death, short stays
    df = df[df["age"] >= 18]
    df = df[df["dischtime"].notna()]
    df = df[df["hospital_expire_flag"] == 0]

    # Length of stay
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    df = df[df["los_days"] >= 1]

    # Sort chronologically per patient
    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    # Identify 30-day readmissions
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
    df["days_to_next"] = (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400
    df["readmitted_30d"] = (
        (df["days_to_next"] >= 0) & (df["days_to_next"] <= 30)
    ).astype(int)

    # Admission timestamp (seconds) for sorting
    df["admittime_ts"] = df["admittime"].astype(np.int64) // 10**9

    # Prior admissions in 12 months — vectorized per patient (fast on large data)
    print("[INFO] Computing prior admissions (vectorized)...")
    prior_counts = []
    for sid, grp in df.groupby("subject_id"):
        times = grp["admittime"].values
        counts = [
            int(((times >= t - np.timedelta64(365, "D")) & (times < t)).sum())
            for t in times
        ]
        prior_counts.extend(counts)
    df["prior_admissions_12mo"] = prior_counts

    # Charlson Comorbidity Index
    print("[INFO] Computing Charlson Comorbidity Index...")
    cci = compute_cci(diagnoses)
    cci_df = cci.reset_index()
    cci_df.columns = ["hadm_id", "charlson_index"]
    df = df.merge(cci_df, on="hadm_id", how="left")
    df["charlson_index"] = df["charlson_index"].fillna(0)

    # Number of diagnoses
    diag_count = diagnoses.groupby("hadm_id").size().reset_index()
    diag_count.columns = ["hadm_id", "n_diagnoses"]
    df = df.merge(diag_count, on="hadm_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0)

    # Number of procedures
    if not procedures.empty:
        proc_count = procedures.groupby("hadm_id").size().reset_index()
        proc_count.columns = ["hadm_id", "n_procedures"]
        df = df.merge(proc_count, on="hadm_id", how="left")
    else:
        df["n_procedures"] = 0
    df["n_procedures"] = df["n_procedures"].fillna(0)

    # Lab features
    print("[INFO] Extracting lab features...")
    labs = get_last_lab(labevents, df["hadm_id"].values)
    df = df.merge(labs, on="hadm_id", how="left")

    # Medication features
    print("[INFO] Extracting medication features...")
    med_feats = get_med_features(prescriptions, df["hadm_id"].values)
    df = df.merge(med_feats, on="hadm_id", how="left")

    # Encode categoricals
    df["gender_enc"]    = (df["gender"] == "M").astype(int)
    df["emergency_adm"] = (
        df["admission_type"].str.upper().str.contains("EMER|URGENT", na=False)
    ).astype(int)
    df["insurance_enc"] = df["insurance"].fillna("Other")
    df["race_enc"]      = df["race"].fillna("UNKNOWN")

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 50, 65, 75, 85, 200],
        labels=["18-50", "51-65", "66-75", "76-85", "85+"]
    )

    # Insurance simplified (MIMIC-IV v3.1 has cleaner insurance categories)
    def simplify_insurance(ins):
        ins = str(ins).upper()
        if "MEDICARE"  in ins: return "Medicare"
        if "MEDICAID"  in ins: return "Medicaid"
        if "SELF"      in ins: return "Uninsured"
        if "NO CHARGE" in ins: return "Uninsured"
        if "PRIVATE"   in ins: return "Private"
        return "Other"
    df["insurance_simple"] = df["insurance_enc"].apply(simplify_insurance)

    # Race simplified
    def simplify_race(r):
        r = str(r).upper()
        if "WHITE"                    in r: return "White"
        if "BLACK"                    in r: return "Black/AA"
        if "HISPANIC" in r or "LATIN" in r: return "Hispanic"
        if "ASIAN"                    in r: return "Asian"
        return "Other/Unknown"
    df["race_simple"] = df["race_enc"].apply(simplify_race)

    print(f"[INFO] Cohort built: {len(df):,} admissions, "
          f"{df['readmitted_30d'].mean()*100:.1f}% readmission rate.")
    return df

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if USE_FULL_MIMIC:
        base_dir = FULL_MIMIC_DIR
        print(f"[INFO] Using full MIMIC-IV from: {base_dir}")
        # Verify the path exists
        if not Path(base_dir).exists():
            raise FileNotFoundError(
                f"MIMIC-IV hosp directory not found: {base_dir}\n"
                f"Check that the download is complete and FULL_MIMIC_DIR is correct."
            )
    else:
        download_demo()
        extracted = [d for d in Path(DEMO_DIR).iterdir() if d.is_dir()]
        base_dir = str(extracted[0]) if extracted else DEMO_DIR
        print(f"[INFO] Using MIMIC-IV demo from: {base_dir}")

    admissions, patients, diagnoses, labevents, procedures, prescriptions = \
        load_tables(base_dir)

    cohort = build_cohort(
        admissions, patients, diagnoses, labevents, procedures, prescriptions
    )

    out_path = os.path.join(OUT_DIR, "cohort.csv")
    cohort.to_csv(out_path, index=False)

    print(f"\n[DONE] Cohort saved to: {out_path}")
    print(f"       Shape: {cohort.shape}")
    print(f"       Readmission rate: {cohort['readmitted_30d'].mean()*100:.2f}%")
    print(f"       Columns: {list(cohort.columns)}")