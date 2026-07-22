# %% Imports and paths

import os
import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_2")

os.makedirs(DATA_PATH, exist_ok=True)


# %% Load lightweight tables

print("Loading patients...")
patients = pd.read_csv(os.path.join(HOSP, "patients.csv.gz"))
print(f"  {len(patients):,} rows")

print("Loading admissions...")
admissions = pd.read_csv(os.path.join(HOSP, "admissions.csv.gz"))
print(f"  {len(admissions):,} rows")

print("Loading d_items...")
d_items = pd.read_csv(os.path.join(ICU, "d_items.csv.gz"))
print(f"  {len(d_items):,} rows")


# %% Define CVC item IDs and load procedureevents

CVC_PROCEDURE_IDS = [
    224264,  # PICC Line
    224270,  # Dialysis Catheter
    224273,  # Presep Catheter
    224560,  # PA Catheter
    225203,  # Pheresis Catheter
    229517,  # Multi Lumen Cooling Catheter
]

print("Loading procedureevents...")
procedureevents = pd.read_csv(os.path.join(ICU, "procedureevents.csv.gz"))
print(f"  {len(procedureevents):,} total rows")

cvc_procedures = procedureevents[
    procedureevents["itemid"].isin(CVC_PROCEDURE_IDS)
].copy()

cvc_procedures["starttime"] = pd.to_datetime(cvc_procedures["starttime"])
cvc_procedures["endtime"] = pd.to_datetime(cvc_procedures["endtime"])
cvc_procedures["dwell_hours"] = cvc_procedures["value"] / 60

cvc_cohort = cvc_procedures[cvc_procedures["dwell_hours"] >= 48].copy()
cvc_cohort = cvc_cohort.merge(
    d_items[["itemid", "label"]],
    on="itemid",
    how="left",
).rename(columns={"label": "cvc_type"})

cvc_cohort = cvc_cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "caregiver_id",
        "starttime",
        "endtime",
        "dwell_hours",
        "cvc_type",
        "location",
    ]
].reset_index(drop=True)

print(f"CVC events >= 48hr dwell time: {len(cvc_cohort):,}")

cvc_cohort = (
    cvc_cohort
    .sort_values("dwell_hours", ascending=False)
    .drop_duplicates(subset="stay_id", keep="first")
    .reset_index(drop=True)
)

print("After deduplication:")
print(f"  Unique ICU stays: {len(cvc_cohort):,}")
print(f"  Unique patients:  {cvc_cohort['subject_id'].nunique():,}")


# %% Create pragmatic v0.2 CLABSI labels

print("Loading microbiologyevents...")
micro = pd.read_csv(os.path.join(HOSP, "microbiologyevents.csv.gz"), low_memory=False)
print(f"  {len(micro):,} total rows")

blood_cultures = micro[
    micro["spec_type_desc"].isin([
        "BLOOD CULTURE",
        "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
    ])
].copy()

positive_cultures = blood_cultures[
    blood_cultures["org_name"].notna()
    & ~blood_cultures["org_name"].str.contains("CANCELLED", case=False, na=False)
].copy()

positive_cultures["charttime"] = pd.to_datetime(positive_cultures["charttime"])
cvc_cohort["starttime"] = pd.to_datetime(cvc_cohort["starttime"])
cvc_cohort["endtime"] = pd.to_datetime(cvc_cohort["endtime"])
cvc_cohort["earliest_clabsi_time"] = cvc_cohort["starttime"] + pd.Timedelta(hours=48)

merged = cvc_cohort.merge(
    positive_cultures[["subject_id", "charttime", "org_name"]],
    on="subject_id",
    how="left",
)

merged["culture_while_cvc"] = (
    (merged["charttime"] >= merged["starttime"])
    & (merged["charttime"] <= merged["endtime"])
)
merged["early_positive_culture"] = (
    merged["culture_while_cvc"]
    & (merged["charttime"] < merged["earliest_clabsi_time"])
)

# v0.2 rule: culture must be drawn at least 48 hours after catheter placement.
merged["clabsi"] = (
    (merged["charttime"] >= merged["earliest_clabsi_time"])
    & (merged["charttime"] <= merged["endtime"])
).astype(int)

culture_times = (
    merged[merged["clabsi"] == 1]
    .sort_values("charttime")
    .groupby("stay_id")["charttime"]
    .first()
    .reset_index()
    .rename(columns={"charttime": "culture_time"})
)

clabsi_labels = merged.groupby("stay_id")["clabsi"].max().reset_index()
early_culture_flags = (
    merged.groupby("stay_id")["early_positive_culture"]
    .max()
    .reset_index()
)

cvc_cohort = cvc_cohort.merge(clabsi_labels, on="stay_id", how="left")
cvc_cohort = cvc_cohort.merge(culture_times, on="stay_id", how="left")
cvc_cohort = cvc_cohort.merge(early_culture_flags, on="stay_id", how="left")
cvc_cohort["clabsi"] = cvc_cohort["clabsi"].fillna(0).astype(int)
cvc_cohort["early_positive_culture"] = (
    cvc_cohort["early_positive_culture"].fillna(False).astype(int)
)

n_total = len(cvc_cohort)
n_clabsi = int(cvc_cohort["clabsi"].sum())
n_early = int(cvc_cohort["early_positive_culture"].sum())

print("v0.2 label summary:")
print(f"  Total CVC stays:                {n_total:,}")
print(f"  CLABSI positive:                {n_clabsi:,} ({n_clabsi / n_total * 100:.1f}%)")
print(f"  CLABSI negative:                {n_total - n_clabsi:,} ({(1 - n_clabsi / n_total) * 100:.1f}%)")
print(f"  Early positive cultures flagged:{n_early:,}")
print(f"  Culture time captured:          {cvc_cohort['culture_time'].notna().sum():,}")

label_audit = pd.DataFrame([{
    "total_cvc_stays": n_total,
    "clabsi_positive": n_clabsi,
    "clabsi_rate": n_clabsi / n_total,
    "clabsi_negative": n_total - n_clabsi,
    "early_positive_culture_stays": n_early,
    "culture_time_captured": int(cvc_cohort["culture_time"].notna().sum()),
}])
label_audit_file = os.path.join(DATA_PATH, "v0_2_label_audit.csv")
label_audit.to_csv(label_audit_file, index=False)


# %% Merge patient demographics

patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")

patients_slim = patients[["subject_id", "gender", "anchor_age"]].copy()
admissions_slim = admissions[
    [
        "subject_id",
        "hadm_id",
        "admission_type",
        "insurance",
        "marital_status",
        "race",
    ]
].copy()

cvc_cohort = cvc_cohort.merge(patients_slim, on="subject_id", how="left")
cvc_cohort = cvc_cohort.merge(admissions_slim, on=["subject_id", "hadm_id"], how="left")


# %% Clean missing values and save cohort

cvc_cohort = cvc_cohort.drop(columns=["caregiver_id"])
cvc_cohort["site_known"] = cvc_cohort["location"].notna().astype(int)
cvc_cohort = cvc_cohort.drop(columns=["location"])

cvc_cohort["insurance"] = cvc_cohort["insurance"].fillna("Unknown")
cvc_cohort["marital_status"] = cvc_cohort["marital_status"].fillna("Unknown")

cohort_file = os.path.join(DATA_PATH, "clabsi_cohort_v0_2.csv")
cvc_cohort.to_csv(cohort_file, index=False)

print("")
print(f"Cohort saved to: {cohort_file}")
print(f"Label audit saved to: {label_audit_file}")
print(f"Shape: {cvc_cohort.shape}")
print("Missing values:")
print(cvc_cohort.isnull().sum()[cvc_cohort.isnull().sum() > 0].to_string())
print("")
print("Data Extraction 01 v0.2 complete.")

