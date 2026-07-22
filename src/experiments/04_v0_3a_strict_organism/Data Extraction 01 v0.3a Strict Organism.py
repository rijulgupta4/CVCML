# %% Imports and paths

import os
import re

import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")

os.makedirs(DATA_PATH, exist_ok=True)


# %% Strict organism helper functions

COMMENSAL_PATTERNS = [
    r"COAGULASE NEGATIVE",
    r"COAGULASE-NEGATIVE",
    r"STAPHYLOCOCCUS EPIDERMIDIS",
    r"STAPHYLOCOCCUS HOMINIS",
    r"STAPHYLOCOCCUS HAEMOLYTICUS",
    r"STAPHYLOCOCCUS CAPITIS",
    r"STAPHYLOCOCCUS WARNERI",
    r"STAPHYLOCOCCUS LUGDUNENSIS",
    r"CORYNEBACTERIUM",
    r"\bBACILLUS\b",
    r"LACTOBACILLUS",
    r"MICROCOCCUS",
    r"CUTIBACTERIUM",
    r"PROPIONIBACTERIUM",
    r"DIPHTHEROID",
    r"VIRIDANS",
    r"AEROCOCCUS",
]
COMMENSAL_RE = re.compile("|".join(COMMENSAL_PATTERNS), flags=re.IGNORECASE)


def is_common_commensal(org_name):
    if pd.isna(org_name):
        return False
    return bool(COMMENSAL_RE.search(str(org_name)))


def summarize_strict_label(group):
    group = group.sort_values("charttime").copy()
    commensal_mask = group["is_common_commensal"].fillna(False).astype(bool)
    clear_pathogens = group[~commensal_mask]
    common_commensals = group[commensal_mask]

    if len(clear_pathogens) > 0:
        strict_rows = clear_pathogens
        strict_reason = "clear_pathogen"
        strict_positive = 1
    else:
        distinct_commensal_times = common_commensals["charttime"].dropna().nunique()
        strict_positive = int(distinct_commensal_times >= 2)
        strict_rows = common_commensals if strict_positive else group.iloc[0:0]
        strict_reason = "two_common_commensal_cultures" if strict_positive else "commensal_single_or_none"

    if strict_positive:
        culture_time = strict_rows["charttime"].min()
        strict_positive_orgs = "; ".join(sorted(strict_rows["org_name"].dropna().astype(str).unique()))
    else:
        culture_time = pd.NaT
        strict_positive_orgs = ""

    return pd.Series({
        "clabsi_strict_organism": strict_positive,
        "strict_culture_time": culture_time,
        "strict_positive_orgs": strict_positive_orgs,
        "strict_label_reason": strict_reason,
        "strict_qualifying_culture_rows": int(len(strict_rows)),
        "strict_clear_pathogen_rows": int(len(clear_pathogens)),
        "strict_commensal_rows": int(len(common_commensals)),
        "strict_distinct_commensal_times": int(common_commensals["charttime"].dropna().nunique()),
    })


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


# %% Create pragmatic v0.2 and strict v0.3a CLABSI labels

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
positive_cultures["is_common_commensal"] = positive_cultures["org_name"].apply(is_common_commensal)

cvc_cohort["starttime"] = pd.to_datetime(cvc_cohort["starttime"])
cvc_cohort["endtime"] = pd.to_datetime(cvc_cohort["endtime"])
cvc_cohort["earliest_clabsi_time"] = cvc_cohort["starttime"] + pd.Timedelta(hours=48)

merged = cvc_cohort.merge(
    positive_cultures[["subject_id", "charttime", "org_name", "is_common_commensal"]],
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
merged["v0_2_qualifying_culture"] = (
    (merged["charttime"] >= merged["earliest_clabsi_time"])
    & (merged["charttime"] <= merged["endtime"])
)

qualifying_cultures = merged[merged["v0_2_qualifying_culture"]].copy()
qualifying_cultures["org_name_upper"] = qualifying_cultures["org_name"].astype(str).str.upper()

pragmatic_labels = (
    merged.groupby("stay_id")["v0_2_qualifying_culture"]
    .max()
    .astype(int)
    .reset_index()
    .rename(columns={"v0_2_qualifying_culture": "clabsi_pragmatic_v0_2"})
)

pragmatic_culture_times = (
    qualifying_cultures
    .sort_values("charttime")
    .groupby("stay_id")["charttime"]
    .first()
    .reset_index()
    .rename(columns={"charttime": "pragmatic_culture_time"})
)

if len(qualifying_cultures) > 0:
    strict_labels = (
        qualifying_cultures
        .groupby("stay_id")
        .apply(summarize_strict_label)
        .reset_index()
    )
else:
    strict_labels = pd.DataFrame(columns=[
        "stay_id",
        "clabsi_strict_organism",
        "strict_culture_time",
        "strict_positive_orgs",
        "strict_label_reason",
        "strict_qualifying_culture_rows",
        "strict_clear_pathogen_rows",
        "strict_commensal_rows",
        "strict_distinct_commensal_times",
    ])

early_culture_flags = (
    merged.groupby("stay_id")["early_positive_culture"]
    .max()
    .reset_index()
)

cvc_cohort = cvc_cohort.merge(pragmatic_labels, on="stay_id", how="left")
cvc_cohort = cvc_cohort.merge(pragmatic_culture_times, on="stay_id", how="left")
cvc_cohort = cvc_cohort.merge(strict_labels, on="stay_id", how="left")
cvc_cohort = cvc_cohort.merge(early_culture_flags, on="stay_id", how="left")

cvc_cohort["clabsi_pragmatic_v0_2"] = cvc_cohort["clabsi_pragmatic_v0_2"].fillna(0).astype(int)
cvc_cohort["clabsi_strict_organism"] = cvc_cohort["clabsi_strict_organism"].fillna(0).astype(int)
cvc_cohort["clabsi"] = cvc_cohort["clabsi_strict_organism"]
cvc_cohort["culture_time"] = pd.to_datetime(cvc_cohort["strict_culture_time"], errors="coerce")
cvc_cohort["early_positive_culture"] = (
    cvc_cohort["early_positive_culture"].fillna(False).astype(int)
)
cvc_cohort["strict_positive_orgs"] = cvc_cohort["strict_positive_orgs"].fillna("")
cvc_cohort["strict_label_reason"] = cvc_cohort["strict_label_reason"].fillna("no_qualifying_positive_culture")
cvc_cohort["pragmatic_downgraded_to_negative"] = (
    (cvc_cohort["clabsi_pragmatic_v0_2"] == 1)
    & (cvc_cohort["clabsi_strict_organism"] == 0)
).astype(int)

for col in [
    "strict_qualifying_culture_rows",
    "strict_clear_pathogen_rows",
    "strict_commensal_rows",
    "strict_distinct_commensal_times",
]:
    cvc_cohort[col] = cvc_cohort[col].fillna(0).astype(int)


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

n_total = len(cvc_cohort)
n_pragmatic = int(cvc_cohort["clabsi_pragmatic_v0_2"].sum())
n_strict = int(cvc_cohort["clabsi_strict_organism"].sum())
n_downgraded = int(cvc_cohort["pragmatic_downgraded_to_negative"].sum())
n_early = int(cvc_cohort["early_positive_culture"].sum())

print("v0.3a strict-organism label summary:")
print(f"  Total CVC stays:                 {n_total:,}")
print(f"  Pragmatic v0.2 positive:         {n_pragmatic:,} ({n_pragmatic / n_total * 100:.1f}%)")
print(f"  Strict organism positive:        {n_strict:,} ({n_strict / n_total * 100:.1f}%)")
print(f"  Downgraded pragmatic positives:  {n_downgraded:,}")
print(f"  Early positive cultures flagged: {n_early:,}")
print(f"  Strict culture time captured:    {cvc_cohort['culture_time'].notna().sum():,}")

label_audit = pd.DataFrame([{
    "total_cvc_stays": n_total,
    "pragmatic_v0_2_positive": n_pragmatic,
    "pragmatic_v0_2_rate": n_pragmatic / n_total,
    "strict_organism_positive": n_strict,
    "strict_organism_rate": n_strict / n_total,
    "strict_negative": n_total - n_strict,
    "pragmatic_downgraded_to_negative": n_downgraded,
    "early_positive_culture_stays": n_early,
    "strict_culture_time_captured": int(cvc_cohort["culture_time"].notna().sum()),
}])

label_audit_file = os.path.join(DATA_PATH, "v0_3a_label_audit.csv")
label_audit.to_csv(label_audit_file, index=False)

strict_detail_file = os.path.join(DATA_PATH, "v0_3a_strict_label_detail.csv")
cvc_cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "endtime",
        "dwell_hours",
        "clabsi_pragmatic_v0_2",
        "clabsi_strict_organism",
        "pragmatic_downgraded_to_negative",
        "pragmatic_culture_time",
        "strict_culture_time",
        "strict_positive_orgs",
        "strict_label_reason",
        "strict_qualifying_culture_rows",
        "strict_clear_pathogen_rows",
        "strict_commensal_rows",
        "strict_distinct_commensal_times",
        "early_positive_culture",
    ]
].to_csv(strict_detail_file, index=False)

organism_audit_file = os.path.join(DATA_PATH, "v0_3a_qualifying_organism_counts.csv")
organism_counts = (
    qualifying_cultures
    .assign(organism_type=lambda x: x["is_common_commensal"].map({True: "common_commensal", False: "clear_pathogen"}))
    .groupby(["org_name", "organism_type"])
    .agg(culture_rows=("org_name", "size"), stays=("stay_id", "nunique"))
    .reset_index()
    .sort_values(["organism_type", "stays", "culture_rows"], ascending=[True, False, False])
)
organism_counts.to_csv(organism_audit_file, index=False)

cohort_file = os.path.join(DATA_PATH, "clabsi_cohort_v0_3a.csv")
cvc_cohort.to_csv(cohort_file, index=False)

print("")
print(f"Cohort saved to:            {cohort_file}")
print(f"Label audit saved to:       {label_audit_file}")
print(f"Strict detail saved to:     {strict_detail_file}")
print(f"Organism audit saved to:    {organism_audit_file}")
print(f"Shape: {cvc_cohort.shape}")
print("Missing values:")
print(cvc_cohort.isnull().sum()[cvc_cohort.isnull().sum() > 0].to_string())
print("")
print("Data Extraction 01 v0.3a Strict Organism complete.")

