# %% Imports and paths

import os

import numpy as np
import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")

os.makedirs(DATA_PATH, exist_ok=True)


# %% Load v0.3a cohort

cohort = pd.read_csv(
    os.path.join(DATA_PATH, "clabsi_cohort_v0_3a.csv"),
    parse_dates=[
        "starttime",
        "endtime",
        "culture_time",
        "pragmatic_culture_time",
        "strict_culture_time",
    ],
)

print(f"Cohort loaded: {cohort.shape}")
print(f"Strict CLABSI positive: {cohort['clabsi'].sum():,}")
print(f"Strict CLABSI negative: {(cohort['clabsi'] == 0).sum():,}")
print(f"Pragmatic v0.2 positive retained in audit: {cohort['clabsi_pragmatic_v0_2'].sum():,}")


# %% Reference time and dwell features

cohort["ref_time"] = np.where(
    cohort["clabsi"] == 1,
    cohort["culture_time"],
    cohort["endtime"],
)
cohort["ref_time"] = pd.to_datetime(cohort["ref_time"])
cohort["window_start"] = cohort["ref_time"] - pd.Timedelta(hours=48)

cohort["dwell_at_ref_hours"] = (
    (cohort["ref_time"] - cohort["starttime"]).dt.total_seconds() / 3600
).clip(lower=0)
cohort["post_ref_dwell_hours"] = cohort["dwell_hours"] - cohort["dwell_at_ref_hours"]

print("Reference time logic:")
print(f"  Strict CLABSI+ uses strict culture_time: {(cohort['clabsi'] == 1).sum():,}")
print(f"  Strict CLABSI- uses catheter endtime:    {(cohort['clabsi'] == 0).sum():,}")
print("")
print("Dwell diagnostic:")
print(
    cohort.groupby("clabsi")[["dwell_hours", "dwell_at_ref_hours", "post_ref_dwell_hours"]]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .to_string()
)


# %% Define lab item IDs

LAB_ITEMS = {
    "wbc": [51301],
    "lactate": [50813, 52442],
    "crp": [50889, 51652],
    "hemoglobin": [51222],
    "platelets": [51265],
    "creatinine": [50912],
    "albumin": [50862],
}
all_lab_ids = [itemid for ids in LAB_ITEMS.values() for itemid in ids]
id_to_lab = {itemid: lab for lab, ids in LAB_ITEMS.items() for itemid in ids}


# %% Load labevents filtered to cohort and target labs

print("Loading labevents...")
cohort_subjects = set(cohort["subject_id"].values)
chunks = []

for chunk in pd.read_csv(
    os.path.join(HOSP, "labevents.csv.gz"),
    chunksize=500000,
    low_memory=False,
    parse_dates=["charttime"],
):
    filtered = chunk[
        chunk["subject_id"].isin(cohort_subjects)
        & chunk["itemid"].isin(all_lab_ids)
        & chunk["valuenum"].notna()
    ]
    if len(filtered) > 0:
        chunks.append(filtered)

labevents = pd.concat(chunks, ignore_index=True)
labevents["lab_name"] = labevents["itemid"].map(id_to_lab)

print(f"  Total lab rows for cohort: {len(labevents):,}")
print("  Rows per lab:")
print(labevents["lab_name"].value_counts().to_string())


# %% Window labs to 48 hours before reference time

print("Joining labs to 48-hour window...")
labs_merged = labevents.merge(
    cohort[["subject_id", "stay_id", "window_start", "ref_time"]],
    on="subject_id",
    how="inner",
)

labs_windowed = labs_merged[
    (labs_merged["charttime"] >= labs_merged["window_start"])
    & (labs_merged["charttime"] <= labs_merged["ref_time"])
].copy()

print(f"  Lab rows before windowing:   {len(labs_merged):,}")
print(f"  Lab rows within 48hr window: {len(labs_windowed):,}")
print(f"  Unique stays with labs:      {labs_windowed['stay_id'].nunique():,}")


# %% Aggregate labs

labs_windowed = labs_windowed.sort_values(["stay_id", "lab_name", "charttime"])
lab_features = (
    labs_windowed
    .groupby(["stay_id", "lab_name"])["valuenum"]
    .agg(mean_val="mean", last_val="last", first_val="first")
    .reset_index()
)
lab_features["trend"] = lab_features["last_val"] - lab_features["first_val"]

lab_pivot = lab_features.pivot_table(
    index="stay_id",
    columns="lab_name",
    values=["mean_val", "last_val", "trend"],
)
lab_pivot.columns = [
    f"{lab}_{metric.replace('_val', '')}"
    for metric, lab in lab_pivot.columns
]
lab_pivot = lab_pivot.reset_index()

cohort_featured = cohort.merge(lab_pivot, on="stay_id", how="left")


# %% Missingness and feature cleanup

lab_cols = [
    c for c in cohort_featured.columns
    if any(c.startswith(lab) for lab in ["wbc", "lactate", "crp", "hemoglobin", "platelets", "creatinine", "albumin"])
]
missing_report = (
    cohort_featured[lab_cols].isnull().sum() / len(cohort_featured) * 100
).round(1)
print("Lab missing rates:")
print(missing_report.to_string())

cohort_featured["lactate_measured"] = cohort_featured["lactate_last"].notna().astype(int)

drop_cols = [c for c in cohort_featured.columns if c.startswith("crp") or c.startswith("albumin")]
cohort_featured = cohort_featured.drop(columns=drop_cols)

core_lab_cols = [
    c for c in cohort_featured.columns
    if any(c.startswith(lab) for lab in ["wbc", "hemoglobin", "platelets", "creatinine"])
]

before = len(cohort_featured)
clabsi_before = int(cohort_featured["clabsi"].sum())
cohort_featured = cohort_featured.dropna(subset=core_lab_cols).reset_index(drop=True)
after = len(cohort_featured)
clabsi_after = int(cohort_featured["clabsi"].sum())

print("Listwise deletion for core labs:")
print(f"  Before: {before:,} stays | {clabsi_before:,} positives")
print(f"  After:  {after:,} stays | {clabsi_after:,} positives")

lactate_cols = [c for c in cohort_featured.columns if c.startswith("lactate")]
for col in lactate_cols:
    cohort_featured[col] = cohort_featured[col].fillna(0)


# %% Consolidate race and encode categoricals

race_map = {
    "WHITE": "White",
    "WHITE - BRAZILIAN": "White",
    "WHITE - EASTERN EUROPEAN": "White",
    "WHITE - OTHER EUROPEAN": "White",
    "WHITE - RUSSIAN": "White",
    "BLACK/AFRICAN AMERICAN": "Black",
    "BLACK/AFRICAN": "Black",
    "BLACK/CAPE VERDEAN": "Black",
    "BLACK/CARIBBEAN ISLAND": "Black",
    "ASIAN": "Asian",
    "ASIAN - ASIAN INDIAN": "Asian",
    "ASIAN - CHINESE": "Asian",
    "ASIAN - KOREAN": "Asian",
    "ASIAN - SOUTH EAST ASIAN": "Asian",
    "HISPANIC OR LATINO": "Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "HISPANIC/LATINO - COLUMBIAN": "Hispanic",
    "HISPANIC/LATINO - CUBAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic",
    "SOUTH AMERICAN": "Hispanic",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Other",
    "AMERICAN INDIAN/ALASKA NATIVE": "Other",
    "MULTIPLE RACE/ETHNICITY": "Other",
    "PORTUGUESE": "Other",
    "OTHER": "Other",
    "PATIENT DECLINED TO ANSWER": "Unknown",
    "UNABLE TO OBTAIN": "Unknown",
    "UNKNOWN": "Unknown",
}

cohort_featured["race_consolidated"] = cohort_featured["race"].map(race_map).fillna("Unknown")
cohort_featured = cohort_featured.drop(columns=["race"])

cohort_encoded = pd.get_dummies(
    cohort_featured,
    columns=[
        "gender",
        "cvc_type",
        "admission_type",
        "insurance",
        "marital_status",
        "race_consolidated",
    ],
    drop_first=False,
)


# %% Save final v0.3a feature matrix

output_file = os.path.join(DATA_PATH, "clabsi_features_v0_3a.csv")
cohort_encoded.to_csv(output_file, index=False)

meta_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "culture_time",
    "pragmatic_culture_time",
    "strict_culture_time",
    "ref_time",
    "window_start",
    "earliest_clabsi_time",
    "post_ref_dwell_hours",
    "dwell_hours",
    "early_positive_culture",
    "clabsi_pragmatic_v0_2",
    "clabsi_strict_organism",
    "strict_positive_orgs",
    "strict_label_reason",
    "strict_qualifying_culture_rows",
    "strict_clear_pathogen_rows",
    "strict_commensal_rows",
    "strict_distinct_commensal_times",
    "pragmatic_downgraded_to_negative",
]
meta_cols = [c for c in meta_cols if c in cohort_encoded.columns]
feature_cols = [c for c in cohort_encoded.columns if c not in meta_cols]

print("")
print(f"Feature matrix saved to: {output_file}")
print(f"Shape: {cohort_encoded.shape}")
print(f"Metadata/audit columns: {len(meta_cols)}")
print(f"Feature columns:        {len([c for c in feature_cols if c != 'clabsi'])}")
print(f"Strict CLABSI positive: {cohort_encoded['clabsi'].sum():,} ({cohort_encoded['clabsi'].mean() * 100:.1f}%)")
print("")
print("Feature Engineering 02 v0.3a complete.")

