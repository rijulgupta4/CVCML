# %% Imports and paths

import os

import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4b")

os.makedirs(DATA_PATH, exist_ok=True)

LAB_CACHE_FILE = os.path.join(DATA_PATH, "v0_4b_labs_long.pkl")
VITAL_CACHE_FILE = os.path.join(DATA_PATH, "v0_4b_vitals_long.pkl")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4b_vitals_extraction_audit.csv")
COUNTS_FILE = os.path.join(DATA_PATH, "v0_4b_vitals_extraction_counts.csv")


# %% Item definitions

LAB_ITEMS = {
    "wbc": [51301],
    "lactate": [50813, 52442],
    "hemoglobin": [51222],
    "platelets": [51265],
    "creatinine": [50912],
}
all_lab_ids = [itemid for ids in LAB_ITEMS.values() for itemid in ids]
id_to_lab = {itemid: lab for lab, ids in LAB_ITEMS.items() for itemid in ids}

VITAL_ITEMS = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "temperature_c": [223761, 223762],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "map": [220052, 220181],
}
all_vital_ids = [itemid for ids in VITAL_ITEMS.values() for itemid in ids]
id_to_vital = {itemid: vital for vital, ids in VITAL_ITEMS.items() for itemid in ids}


# %% Helpers

def apply_vital_cleaning(df):
    df = df.copy()
    df["vital_value"] = df["valuenum"]

    temp_f = df["itemid"].eq(223761) | (
        df["vital_name"].eq("temperature_c") & df["vital_value"].gt(80)
    )
    df.loc[temp_f, "vital_value"] = (df.loc[temp_f, "vital_value"] - 32) * 5 / 9

    ranges = {
        "heart_rate": (20, 250),
        "respiratory_rate": (4, 80),
        "spo2": (50, 100),
        "temperature_c": (25, 45),
        "sbp": (40, 260),
        "dbp": (20, 160),
        "map": (30, 200),
    }
    keep = pd.Series(True, index=df.index)
    for vital_name, (low, high) in ranges.items():
        mask = df["vital_name"].eq(vital_name)
        keep.loc[mask] = df.loc[mask, "vital_value"].between(low, high)
    return df.loc[keep].copy()


# %% Load strict cohort

cohort = pd.read_csv(
    os.path.join(SOURCE_DATA_PATH, "clabsi_cohort_v0_3a.csv"),
    usecols=["subject_id", "hadm_id", "stay_id", "clabsi"],
)

cohort_hadm = set(cohort["hadm_id"].dropna().astype(int).values)
cohort_stays = set(cohort["stay_id"].dropna().astype(int).values)

print(f"Strict v0.3a cohort loaded: {cohort.shape}")
print(f"Unique hospitalizations: {cohort['hadm_id'].nunique():,}")
print(f"Unique ICU stays:        {cohort['stay_id'].nunique():,}")
print(f"Strict CLABSI-positive:  {cohort['clabsi'].sum():,}")


# %% Extract target labs from labevents

if os.path.exists(LAB_CACHE_FILE):
    print("")
    print(f"Lab extraction already exists, reading: {LAB_CACHE_FILE}")
    labevents = pd.read_pickle(LAB_CACHE_FILE)
else:
    print("")
    print("Extracting target labs from labevents...")
    lab_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        os.path.join(HOSP, "labevents.csv.gz"),
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=500000,
        low_memory=False,
        parse_dates=["charttime"],
    ), start=1):
        filtered = chunk[
            chunk["hadm_id"].isin(cohort_hadm)
            & chunk["itemid"].isin(all_lab_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if len(filtered) > 0:
            filtered["lab_name"] = filtered["itemid"].map(id_to_lab)
            lab_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} lab chunks | matched rows: {matched_rows:,}")

    labevents = pd.concat(lab_chunks, ignore_index=True) if lab_chunks else pd.DataFrame()
    labevents.to_pickle(LAB_CACHE_FILE)
    print(f"Saved extracted labs: {LAB_CACHE_FILE}")

print(f"Total extracted lab rows: {len(labevents):,}")
if len(labevents):
    print(labevents["lab_name"].value_counts().to_string())


# %% Extract target vitals from chartevents

if os.path.exists(VITAL_CACHE_FILE):
    print("")
    print(f"Vital extraction already exists, reading: {VITAL_CACHE_FILE}")
    vitals = pd.read_pickle(VITAL_CACHE_FILE)
    raw_vital_rows = len(vitals)
else:
    print("")
    print("Extracting target vitals from chartevents...")
    vital_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        os.path.join(ICU, "chartevents.csv.gz"),
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        chunksize=750000,
        low_memory=False,
        parse_dates=["charttime"],
    ), start=1):
        filtered = chunk[
            chunk["stay_id"].isin(cohort_stays)
            & chunk["itemid"].isin(all_vital_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if len(filtered) > 0:
            filtered["vital_name"] = filtered["itemid"].map(id_to_vital)
            vital_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} chart chunks | matched rows: {matched_rows:,}")

    raw_vital_rows = matched_rows
    vitals = pd.concat(vital_chunks, ignore_index=True) if vital_chunks else pd.DataFrame()
    vitals = apply_vital_cleaning(vitals)
    vitals.to_pickle(VITAL_CACHE_FILE)
    print(f"Saved extracted vitals: {VITAL_CACHE_FILE}")

print(f"Total cleaned vital rows: {len(vitals):,}")
if len(vitals):
    print(vitals["vital_name"].value_counts().to_string())


# %% Save extraction audit

audit = pd.DataFrame([{
    "source_cohort_stays": int(cohort["stay_id"].nunique()),
    "source_cohort_hadm": int(cohort["hadm_id"].nunique()),
    "source_strict_positive_stays": int(cohort["clabsi"].sum()),
    "target_lab_itemids": ", ".join(str(x) for x in all_lab_ids),
    "target_vital_itemids": ", ".join(str(x) for x in all_vital_ids),
    "extracted_lab_rows": int(len(labevents)),
    "raw_matched_vital_rows": int(raw_vital_rows),
    "cleaned_vital_rows": int(len(vitals)),
    "lab_cache_file": LAB_CACHE_FILE,
    "vital_cache_file": VITAL_CACHE_FILE,
}])
audit.to_csv(AUDIT_FILE, index=False)

count_rows = []
if len(labevents):
    for name, count in labevents["lab_name"].value_counts().items():
        count_rows.append({"source": "labevents", "name": name, "rows": int(count)})
if len(vitals):
    for name, count in vitals["vital_name"].value_counts().items():
        count_rows.append({"source": "chartevents", "name": name, "rows": int(count)})
pd.DataFrame(count_rows).to_csv(COUNTS_FILE, index=False)

print("")
print(f"Extraction audit saved to:  {AUDIT_FILE}")
print(f"Extraction counts saved to: {COUNTS_FILE}")
print("")
print("Data Extraction 01 v0.4B Vitals complete.")

