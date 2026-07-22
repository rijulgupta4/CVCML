# %% Imports and paths

import os
import re

import pandas as pd


MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
BASE_FEATURE_PATH = os.path.join(PROJECT_PATH, "data", "v0_4e")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4g")

os.makedirs(DATA_PATH, exist_ok=True)

CAREGIVER_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_caregiver_chartevents_long.pkl")
LINECARE_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_linecare_datetimeevents_long.pkl")
FLUID_INPUT_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_fluid_inputevents_long.pkl")
FLUID_OUTPUT_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_fluid_outputevents_long.pkl")
ITEM_MAP_FILE = os.path.join(DATA_PATH, "v0_4g_care_process_item_map.csv")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4g_care_process_extraction_audit.csv")
COUNTS_FILE = os.path.join(DATA_PATH, "v0_4g_care_process_extraction_counts.csv")

BASE_FEATURE_FILE = os.path.join(BASE_FEATURE_PATH, "clabsi_landmark_features_v0_4e.csv")


# %% Item definitions

VITAL_ITEMS = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "temperature_c": [223761, 223762],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "map": [220052, 220181],
}
all_vital_ids = {itemid for ids in VITAL_ITEMS.values() for itemid in ids}

LINECARE_INCLUDE_PATTERN = re.compile(
    r"(dressing change|cap change|tubing change|site appear|biopatch|secureport|change over wire|insertion date)",
    flags=re.IGNORECASE,
)
LINECARE_EXCLUDE_PATTERN = re.compile(
    r"(tip cultured|blood cultured|culture|discontinued|removed|reason discontinued)",
    flags=re.IGNORECASE,
)


# %% Helpers

def classify_linecare_event(label):
    text = str(label).lower()
    if "dressing change" in text:
        return "dressing_change"
    if "cap change" in text:
        return "cap_change"
    if "tubing change" in text:
        return "tubing_change"
    if "site appear" in text:
        return "site_assessment"
    if "biopatch" in text:
        return "biopatch_documented"
    if "secureport" in text:
        return "securement_documented"
    if "change over wire" in text:
        return "change_over_wire"
    if "insertion date" in text:
        return "insertion_date_documented"
    return "other_linecare"


def classify_output_event(label):
    text = str(label).lower()
    if any(term in text for term in ["urine", "foley", "void", "straight cath", "condom cath"]):
        return "urine_output"
    if any(term in text for term in ["drain", "chest tube", "jp", "hemovac", "ng", "ostomy"]):
        return "drain_output"
    return "other_output"


def is_ml_unit(series):
    return series.fillna("").str.lower().str.contains(r"\bml\b|milliliter")


def time_window_by_stay():
    if not os.path.exists(BASE_FEATURE_FILE):
        raise FileNotFoundError(
            f"Missing base landmark matrix: {BASE_FEATURE_FILE}\n"
            "Run v0.4E feature engineering before v0.4G care-process extraction."
        )
    landmarks = pd.read_csv(BASE_FEATURE_FILE, usecols=["stay_id", "landmark_time"])
    landmarks["landmark_time"] = pd.to_datetime(landmarks["landmark_time"], errors="coerce")
    windows = (
        landmarks.dropna(subset=["landmark_time"])
        .groupby("stay_id")
        .agg(
            min_extract_time=("landmark_time", lambda s: s.min() - pd.Timedelta(hours=72)),
            max_extract_time=("landmark_time", "max"),
        )
        .reset_index()
    )
    return windows


def apply_stay_time_window(events, windows, time_col):
    merged = events.merge(windows, on="stay_id", how="inner")
    keep = (
        merged[time_col].ge(merged["min_extract_time"])
        & merged[time_col].le(merged["max_extract_time"])
    )
    out = merged.loc[keep].drop(columns=["min_extract_time", "max_extract_time"])
    return out


# %% Load strict cohort and item maps

cohort = pd.read_csv(
    os.path.join(SOURCE_DATA_PATH, "clabsi_cohort_v0_3a.csv"),
    usecols=["subject_id", "hadm_id", "stay_id", "clabsi"],
)
cohort_hadm = set(cohort["hadm_id"].dropna().astype(int).values)
cohort_stays = set(cohort["stay_id"].dropna().astype(int).values)
windows = time_window_by_stay()

d_items = pd.read_csv(
    os.path.join(ICU, "d_items.csv.gz"),
    usecols=["itemid", "label", "category", "linksto"],
)
d_items["label_lc"] = d_items["label"].fillna("").str.lower()
d_items["category_lc"] = d_items["category"].fillna("").str.lower()

linecare_item_map = d_items[
    d_items["category"].fillna("").eq("Access Lines - Invasive")
    & d_items["label"].fillna("").str.contains(LINECARE_INCLUDE_PATTERN)
    & ~d_items["label"].fillna("").str.contains(LINECARE_EXCLUDE_PATTERN)
].copy()
linecare_item_map["linecare_event_type"] = linecare_item_map["label"].apply(classify_linecare_event)
linecare_itemids = set(linecare_item_map["itemid"].astype(int).values)

caregiver_item_map = d_items[
    d_items["itemid"].isin(all_vital_ids | linecare_itemids)
].copy()
caregiver_itemids = set(caregiver_item_map["itemid"].astype(int).values)

item_map = pd.concat(
    [
        caregiver_item_map.assign(run13_source="caregiver_chartevents"),
        linecare_item_map.assign(run13_source="linecare_datetimeevents"),
    ],
    ignore_index=True,
)
item_map[["run13_source", "itemid", "label", "category", "linksto"]].drop_duplicates().to_csv(
    ITEM_MAP_FILE,
    index=False,
)

print(f"Strict v0.3a cohort loaded: {cohort.shape}")
print(f"Unique ICU stays: {cohort['stay_id'].nunique():,}")
print(f"Strict CLABSI-positive stays: {cohort['clabsi'].sum():,}")
print(f"Caregiver chart itemids: {len(caregiver_itemids):,}")
print(f"Line-care datetime itemids: {len(linecare_itemids):,}")


# %% Extract caregiver exposure from chartevents

if os.path.exists(CAREGIVER_CACHE_FILE):
    print("")
    print(f"Caregiver chartevents extraction already exists, reading: {CAREGIVER_CACHE_FILE}")
    caregiver_events = pd.read_pickle(CAREGIVER_CACHE_FILE)
else:
    print("")
    print("Extracting caregiver exposure from chartevents...")
    caregiver_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(ICU, "chartevents.csv.gz"),
            usecols=["subject_id", "stay_id", "caregiver_id", "charttime", "itemid"],
            chunksize=750000,
            low_memory=False,
            parse_dates=["charttime"],
        ),
        start=1,
    ):
        filtered = chunk[
            chunk["stay_id"].isin(cohort_stays)
            & chunk["itemid"].isin(caregiver_itemids)
            & chunk["caregiver_id"].notna()
            & chunk["charttime"].notna()
        ].copy()
        if len(filtered) > 0:
            filtered = apply_stay_time_window(filtered, windows, "charttime")
            filtered = filtered.drop_duplicates(["stay_id", "charttime", "caregiver_id", "itemid"])
            caregiver_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} chart chunks | retained rows: {matched_rows:,}")

    caregiver_events = (
        pd.concat(caregiver_chunks, ignore_index=True)
        if caregiver_chunks
        else pd.DataFrame(columns=["subject_id", "stay_id", "caregiver_id", "charttime", "itemid"])
    )
    caregiver_events.to_pickle(CAREGIVER_CACHE_FILE)
    print(f"Saved caregiver chartevents: {CAREGIVER_CACHE_FILE}")

print(f"Total caregiver chart rows retained: {len(caregiver_events):,}")


# %% Extract line-care documentation from datetimeevents

if os.path.exists(LINECARE_CACHE_FILE):
    print("")
    print(f"Line-care datetimeevents extraction already exists, reading: {LINECARE_CACHE_FILE}")
    linecare_events = pd.read_pickle(LINECARE_CACHE_FILE)
else:
    print("")
    print("Extracting line-care documentation from datetimeevents...")
    linecare_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(ICU, "datetimeevents.csv.gz"),
            usecols=["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "itemid", "value"],
            chunksize=500000,
            low_memory=False,
            parse_dates=["charttime"],
        ),
        start=1,
    ):
        filtered = chunk[
            chunk["stay_id"].isin(cohort_stays)
            & chunk["itemid"].isin(linecare_itemids)
            & chunk["charttime"].notna()
        ].copy()
        if len(filtered) > 0:
            filtered = apply_stay_time_window(filtered, windows, "charttime")
            filtered = filtered.merge(
                linecare_item_map[["itemid", "label", "linecare_event_type"]],
                on="itemid",
                how="left",
            )
            linecare_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} datetime chunks | retained rows: {matched_rows:,}")

    linecare_events = (
        pd.concat(linecare_chunks, ignore_index=True)
        if linecare_chunks
        else pd.DataFrame()
    )
    linecare_events.to_pickle(LINECARE_CACHE_FILE)
    print(f"Saved line-care datetimeevents: {LINECARE_CACHE_FILE}")

print(f"Total line-care rows retained: {len(linecare_events):,}")


# %% Extract fluid inputs from inputevents

if os.path.exists(FLUID_INPUT_CACHE_FILE):
    print("")
    print(f"Fluid input extraction already exists, reading: {FLUID_INPUT_CACHE_FILE}")
    fluid_inputs = pd.read_pickle(FLUID_INPUT_CACHE_FILE)
else:
    print("")
    print("Extracting fluid inputs from inputevents...")
    input_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(ICU, "inputevents.csv.gz"),
            usecols=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "caregiver_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "amountuom",
                "statusdescription",
            ],
            chunksize=500000,
            low_memory=False,
            parse_dates=["starttime", "endtime"],
        ),
        start=1,
    ):
        filtered = chunk[
            chunk["stay_id"].isin(cohort_stays)
            & chunk["starttime"].notna()
            & chunk["amount"].notna()
            & is_ml_unit(chunk["amountuom"])
            & ~chunk["statusdescription"].fillna("").str.lower().str.contains("rewritten|cancel")
        ].copy()
        if len(filtered) > 0:
            filtered = apply_stay_time_window(filtered, windows, "starttime")
            filtered["fluid_input_ml"] = pd.to_numeric(filtered["amount"], errors="coerce")
            input_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} input chunks | retained rows: {matched_rows:,}")

    fluid_inputs = (
        pd.concat(input_chunks, ignore_index=True)
        if input_chunks
        else pd.DataFrame()
    )
    fluid_inputs.to_pickle(FLUID_INPUT_CACHE_FILE)
    print(f"Saved fluid inputs: {FLUID_INPUT_CACHE_FILE}")

print(f"Total fluid input rows retained: {len(fluid_inputs):,}")


# %% Extract fluid outputs from outputevents

if os.path.exists(FLUID_OUTPUT_CACHE_FILE):
    print("")
    print(f"Fluid output extraction already exists, reading: {FLUID_OUTPUT_CACHE_FILE}")
    fluid_outputs = pd.read_pickle(FLUID_OUTPUT_CACHE_FILE)
else:
    print("")
    print("Extracting fluid outputs from outputevents...")
    output_chunks = []
    matched_rows = 0

    output_item_map = d_items[["itemid", "label", "category"]].copy()
    output_item_map["fluid_output_type"] = output_item_map["label"].apply(classify_output_event)

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(ICU, "outputevents.csv.gz"),
            usecols=["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "itemid", "value", "valueuom"],
            chunksize=500000,
            low_memory=False,
            parse_dates=["charttime"],
        ),
        start=1,
    ):
        filtered = chunk[
            chunk["stay_id"].isin(cohort_stays)
            & chunk["charttime"].notna()
            & chunk["value"].notna()
            & is_ml_unit(chunk["valueuom"])
        ].copy()
        if len(filtered) > 0:
            filtered = apply_stay_time_window(filtered, windows, "charttime")
            filtered = filtered.merge(output_item_map, on="itemid", how="left")
            filtered["fluid_output_ml"] = pd.to_numeric(filtered["value"], errors="coerce")
            output_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} output chunks | retained rows: {matched_rows:,}")

    fluid_outputs = (
        pd.concat(output_chunks, ignore_index=True)
        if output_chunks
        else pd.DataFrame()
    )
    fluid_outputs.to_pickle(FLUID_OUTPUT_CACHE_FILE)
    print(f"Saved fluid outputs: {FLUID_OUTPUT_CACHE_FILE}")

print(f"Total fluid output rows retained: {len(fluid_outputs):,}")


# %% Save audit and counts

audit = pd.DataFrame([{
    "source_cohort_stays": int(cohort["stay_id"].nunique()),
    "source_strict_positive_stays": int(cohort["clabsi"].sum()),
    "caregiver_itemids": ", ".join(str(x) for x in sorted(caregiver_itemids)),
    "linecare_itemids": ", ".join(str(x) for x in sorted(linecare_itemids)),
    "caregiver_chart_rows": int(len(caregiver_events)),
    "linecare_rows": int(len(linecare_events)),
    "fluid_input_rows": int(len(fluid_inputs)),
    "fluid_output_rows": int(len(fluid_outputs)),
    "caregiver_cache_file": CAREGIVER_CACHE_FILE,
    "linecare_cache_file": LINECARE_CACHE_FILE,
    "fluid_input_cache_file": FLUID_INPUT_CACHE_FILE,
    "fluid_output_cache_file": FLUID_OUTPUT_CACHE_FILE,
}])
audit.to_csv(AUDIT_FILE, index=False)

count_rows = []
if len(caregiver_events):
    count_rows.append({
        "source": "chartevents",
        "name": "unique_caregivers",
        "rows": int(caregiver_events["caregiver_id"].nunique()),
    })
    count_rows.append({
        "source": "chartevents",
        "name": "caregiver_chart_rows",
        "rows": int(len(caregiver_events)),
    })
if len(linecare_events):
    for name, count in linecare_events["linecare_event_type"].value_counts().items():
        count_rows.append({"source": "datetimeevents_linecare", "name": name, "rows": int(count)})
if len(fluid_outputs):
    for name, count in fluid_outputs["fluid_output_type"].value_counts().items():
        count_rows.append({"source": "outputevents", "name": name, "rows": int(count)})
if len(fluid_inputs):
    count_rows.append({"source": "inputevents", "name": "fluid_input_rows", "rows": int(len(fluid_inputs))})
pd.DataFrame(count_rows).to_csv(COUNTS_FILE, index=False)

print("")
print(f"Care-process item map saved to: {ITEM_MAP_FILE}")
print(f"Extraction audit saved to:      {AUDIT_FILE}")
print(f"Extraction counts saved to:     {COUNTS_FILE}")
print("")
print("Data Extraction 01 v0.4G Care Process complete.")

