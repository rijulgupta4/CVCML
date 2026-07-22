# %% Imports and paths

import re
import sys
import warnings
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print("Missing Run 20 feature engineering dependency:", exc)
    print("Install required packages: pip install pandas numpy")
    sys.exit(1)

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


MIMIC_PATH = Path(r"C:\path\to\mimic-iv")
PROJECT_PATH = Path(r"C:\path\to\CVCML")
HOSP = MIMIC_PATH / "hosp"
ICU = MIMIC_PATH / "icu"
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 20 (v0.5 Dynamic Feature Enrichment)"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RUN18_FEATURE_FILE = DATA_PATH / "v0_5_run18_development_features.csv"
ENRICHED_FEATURE_FILE = DATA_PATH / "v0_5_run20_dynamic_enriched_features.csv"
VITAL_CACHE_FILE = DATA_PATH / "v0_5_run20_vitals_long.pkl"
ANTIBIOTIC_CACHE_FILE = DATA_PATH / "v0_5_run20_antibiotics_long.pkl"
VASOPRESSOR_CACHE_FILE = DATA_PATH / "v0_5_run20_vasopressors_long.pkl"
FEATURE_AUDIT_FILE = OUTPUT_PATH / "v0_5_run20_dynamic_feature_audit.csv"

LOOKBACK_WINDOWS = [24, 48]


# %% Definitions

VITAL_ITEMS = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "temperature_c": [223761, 223762],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "map": [220052, 220181],
}
ALL_VITAL_IDS = [itemid for ids in VITAL_ITEMS.values() for itemid in ids]
ID_TO_VITAL = {itemid: vital for vital, ids in VITAL_ITEMS.items() for itemid in ids}

ANTIBIOTIC_PATTERN = re.compile(
    r"("
    r"vancomycin|cefazolin|cefepime|ceftriaxone|ceftazidime|cefuroxime|ceftaroline|"
    r"cephalexin|cefpodoxime|cef|piperacillin|tazobactam|zosyn|meropenem|imipenem|"
    r"ertapenem|aztreonam|ciprofloxacin|levofloxacin|moxifloxacin|metronidazole|"
    r"clindamycin|linezolid|daptomycin|gentamicin|tobramycin|amikacin|ampicillin|"
    r"amoxicillin|nafcillin|oxacillin|penicillin|doxycycline|azithromycin|"
    r"clarithromycin|trimethoprim|sulfamethoxazole|bactrim|tigecycline|colistin|"
    r"polymyxin"
    r")",
    flags=re.IGNORECASE,
)
BROAD_PATTERN = re.compile(
    r"(vancomycin|cefepime|ceftazidime|ceftaroline|piperacillin|tazobactam|zosyn|"
    r"meropenem|imipenem|ertapenem|aztreonam|ciprofloxacin|levofloxacin|moxifloxacin|"
    r"linezolid|daptomycin|colistin|polymyxin)",
    flags=re.IGNORECASE,
)
ANTI_MRSA_PATTERN = re.compile(r"(vancomycin|linezolid|daptomycin|ceftaroline)", flags=re.IGNORECASE)
ANTIPSEUDOMONAL_PATTERN = re.compile(
    r"(cefepime|ceftazidime|piperacillin|tazobactam|zosyn|meropenem|imipenem|aztreonam|ciprofloxacin|levofloxacin)",
    flags=re.IGNORECASE,
)
CARBAPENEM_PATTERN = re.compile(r"(meropenem|imipenem|ertapenem)", flags=re.IGNORECASE)
ANAEROBE_PATTERN = re.compile(r"(metronidazole|clindamycin|piperacillin|tazobactam|zosyn|meropenem|imipenem)", flags=re.IGNORECASE)
ANTIBIOTIC_FLAGS = [
    "antibiotic_any",
    "broad_antibiotic",
    "anti_mrsa_antibiotic",
    "antipseudomonal_antibiotic",
    "carbapenem_antibiotic",
    "anaerobe_antibiotic",
]

VASOPRESSOR_TERMS = [
    "norepinephrine",
    "epinephrine",
    "phenylephrine",
    "vasopressin",
    "dopamine",
    "dobutamine",
    "milrinone",
]


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


def add_antibiotic_flags(df):
    df = df.copy()
    drug = df["drug"].fillna("")
    df["antibiotic_name"] = drug.str.lower().str.strip()
    df["antibiotic_any"] = 1
    df["broad_antibiotic"] = drug.str.contains(BROAD_PATTERN).astype(int)
    df["anti_mrsa_antibiotic"] = drug.str.contains(ANTI_MRSA_PATTERN).astype(int)
    df["antipseudomonal_antibiotic"] = drug.str.contains(ANTIPSEUDOMONAL_PATTERN).astype(int)
    df["carbapenem_antibiotic"] = drug.str.contains(CARBAPENEM_PATTERN).astype(int)
    df["anaerobe_antibiotic"] = drug.str.contains(ANAEROBE_PATTERN).astype(int)
    return df


def classify_pressor(label):
    text = str(label).lower()
    for term in VASOPRESSOR_TERMS:
        if term in text:
            return term
    return "other_vasoactive"


def make_windows(features, key_cols):
    cols = ["landmark_id", "landmark_time"] + key_cols
    windows = features[cols].dropna(subset=key_cols).copy()
    for key in key_cols:
        windows[key] = windows[key].astype("int64")
    for window_hours in LOOKBACK_WINDOWS:
        windows[f"lookback_start_{window_hours}h"] = windows["landmark_time"] - pd.Timedelta(hours=window_hours)
    return windows


def extract_vitals(windows):
    if VITAL_CACHE_FILE.exists():
        print(f"Loading cached v0.5 vitals: {VITAL_CACHE_FILE}", flush=True)
        return pd.read_pickle(VITAL_CACHE_FILE)

    stay_ids = set(windows["stay_id"].astype("int64"))
    min_time = min(windows[f"lookback_start_{h}h"].min() for h in LOOKBACK_WINDOWS)
    max_time = windows["landmark_time"].max()

    chunks = []
    matched_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(
        ICU / "chartevents.csv.gz",
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        chunksize=750000,
        low_memory=False,
        parse_dates=["charttime"],
    ), start=1):
        chunk = chunk[chunk["stay_id"].notna()].copy()
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        filtered = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(ALL_VITAL_IDS)
            & chunk["valuenum"].notna()
            & (chunk["charttime"] >= min_time)
            & (chunk["charttime"] <= max_time)
        ].copy()
        if len(filtered):
            filtered["vital_name"] = filtered["itemid"].map(ID_TO_VITAL)
            chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  chart chunks scanned: {chunk_idx:,} | matched vitals: {matched_rows:,}", flush=True)

    vitals = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(vitals):
        vitals = apply_vital_cleaning(vitals)
    vitals.to_pickle(VITAL_CACHE_FILE)
    print(f"Cached v0.5 vitals: {VITAL_CACHE_FILE}", flush=True)
    return vitals


def extract_antibiotics(windows):
    if ANTIBIOTIC_CACHE_FILE.exists():
        print(f"Loading cached v0.5 antibiotics: {ANTIBIOTIC_CACHE_FILE}", flush=True)
        return pd.read_pickle(ANTIBIOTIC_CACHE_FILE)

    hadm_ids = set(windows["hadm_id"].astype("int64"))
    min_time = min(windows[f"lookback_start_{h}h"].min() for h in LOOKBACK_WINDOWS)
    max_time = windows["landmark_time"].max()

    chunks = []
    matched_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(
        HOSP / "prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"],
        chunksize=500000,
        low_memory=False,
        parse_dates=["starttime", "stoptime"],
    ), start=1):
        chunk = chunk[chunk["hadm_id"].notna()].copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        filtered = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["starttime"].notna()
            & chunk["drug"].fillna("").str.contains(ANTIBIOTIC_PATTERN)
            & (chunk["starttime"] <= max_time)
            & (chunk["stoptime"].fillna(chunk["starttime"]) >= min_time)
        ].copy()
        if len(filtered):
            filtered = add_antibiotic_flags(filtered)
            chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  prescription chunks scanned: {chunk_idx:,} | matched antibiotics: {matched_rows:,}", flush=True)

    antibiotics = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    antibiotics.to_pickle(ANTIBIOTIC_CACHE_FILE)
    print(f"Cached v0.5 antibiotics: {ANTIBIOTIC_CACHE_FILE}", flush=True)
    return antibiotics


def extract_vasopressors(windows):
    if VASOPRESSOR_CACHE_FILE.exists():
        print(f"Loading cached v0.5 vasopressors: {VASOPRESSOR_CACHE_FILE}", flush=True)
        return pd.read_pickle(VASOPRESSOR_CACHE_FILE)

    d_items = pd.read_csv(ICU / "d_items.csv.gz", usecols=["itemid", "label", "category"])
    d_items["label_lc"] = d_items["label"].fillna("").str.lower()
    item_map = d_items[
        d_items["label_lc"].apply(lambda value: any(term in value for term in VASOPRESSOR_TERMS))
        & ~d_items["label_lc"].str.contains("intubation", na=False)
    ].copy()
    item_map["vasopressor_name"] = item_map["label"].apply(classify_pressor)
    itemids = set(item_map["itemid"].astype(int))
    stay_ids = set(windows["stay_id"].astype("int64"))
    min_time = min(windows[f"lookback_start_{h}h"].min() for h in LOOKBACK_WINDOWS)
    max_time = windows["landmark_time"].max()

    chunks = []
    matched_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(
        ICU / "inputevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "rate", "statusdescription"],
        chunksize=500000,
        low_memory=False,
        parse_dates=["starttime", "endtime"],
    ), start=1):
        chunk = chunk[chunk["stay_id"].notna()].copy()
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        filtered = chunk[
            chunk["stay_id"].isin(stay_ids)
            & chunk["itemid"].isin(itemids)
            & chunk["starttime"].notna()
            & (chunk["starttime"] <= max_time)
            & (chunk["endtime"].fillna(chunk["starttime"]) >= min_time)
        ].copy()
        if len(filtered):
            filtered = filtered.merge(item_map[["itemid", "label", "vasopressor_name"]], on="itemid", how="left")
            chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  input chunks scanned: {chunk_idx:,} | matched vasopressors: {matched_rows:,}", flush=True)

    vasopressors = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    vasopressors.to_pickle(VASOPRESSOR_CACHE_FILE)
    print(f"Cached v0.5 vasopressors: {VASOPRESSOR_CACHE_FILE}", flush=True)
    return vasopressors


def aggregate_point_events_by_key(events, windows, key_col, name_col, value_col, prefix, window_hours):
    if len(events) == 0:
        return pd.DataFrame({"landmark_id": windows["landmark_id"].head(0)})

    events = events[[key_col, "charttime", name_col, value_col]].dropna(subset=[key_col, "charttime"]).copy()
    events[key_col] = events[key_col].astype("int64")
    event_groups = {
        key: group[["charttime", name_col, value_col]].sort_values("charttime")
        for key, group in events.groupby(key_col, sort=False)
    }

    frames = []
    start_col = f"lookback_start_{window_hours}h"
    for idx, (key, group_windows) in enumerate(windows.groupby(key_col, sort=False), start=1):
        group_events = event_groups.get(key)
        if group_events is None or len(group_events) == 0:
            continue

        merged = group_events.merge(group_windows[["landmark_id", "landmark_time", start_col]], how="cross")
        merged = merged[
            (merged["charttime"] >= merged[start_col])
            & (merged["charttime"] <= merged["landmark_time"])
        ].copy()
        if len(merged):
            merged["hours_since"] = (merged["landmark_time"] - merged["charttime"]).dt.total_seconds() / 3600
            merged = merged.sort_values(["landmark_id", name_col, "charttime"])
            features = (
                merged
                .groupby(["landmark_id", name_col])
                .agg(
                    mean_val=(value_col, "mean"),
                    min_val=(value_col, "min"),
                    max_val=(value_col, "max"),
                    last_val=(value_col, "last"),
                    first_val=(value_col, "first"),
                    count=(value_col, "size"),
                    hours_since_last=("hours_since", "min"),
                )
                .reset_index()
            )
            features["trend"] = features["last_val"] - features["first_val"]
            frames.append(features)
        if idx % 1000 == 0:
            print(f"    {prefix} {window_hours}h groups processed: {idx:,}", flush=True)

    if not frames:
        return pd.DataFrame({"landmark_id": windows["landmark_id"].head(0)})
    features = pd.concat(frames, ignore_index=True)
    pivot = features.pivot_table(
        index="landmark_id",
        columns=name_col,
        values=["mean_val", "min_val", "max_val", "last_val", "trend", "count", "hours_since_last"],
    )
    pivot.columns = [
        f"{prefix}_{event}_{metric.replace('_val', '')}_{window_hours}h"
        for metric, event in pivot.columns
    ]
    return pivot.reset_index()


def aggregate_interval_features(events, windows, key_col, start_col, stop_col, flag_cols, prefix, window_hours):
    if len(events) == 0:
        return pd.DataFrame({"landmark_id": windows["landmark_id"].head(0)})

    use_cols = [key_col, start_col, stop_col] + flag_cols
    if "rate" in events.columns:
        use_cols.append("rate")
    events = events[use_cols].dropna(subset=[key_col, start_col]).copy()
    events[key_col] = events[key_col].astype("int64")
    events[stop_col] = events[stop_col].fillna(events[start_col])
    event_groups = {
        key: group.sort_values(start_col)
        for key, group in events.groupby(key_col, sort=False)
    }

    rows = []
    lb_col = f"lookback_start_{window_hours}h"
    for idx, (key, group_windows) in enumerate(windows.groupby(key_col, sort=False), start=1):
        group_events = event_groups.get(key)
        if group_events is None or len(group_events) == 0:
            continue
        merged = group_events.merge(group_windows[["landmark_id", "landmark_time", lb_col]], how="cross")
        overlap = merged[
            (merged[start_col] <= merged["landmark_time"])
            & (merged[stop_col] >= merged[lb_col])
        ].copy()
        if len(overlap):
            started = overlap[(overlap[start_col] >= overlap[lb_col]) & (overlap[start_col] <= overlap["landmark_time"])]
            row = overlap.groupby("landmark_id").size().rename(f"{prefix}_active_rows_{window_hours}h").reset_index()
            for flag in flag_cols:
                active_flag = overlap.groupby("landmark_id")[flag].max().rename(f"{prefix}_{flag}_active_{window_hours}h").reset_index()
                start_flag = started.groupby("landmark_id")[flag].sum().rename(f"{prefix}_{flag}_starts_{window_hours}h").reset_index()
                row = row.merge(active_flag, on="landmark_id", how="left")
                row = row.merge(start_flag, on="landmark_id", how="left")
            if "rate" in overlap.columns:
                rate = overlap.groupby("landmark_id")["rate"].agg(["mean", "max"]).reset_index()
                rate = rate.rename(columns={"mean": f"{prefix}_rate_mean_{window_hours}h", "max": f"{prefix}_rate_max_{window_hours}h"})
                row = row.merge(rate, on="landmark_id", how="left")
            rows.append(row)
        if idx % 1000 == 0:
            print(f"    {prefix} {window_hours}h groups processed: {idx:,}", flush=True)

    if not rows:
        return pd.DataFrame({"landmark_id": windows["landmark_id"].head(0)})
    return pd.concat(rows, ignore_index=True).groupby("landmark_id").max().reset_index()


# %% Load v0.5 features and build windows

print("Loading Run 18 v0.5 feature matrix...", flush=True)
features = pd.read_csv(RUN18_FEATURE_FILE, parse_dates=["landmark_time"])
stay_windows = make_windows(features, ["stay_id"])
hadm_windows = make_windows(features, ["hadm_id"])

print(f"  Landmark rows: {len(features):,}", flush=True)
print(f"  Unique stays:  {features['stay_id'].nunique():,}", flush=True)
print(f"  Unique hadm:   {features['hadm_id'].nunique():,}", flush=True)


# %% Extract source rows

print("", flush=True)
print("Extracting/loading v0.5 vitals...", flush=True)
vitals = extract_vitals(stay_windows)
print(f"  Cleaned vital rows: {len(vitals):,}", flush=True)

print("", flush=True)
print("Extracting/loading v0.5 antibiotics...", flush=True)
antibiotics = extract_antibiotics(hadm_windows)
print(f"  Antibiotic rows: {len(antibiotics):,}", flush=True)

print("", flush=True)
print("Extracting/loading v0.5 vasopressors...", flush=True)
vasopressors = extract_vasopressors(stay_windows)
print(f"  Vasopressor rows: {len(vasopressors):,}", flush=True)


# %% Aggregate dynamic features

enriched = features.copy()
audit_rows = []

for window_hours in LOOKBACK_WINDOWS:
    print("", flush=True)
    print(f"Aggregating vitals over {window_hours}h windows...", flush=True)
    vital_features = aggregate_point_events_by_key(
        vitals,
        stay_windows,
        key_col="stay_id",
        name_col="vital_name",
        value_col="vital_value",
        prefix="vital",
        window_hours=window_hours,
    )
    enriched = enriched.merge(vital_features, on="landmark_id", how="left")
    audit_rows.append({
        "feature_family": f"vitals_{window_hours}h",
        "source_rows": int(len(vitals)),
        "landmarks_with_features": int(vital_features["landmark_id"].nunique()) if "landmark_id" in vital_features else 0,
        "feature_columns": int(max(0, len(vital_features.columns) - 1)),
    })

    print(f"Aggregating antibiotics over {window_hours}h windows...", flush=True)
    antibiotic_features = aggregate_interval_features(
        antibiotics,
        hadm_windows,
        key_col="hadm_id",
        start_col="starttime",
        stop_col="stoptime",
        flag_cols=ANTIBIOTIC_FLAGS,
        prefix="abx",
        window_hours=window_hours,
    )
    enriched = enriched.merge(antibiotic_features, on="landmark_id", how="left")
    audit_rows.append({
        "feature_family": f"antibiotics_{window_hours}h",
        "source_rows": int(len(antibiotics)),
        "landmarks_with_features": int(antibiotic_features["landmark_id"].nunique()) if "landmark_id" in antibiotic_features else 0,
        "feature_columns": int(max(0, len(antibiotic_features.columns) - 1)),
    })

    print(f"Aggregating vasopressors over {window_hours}h windows...", flush=True)
    vaso_flag_cols = ["vasopressor_any"]
    vasopressors_for_features = vasopressors.copy()
    if len(vasopressors_for_features):
        vasopressors_for_features["vasopressor_any"] = 1
    vasopressor_features = aggregate_interval_features(
        vasopressors_for_features,
        stay_windows,
        key_col="stay_id",
        start_col="starttime",
        stop_col="endtime",
        flag_cols=vaso_flag_cols,
        prefix="vaso",
        window_hours=window_hours,
    )
    enriched = enriched.merge(vasopressor_features, on="landmark_id", how="left")
    audit_rows.append({
        "feature_family": f"vasopressors_{window_hours}h",
        "source_rows": int(len(vasopressors)),
        "landmarks_with_features": int(vasopressor_features["landmark_id"].nunique()) if "landmark_id" in vasopressor_features else 0,
        "feature_columns": int(max(0, len(vasopressor_features.columns) - 1)),
    })


# %% Fill count/binary columns and save

count_like = [c for c in enriched.columns if c.endswith("_count_24h") or c.endswith("_count_48h") or "_starts_" in c or "_active_" in c or c.endswith("_active_rows_24h") or c.endswith("_active_rows_48h")]
for col in count_like:
    enriched[col] = enriched[col].fillna(0)

audit = pd.DataFrame(audit_rows)
audit["output_feature_matrix"] = str(ENRICHED_FEATURE_FILE)
audit["vital_cache"] = str(VITAL_CACHE_FILE)
audit["antibiotic_cache"] = str(ANTIBIOTIC_CACHE_FILE)
audit["vasopressor_cache"] = str(VASOPRESSOR_CACHE_FILE)

enriched.to_csv(ENRICHED_FEATURE_FILE, index=False)
audit.to_csv(FEATURE_AUDIT_FILE, index=False)

print("", flush=True)
print(f"Saved enriched feature matrix: {ENRICHED_FEATURE_FILE}", flush=True)
print(f"  Shape: {enriched.shape}", flush=True)
print(f"Saved feature audit: {FEATURE_AUDIT_FILE}", flush=True)
print("Feature Engineering 05 v0.5 Dynamic Enrichment complete.", flush=True)

