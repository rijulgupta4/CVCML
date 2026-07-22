# %% Imports and paths

import os

import numpy as np
import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4b")

os.makedirs(DATA_PATH, exist_ok=True)

LANDMARK_HOURS = [48, 72, 96, 120, 144, 168, 240]
LOOKBACK_WINDOWS = [24, 48]
PREDICTION_HORIZONS = [24, 48, 72, 168]


# %% Helpers

def safe_to_datetime(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def add_window_features(events, landmarks, event_name_col, value_col, window_hours, entity_cols):
    """Aggregate event rows into rolling landmark features."""
    keys = ["stay_id", "landmark_hour", event_name_col]
    landmark_cols = list(dict.fromkeys(entity_cols + ["stay_id", "landmark_hour", "landmark_time"]))
    merged = events.merge(
        landmarks[landmark_cols],
        on=entity_cols,
        how="inner",
    )
    merged["hours_since"] = (
        (merged["landmark_time"] - merged["charttime"]).dt.total_seconds() / 3600
    )
    merged = merged[
        (merged["hours_since"] >= 0)
        & (merged["hours_since"] <= window_hours)
    ].copy()

    if len(merged) == 0:
        return pd.DataFrame(columns=["stay_id", "landmark_hour"])

    merged = merged.sort_values(keys + ["charttime"])
    features = (
        merged
        .groupby(keys)
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

    pivot = features.pivot_table(
        index=["stay_id", "landmark_hour"],
        columns=event_name_col,
        values=[
            "mean_val",
            "min_val",
            "max_val",
            "last_val",
            "trend",
            "count",
            "hours_since_last",
        ],
    )
    pivot.columns = [
        f"{event}_{metric.replace('_val', '')}_{window_hours}h"
        for metric, event in pivot.columns
    ]
    return pivot.reset_index()


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


# %% Load v0.3a strict cohort

cohort = pd.read_csv(os.path.join(SOURCE_DATA_PATH, "clabsi_cohort_v0_3a.csv"))
cohort = safe_to_datetime(
    cohort,
    ["starttime", "endtime", "culture_time", "strict_culture_time", "pragmatic_culture_time"],
)
cohort["clabsi"] = cohort["clabsi"].astype(int)

print(f"Strict v0.3a cohort loaded: {cohort.shape}")
print(f"Strict CLABSI-positive stays: {cohort['clabsi'].sum():,}")
print(f"Strict CLABSI-negative stays: {(cohort['clabsi'] == 0).sum():,}")


# %% Build landmark rows with multiple prospective horizons

rows = []
for _, row in cohort.iterrows():
    starttime = row["starttime"]
    endtime = row["endtime"]
    culture_time = row["strict_culture_time"] if row["clabsi"] == 1 else pd.NaT

    for landmark_hour in LANDMARK_HOURS:
        landmark_time = starttime + pd.Timedelta(hours=landmark_hour)

        if landmark_time >= endtime:
            continue
        if pd.notna(culture_time) and culture_time <= landmark_time:
            continue

        new_row = row.to_dict()
        new_row.update({
            "landmark_hour": landmark_hour,
            "landmark_time": landmark_time,
            "dwell_at_landmark_hours": landmark_hour,
        })

        if pd.notna(culture_time):
            time_to_event_hours = (culture_time - landmark_time).total_seconds() / 3600
        else:
            time_to_event_hours = np.nan
        new_row["time_to_event_hours"] = time_to_event_hours

        for horizon in PREDICTION_HORIZONS:
            window_end = min(endtime, landmark_time + pd.Timedelta(hours=horizon))
            target = int(
                pd.notna(culture_time)
                and culture_time > landmark_time
                and culture_time <= window_end
            )
            new_row[f"future_clabsi_{horizon}h"] = target
            new_row[f"prediction_window_end_{horizon}h"] = window_end

        rows.append(new_row)

landmarks = pd.DataFrame(rows)

print("")
print("Landmark row construction:")
print(f"  Landmark rows:            {len(landmarks):,}")
print(f"  Unique stays represented: {landmarks['stay_id'].nunique():,}")
for horizon in PREDICTION_HORIZONS:
    target_col = f"future_clabsi_{horizon}h"
    print(
        f"  {horizon:>3}h future-positive rows: "
        f"{landmarks[target_col].sum():,} ({landmarks[target_col].mean() * 100:.2f}%)"
    )


# %% Define lab and vital item IDs

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


# %% Load extracted lab and vital rows

print("")
print("Loading extracted labs for landmark windows...")
lab_cache_file = os.path.join(DATA_PATH, "v0_4b_labs_long.pkl")

if not os.path.exists(lab_cache_file):
    raise FileNotFoundError(
        f"Missing extracted labs file: {lab_cache_file}\n"
        "Run `Data Extraction 01 v0.4B Vitals.py` before feature engineering."
    )

labevents = pd.read_pickle(lab_cache_file)

print(f"  Total lab rows: {len(labevents):,}")
if len(labevents):
    print(labevents["lab_name"].value_counts().to_string())


print("")
print("Loading extracted vitals for landmark windows...")
vital_cache_file = os.path.join(DATA_PATH, "v0_4b_vitals_long.pkl")

if not os.path.exists(vital_cache_file):
    raise FileNotFoundError(
        f"Missing extracted vitals file: {vital_cache_file}\n"
        "Run `Data Extraction 01 v0.4B Vitals.py` before feature engineering."
    )

vitals = pd.read_pickle(vital_cache_file)

print(f"  Total cleaned vital rows: {len(vitals):,}")
if len(vitals):
    print(vitals["vital_name"].value_counts().to_string())


# %% Aggregate labs and vitals for each landmark row

landmark_featured = landmarks.copy()

for window_hours in LOOKBACK_WINDOWS:
    print("")
    print(f"Aggregating {window_hours}h lab window...")
    lab_features = add_window_features(
        labevents,
        landmarks,
        event_name_col="lab_name",
        value_col="valuenum",
        window_hours=window_hours,
        entity_cols=["subject_id", "hadm_id"],
    )
    landmark_featured = landmark_featured.merge(
        lab_features,
        on=["stay_id", "landmark_hour"],
        how="left",
    )

    print(f"Aggregating {window_hours}h vital window...")
    vital_features = add_window_features(
        vitals,
        landmarks,
        event_name_col="vital_name",
        value_col="vital_value",
        window_hours=window_hours,
        entity_cols=["subject_id", "stay_id"],
    )
    landmark_featured = landmark_featured.merge(
        vital_features,
        on=["stay_id", "landmark_hour"],
        how="left",
    )


# %% Missingness indicators and clinically interpretable flags

for lab in LAB_ITEMS:
    for window_hours in LOOKBACK_WINDOWS:
        last_col = f"{lab}_last_{window_hours}h"
        if last_col in landmark_featured.columns:
            landmark_featured[f"{lab}_measured_{window_hours}h"] = (
                landmark_featured[last_col].notna().astype(int)
            )

for vital in VITAL_ITEMS:
    for window_hours in LOOKBACK_WINDOWS:
        last_col = f"{vital}_last_{window_hours}h"
        if last_col in landmark_featured.columns:
            landmark_featured[f"{vital}_measured_{window_hours}h"] = (
                landmark_featured[last_col].notna().astype(int)
            )

if "temperature_c_max_24h" in landmark_featured.columns:
    landmark_featured["fever_24h"] = (landmark_featured["temperature_c_max_24h"] >= 38.0).astype(int)
    landmark_featured["hypothermia_24h"] = (landmark_featured["temperature_c_min_24h"] <= 36.0).astype(int)
if "heart_rate_max_24h" in landmark_featured.columns:
    landmark_featured["tachycardia_24h"] = (landmark_featured["heart_rate_max_24h"] >= 100).astype(int)
if "map_min_24h" in landmark_featured.columns:
    landmark_featured["hypotension_map_24h"] = (landmark_featured["map_min_24h"] < 65).astype(int)
if "respiratory_rate_max_24h" in landmark_featured.columns:
    landmark_featured["tachypnea_24h"] = (landmark_featured["respiratory_rate_max_24h"] >= 22).astype(int)


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

landmark_featured["race_consolidated"] = landmark_featured["race"].map(race_map).fillna("Unknown")
landmark_featured = landmark_featured.drop(columns=["race"])

landmark_encoded = pd.get_dummies(
    landmark_featured,
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


# %% Save landmark feature matrix and audit

feature_file = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4b.csv")
landmark_encoded.to_csv(feature_file, index=False)

audit_rows = []
for horizon in PREDICTION_HORIZONS:
    target_col = f"future_clabsi_{horizon}h"
    audit_rows.append({
        "horizon_hours": horizon,
        "landmark_hours": ", ".join(str(x) for x in LANDMARK_HOURS),
        "lookback_windows": ", ".join(str(x) for x in LOOKBACK_WINDOWS),
        "source_cohort_stays": int(cohort["stay_id"].nunique()),
        "source_strict_positive_stays": int(cohort["clabsi"].sum()),
        "landmark_rows": int(len(landmark_encoded)),
        "represented_stays": int(landmark_encoded["stay_id"].nunique()),
        "future_positive_rows": int(landmark_encoded[target_col].sum()),
        "future_positive_row_rate": float(landmark_encoded[target_col].mean()),
        "future_positive_stays": int(
            landmark_encoded.loc[landmark_encoded[target_col] == 1, "stay_id"].nunique()
        ),
        "vital_rows_loaded": int(len(vitals)),
        "lab_rows_loaded": int(len(labevents)),
    })
audit = pd.DataFrame(audit_rows)
audit_file = os.path.join(DATA_PATH, "v0_4b_landmark_feature_audit.csv")
audit.to_csv(audit_file, index=False)

summary_rows = []
for horizon in PREDICTION_HORIZONS:
    target_col = f"future_clabsi_{horizon}h"
    horizon_summary = (
        landmark_encoded
        .groupby("landmark_hour")[target_col]
        .agg(landmark_rows="size", future_positive_rows="sum", future_positive_rate="mean")
        .reset_index()
    )
    horizon_summary["horizon_hours"] = horizon
    summary_rows.append(horizon_summary)
landmark_summary = pd.concat(summary_rows, ignore_index=True)
landmark_summary_file = os.path.join(DATA_PATH, "v0_4b_landmark_row_summary.csv")
landmark_summary.to_csv(landmark_summary_file, index=False)

feature_cols = [
    c for c in landmark_encoded.columns
    if any(token in c for token in ["_24h", "_48h"])
]
missing_report = (
    landmark_encoded[feature_cols].isnull().sum() / len(landmark_encoded) * 100
).round(1)
missing_file = os.path.join(DATA_PATH, "v0_4b_dynamic_feature_missingness.csv")
missing_report.reset_index().rename(columns={"index": "feature", 0: "missing_percent"}).to_csv(
    missing_file,
    index=False,
)

print("")
print(f"Dynamic landmark feature matrix saved to: {feature_file}")
print(f"Audit saved to:                           {audit_file}")
print(f"Landmark row summary saved to:            {landmark_summary_file}")
print(f"Feature missingness saved to:             {missing_file}")
print(f"Shape: {landmark_encoded.shape}")
print("")
print("Feature Engineering 03 v0.4B Landmark Dynamic Vitals complete.")

