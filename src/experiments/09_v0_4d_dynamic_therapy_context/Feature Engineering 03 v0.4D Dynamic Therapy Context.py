# %% Imports and paths

import os

import numpy as np
import pandas as pd

PROJECT_PATH = r"C:\path\to\CVCML"
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4b")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4d")

os.makedirs(DATA_PATH, exist_ok=True)

BASE_FEATURE_FILE = os.path.join(SOURCE_DATA_PATH, "clabsi_landmark_features_v0_4b.csv")
ANTIBIOTIC_CACHE_FILE = os.path.join(DATA_PATH, "v0_4d_antibiotics_long.pkl")
VASOPRESSOR_CACHE_FILE = os.path.join(DATA_PATH, "v0_4d_vasopressors_long.pkl")

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4d.csv")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4d_therapy_feature_audit.csv")
MISSINGNESS_FILE = os.path.join(DATA_PATH, "v0_4d_therapy_feature_missingness.csv")

LOOKBACK_WINDOWS = [24, 48]
HORIZONS = [24, 48, 72, 168]


# %% Helpers

def safe_to_datetime(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def add_interval_features(
    events,
    landmarks,
    prefix,
    start_col,
    end_col,
    entity_cols,
    flag_cols,
    window_hours,
):
    keys = ["stay_id", "landmark_hour"]
    if len(events) == 0:
        return pd.DataFrame(columns=keys)

    landmark_cols = list(dict.fromkeys(entity_cols + keys + ["landmark_time"]))
    merged = events.merge(landmarks[landmark_cols], on=entity_cols, how="inner")
    if len(merged) == 0:
        return pd.DataFrame(columns=keys)

    merged[start_col] = pd.to_datetime(merged[start_col], errors="coerce")
    merged[end_col] = pd.to_datetime(merged[end_col], errors="coerce")
    merged[end_col] = merged[end_col].fillna(merged[start_col] + pd.Timedelta(hours=24))

    merged["window_start"] = merged["landmark_time"] - pd.Timedelta(hours=window_hours)
    merged["overlaps_window"] = (
        merged[start_col].le(merged["landmark_time"])
        & merged[end_col].ge(merged["window_start"])
    )
    merged["starts_in_window"] = (
        merged[start_col].ge(merged["window_start"])
        & merged[start_col].le(merged["landmark_time"])
    )
    merged["active_at_landmark"] = (
        merged[start_col].le(merged["landmark_time"])
        & merged[end_col].ge(merged["landmark_time"])
    )
    merged["hours_since_last_start"] = (
        (merged["landmark_time"] - merged[start_col]).dt.total_seconds() / 3600
    )

    in_window = merged[merged["overlaps_window"]].copy()
    if len(in_window) == 0:
        return pd.DataFrame(columns=keys)

    base = (
        in_window.groupby(keys)
        .agg(
            any_exposure=("overlaps_window", "max"),
            starts_count=("starts_in_window", "sum"),
            active_at_landmark=("active_at_landmark", "max"),
            hours_since_last_start=("hours_since_last_start", "min"),
        )
        .reset_index()
    )
    base = base.rename(
        columns={
            "any_exposure": f"{prefix}_any_{window_hours}h",
            "starts_count": f"{prefix}_starts_count_{window_hours}h",
            "active_at_landmark": f"{prefix}_active_at_landmark_{window_hours}h",
            "hours_since_last_start": f"{prefix}_hours_since_last_start_{window_hours}h",
        }
    )

    for flag_col in flag_cols:
        if flag_col not in in_window.columns:
            continue
        exposure = (
            in_window.groupby(keys)[flag_col]
            .max()
            .reset_index()
            .rename(columns={flag_col: f"{prefix}_{flag_col}_any_{window_hours}h"})
        )
        active = (
            in_window[in_window["active_at_landmark"]]
            .groupby(keys)[flag_col]
            .max()
            .reset_index()
            .rename(columns={flag_col: f"{prefix}_{flag_col}_active_{window_hours}h"})
        )
        base = base.merge(exposure, on=keys, how="left")
        base = base.merge(active, on=keys, how="left")

    return base


def add_pressor_name_flags(vasopressors):
    df = vasopressors.copy()
    for name in sorted(df["vasopressor_name"].dropna().unique()):
        safe_name = str(name).lower().replace(" ", "_").replace("-", "_")
        df[f"{safe_name}_pressor"] = df["vasopressor_name"].eq(name).astype(int)
    return df


# %% Load Run 8 landmark matrix and therapy extracts

print("Loading v0.4B dynamic landmark matrix...")
landmarks = pd.read_csv(BASE_FEATURE_FILE)
landmarks = safe_to_datetime(
    landmarks,
    ["starttime", "endtime", "landmark_time", "strict_culture_time", "pragmatic_culture_time"],
)
print(f"Base landmark matrix: {landmarks.shape}")

if not os.path.exists(ANTIBIOTIC_CACHE_FILE):
    raise FileNotFoundError(
        f"Missing antibiotic extract: {ANTIBIOTIC_CACHE_FILE}\n"
        "Run `Data Extraction 01 v0.4D Therapy Context.py` first."
    )
if not os.path.exists(VASOPRESSOR_CACHE_FILE):
    raise FileNotFoundError(
        f"Missing vasopressor extract: {VASOPRESSOR_CACHE_FILE}\n"
        "Run `Data Extraction 01 v0.4D Therapy Context.py` first."
    )

antibiotics = pd.read_pickle(ANTIBIOTIC_CACHE_FILE)
vasopressors = pd.read_pickle(VASOPRESSOR_CACHE_FILE)

antibiotics = safe_to_datetime(antibiotics, ["starttime", "stoptime"])
vasopressors = safe_to_datetime(vasopressors, ["starttime", "endtime"])
if len(vasopressors):
    vasopressors = add_pressor_name_flags(vasopressors)

print(f"Antibiotic rows loaded:   {len(antibiotics):,}")
print(f"Vasopressor rows loaded:  {len(vasopressors):,}")


# %% Aggregate therapy exposure around each landmark

featured = landmarks.copy()
therapy_feature_cols = []

antibiotic_flags = [
    "broad_antibiotic",
    "anti_mrsa_antibiotic",
    "antipseudomonal_antibiotic",
    "carbapenem_antibiotic",
    "anaerobe_antibiotic",
]
pressor_flags = [col for col in vasopressors.columns if col.endswith("_pressor")]

for window_hours in LOOKBACK_WINDOWS:
    print("")
    print(f"Aggregating {window_hours}h antibiotic exposure...")
    antibiotic_features = add_interval_features(
        antibiotics,
        landmarks,
        prefix="antibiotic",
        start_col="starttime",
        end_col="stoptime",
        entity_cols=["subject_id", "hadm_id"],
        flag_cols=antibiotic_flags,
        window_hours=window_hours,
    )
    featured = featured.merge(antibiotic_features, on=["stay_id", "landmark_hour"], how="left")
    therapy_feature_cols.extend([c for c in antibiotic_features.columns if c not in ["stay_id", "landmark_hour"]])

    print(f"Aggregating {window_hours}h vasopressor exposure...")
    vasopressor_features = add_interval_features(
        vasopressors,
        landmarks,
        prefix="vasopressor",
        start_col="starttime",
        end_col="endtime",
        entity_cols=["subject_id", "stay_id"],
        flag_cols=pressor_flags,
        window_hours=window_hours,
    )
    featured = featured.merge(vasopressor_features, on=["stay_id", "landmark_hour"], how="left")
    therapy_feature_cols.extend([c for c in vasopressor_features.columns if c not in ["stay_id", "landmark_hour"]])


# %% Fill missing therapy values and add clinically compact flags

therapy_feature_cols = sorted(set(therapy_feature_cols))
for col in therapy_feature_cols:
    if col in featured.columns:
        if "hours_since_last_start" in col:
            featured[col] = featured[col].fillna(np.nan)
        else:
            featured[col] = featured[col].fillna(0).astype(float)

for window_hours in LOOKBACK_WINDOWS:
    antibiotic_any = f"antibiotic_any_{window_hours}h"
    vasopressor_any = f"vasopressor_any_{window_hours}h"
    broad_any = f"antibiotic_broad_antibiotic_any_{window_hours}h"
    pressor_active = f"vasopressor_active_at_landmark_{window_hours}h"

    if antibiotic_any in featured.columns:
        featured[f"therapy_any_antibiotic_{window_hours}h"] = featured[antibiotic_any].fillna(0)
    if vasopressor_any in featured.columns:
        featured[f"therapy_any_vasopressor_{window_hours}h"] = featured[vasopressor_any].fillna(0)
    if broad_any in featured.columns:
        featured[f"therapy_broad_antibiotic_{window_hours}h"] = featured[broad_any].fillna(0)
    if pressor_active in featured.columns:
        featured[f"therapy_active_vasopressor_{window_hours}h"] = featured[pressor_active].fillna(0)
    if antibiotic_any in featured.columns and vasopressor_any in featured.columns:
        featured[f"therapy_antibiotic_plus_pressor_{window_hours}h"] = (
            (featured[antibiotic_any].fillna(0) > 0)
            & (featured[vasopressor_any].fillna(0) > 0)
        ).astype(int)


# %% Save feature matrix and audits

featured.to_csv(FEATURE_FILE, index=False)

audit_rows = []
for horizon in HORIZONS:
    target_col = f"future_clabsi_{horizon}h"
    if target_col not in featured.columns:
        continue
    audit_rows.append({
        "horizon_hours": horizon,
        "landmark_rows": int(len(featured)),
        "represented_stays": int(featured["stay_id"].nunique()),
        "future_positive_rows": int(featured[target_col].sum()),
        "future_positive_row_rate": float(featured[target_col].mean()),
        "antibiotic_rows_loaded": int(len(antibiotics)),
        "vasopressor_rows_loaded": int(len(vasopressors)),
        "therapy_feature_count": int(len([c for c in featured.columns if c.startswith(("antibiotic_", "vasopressor_", "therapy_"))])),
        "antibiotic_any_48h_rate": float(featured.get("antibiotic_any_48h", pd.Series(0, index=featured.index)).mean()),
        "vasopressor_any_48h_rate": float(featured.get("vasopressor_any_48h", pd.Series(0, index=featured.index)).mean()),
    })
pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False)

therapy_cols = [c for c in featured.columns if c.startswith(("antibiotic_", "vasopressor_", "therapy_"))]
missing = (
    featured[therapy_cols].isnull().sum() / len(featured) * 100
).round(1)
missing.reset_index().rename(columns={"index": "feature", 0: "missing_percent"}).to_csv(
    MISSINGNESS_FILE,
    index=False,
)

print("")
print(f"v0.4D therapy landmark matrix saved to: {FEATURE_FILE}")
print(f"Feature audit saved to:                  {AUDIT_FILE}")
print(f"Therapy missingness saved to:            {MISSINGNESS_FILE}")
print(f"Shape: {featured.shape}")
print("")
print("Feature Engineering 03 v0.4D Dynamic Therapy Context complete.")

