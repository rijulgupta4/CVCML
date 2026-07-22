# %% Imports and paths

import os

import numpy as np
import pandas as pd


PROJECT_PATH = r"C:\path\to\CVCML"
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4e")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4g")

os.makedirs(DATA_PATH, exist_ok=True)

BASE_FEATURE_FILE = os.path.join(SOURCE_DATA_PATH, "clabsi_landmark_features_v0_4e.csv")
CAREGIVER_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_caregiver_chartevents_long.pkl")
LINECARE_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_linecare_datetimeevents_long.pkl")
FLUID_INPUT_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_fluid_inputevents_long.pkl")
FLUID_OUTPUT_CACHE_FILE = os.path.join(DATA_PATH, "v0_4g_fluid_outputevents_long.pkl")

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4g.csv")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4g_care_process_feature_audit.csv")
MISSINGNESS_FILE = os.path.join(DATA_PATH, "v0_4g_care_process_feature_missingness.csv")
ROW_SUMMARY_FILE = os.path.join(DATA_PATH, "v0_4g_landmark_row_summary.csv")

LOOKBACK_WINDOWS = [24, 48, 72]


# %% Helpers

def safe_to_datetime(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def merge_window(events, landmarks, time_col, window_hours, entity_cols=None):
    keys = ["stay_id", "landmark_hour"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=keys)

    entity_cols = entity_cols or ["stay_id"]
    landmark_cols = list(dict.fromkeys(entity_cols + keys + ["landmark_time"]))
    merged = events.merge(landmarks[landmark_cols], on=entity_cols, how="inner")
    if len(merged) == 0:
        return pd.DataFrame(columns=keys)

    merged[time_col] = pd.to_datetime(merged[time_col], errors="coerce")
    merged["window_start"] = merged["landmark_time"] - pd.Timedelta(hours=window_hours)
    keep = (
        merged[time_col].notna()
        & merged[time_col].ge(merged["window_start"])
        & merged[time_col].le(merged["landmark_time"])
    )
    return merged.loc[keep].copy()


def add_caregiver_features(events, landmarks, window_hours):
    keys = ["stay_id", "landmark_hour"]
    merged = merge_window(events, landmarks, "charttime", window_hours, entity_cols=["stay_id"])
    if len(merged) == 0:
        return pd.DataFrame(columns=keys)

    prefix = f"caregiver_{window_hours}h"
    base = (
        merged.groupby(keys)
        .agg(
            caregiver_event_count=("caregiver_id", "size"),
            caregiver_unique_count=("caregiver_id", "nunique"),
            caregiver_item_count=("itemid", "nunique"),
        )
        .reset_index()
    )
    base = base.rename(
        columns={
            "caregiver_event_count": f"{prefix}_event_count",
            "caregiver_unique_count": f"{prefix}_unique_count",
            "caregiver_item_count": f"{prefix}_item_count",
        }
    )

    caregiver_counts = (
        merged.groupby(keys + ["caregiver_id"])
        .size()
        .reset_index(name="caregiver_rows")
    )
    dominant = (
        caregiver_counts.groupby(keys)["caregiver_rows"]
        .max()
        .reset_index()
        .rename(columns={"caregiver_rows": f"{prefix}_dominant_rows"})
    )
    base = base.merge(dominant, on=keys, how="left")
    base[f"{prefix}_dominant_fraction"] = (
        base[f"{prefix}_dominant_rows"] / base[f"{prefix}_event_count"].replace(0, np.nan)
    )
    base[f"{prefix}_handoff_count"] = (base[f"{prefix}_unique_count"] - 1).clip(lower=0)
    return base.drop(columns=[f"{prefix}_dominant_rows"])


def add_linecare_features(events, landmarks, window_hours):
    keys = ["stay_id", "landmark_hour"]
    merged = merge_window(events, landmarks, "charttime", window_hours, entity_cols=["stay_id"])
    if len(merged) == 0:
        return pd.DataFrame(columns=keys)

    prefix = f"linecare_{window_hours}h"
    base = (
        merged.groupby(keys)
        .agg(
            linecare_event_count=("itemid", "size"),
            linecare_unique_event_types=("linecare_event_type", "nunique"),
            linecare_caregiver_unique_count=("caregiver_id", "nunique"),
            last_linecare_time=("charttime", "max"),
        )
        .reset_index()
    )
    base = base.merge(landmarks[keys + ["landmark_time"]], on=keys, how="left")
    base[f"{prefix}_hours_since_last"] = (
        (base["landmark_time"] - base["last_linecare_time"]).dt.total_seconds() / 3600
    )
    base = base.drop(columns=["last_linecare_time", "landmark_time"])
    base = base.rename(
        columns={
            "linecare_event_count": f"{prefix}_event_count",
            "linecare_unique_event_types": f"{prefix}_unique_event_types",
            "linecare_caregiver_unique_count": f"{prefix}_caregiver_unique_count",
        }
    )

    by_type = (
        merged.assign(value=1)
        .pivot_table(
            index=keys,
            columns="linecare_event_type",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    by_type.columns = [
        col if col in keys else f"{prefix}_{col}_count"
        for col in by_type.columns
    ]
    base = base.merge(by_type, on=keys, how="left")
    base[f"{prefix}_any"] = 1.0
    return base


def add_fluid_features(inputs, outputs, landmarks, window_hours):
    keys = ["stay_id", "landmark_hour"]
    prefix = f"fluid_{window_hours}h"

    frames = []
    input_merged = merge_window(inputs, landmarks, "starttime", window_hours, entity_cols=["stay_id"])
    if len(input_merged):
        input_features = (
            input_merged.groupby(keys)
            .agg(
                fluid_input_ml=("fluid_input_ml", "sum"),
                fluid_input_event_count=("fluid_input_ml", "size"),
                fluid_input_caregiver_unique_count=("caregiver_id", "nunique"),
            )
            .reset_index()
            .rename(
                columns={
                    "fluid_input_ml": f"{prefix}_input_ml",
                    "fluid_input_event_count": f"{prefix}_input_event_count",
                    "fluid_input_caregiver_unique_count": f"{prefix}_input_caregiver_unique_count",
                }
            )
        )
        frames.append(input_features)

    output_merged = merge_window(outputs, landmarks, "charttime", window_hours, entity_cols=["stay_id"])
    if len(output_merged):
        output_features = (
            output_merged.groupby(keys)
            .agg(
                fluid_output_ml=("fluid_output_ml", "sum"),
                fluid_output_event_count=("fluid_output_ml", "size"),
                fluid_output_caregiver_unique_count=("caregiver_id", "nunique"),
            )
            .reset_index()
            .rename(
                columns={
                    "fluid_output_ml": f"{prefix}_output_ml",
                    "fluid_output_event_count": f"{prefix}_output_event_count",
                    "fluid_output_caregiver_unique_count": f"{prefix}_output_caregiver_unique_count",
                }
            )
        )
        urine = output_merged[output_merged["fluid_output_type"].eq("urine_output")]
        if len(urine):
            urine_features = (
                urine.groupby(keys)["fluid_output_ml"]
                .sum()
                .reset_index()
                .rename(columns={"fluid_output_ml": f"{prefix}_urine_output_ml"})
            )
            output_features = output_features.merge(urine_features, on=keys, how="left")
        frames.append(output_features)

    if not frames:
        return pd.DataFrame(columns=keys)

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=keys, how="outer")

    input_col = f"{prefix}_input_ml"
    output_col = f"{prefix}_output_ml"
    if input_col in out.columns or output_col in out.columns:
        out[f"{prefix}_net_ml"] = out.get(input_col, 0).fillna(0) - out.get(output_col, 0).fillna(0)
    return out


def fill_count_like_features(df, feature_cols):
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        if "hours_since_last" in col or "dominant_fraction" in col:
            continue
        out[col] = out[col].fillna(0)
    return out


# %% Load base matrix and care-process extracts

print("Loading v0.4E landmark matrix...")
featured = pd.read_csv(BASE_FEATURE_FILE)
featured = safe_to_datetime(
    featured,
    ["starttime", "endtime", "landmark_time", "strict_culture_time", "pragmatic_culture_time"],
)
landmarks = featured[["stay_id", "landmark_hour", "landmark_time"]].copy()
print(f"Base landmark matrix: {featured.shape}")

required_files = [
    CAREGIVER_CACHE_FILE,
    LINECARE_CACHE_FILE,
    FLUID_INPUT_CACHE_FILE,
    FLUID_OUTPUT_CACHE_FILE,
]
missing = [path for path in required_files if not os.path.exists(path)]
if missing:
    raise FileNotFoundError(
        "Missing v0.4G care-process extract(s):\n"
        + "\n".join(missing)
        + "\nRun `Data Extraction 01 v0.4G Care Process.py` first."
    )

caregiver_events = pd.read_pickle(CAREGIVER_CACHE_FILE)
linecare_events = pd.read_pickle(LINECARE_CACHE_FILE)
fluid_inputs = pd.read_pickle(FLUID_INPUT_CACHE_FILE)
fluid_outputs = pd.read_pickle(FLUID_OUTPUT_CACHE_FILE)

caregiver_events = safe_to_datetime(caregiver_events, ["charttime"])
linecare_events = safe_to_datetime(linecare_events, ["charttime"])
fluid_inputs = safe_to_datetime(fluid_inputs, ["starttime", "endtime"])
fluid_outputs = safe_to_datetime(fluid_outputs, ["charttime"])

print(f"Caregiver chart rows: {len(caregiver_events):,}")
print(f"Line-care rows:       {len(linecare_events):,}")
print(f"Fluid input rows:     {len(fluid_inputs):,}")
print(f"Fluid output rows:    {len(fluid_outputs):,}")


# %% Aggregate care-process windows

care_process_cols = []

for window_hours in LOOKBACK_WINDOWS:
    print("")
    print(f"Aggregating {window_hours}h caregiver window...")
    caregiver_features = add_caregiver_features(caregiver_events, landmarks, window_hours)
    featured = featured.merge(caregiver_features, on=["stay_id", "landmark_hour"], how="left")
    care_process_cols.extend([c for c in caregiver_features.columns if c not in ["stay_id", "landmark_hour"]])

    print(f"Aggregating {window_hours}h line-care window...")
    linecare_features = add_linecare_features(linecare_events, landmarks, window_hours)
    featured = featured.merge(linecare_features, on=["stay_id", "landmark_hour"], how="left")
    care_process_cols.extend([c for c in linecare_features.columns if c not in ["stay_id", "landmark_hour"]])

    print(f"Aggregating {window_hours}h fluid-balance window...")
    fluid_features = add_fluid_features(fluid_inputs, fluid_outputs, landmarks, window_hours)
    featured = featured.merge(fluid_features, on=["stay_id", "landmark_hour"], how="left")
    care_process_cols.extend([c for c in fluid_features.columns if c not in ["stay_id", "landmark_hour"]])

care_process_cols = sorted(set(care_process_cols))
featured = fill_count_like_features(featured, care_process_cols)


# %% Save feature matrix and audits

featured.to_csv(FEATURE_FILE, index=False)

audit = pd.DataFrame([{
    "base_rows": int(len(landmarks)),
    "base_columns": int(pd.read_csv(BASE_FEATURE_FILE, nrows=0).shape[1]),
    "final_rows": int(len(featured)),
    "final_columns": int(featured.shape[1]),
    "care_process_features_added": int(len(care_process_cols)),
    "caregiver_features": int(sum(col.startswith("caregiver_") for col in care_process_cols)),
    "linecare_features": int(sum(col.startswith("linecare_") for col in care_process_cols)),
    "fluid_features": int(sum(col.startswith("fluid_") for col in care_process_cols)),
    "caregiver_chart_rows": int(len(caregiver_events)),
    "linecare_rows": int(len(linecare_events)),
    "fluid_input_rows": int(len(fluid_inputs)),
    "fluid_output_rows": int(len(fluid_outputs)),
    "feature_file": FEATURE_FILE,
}])
audit.to_csv(AUDIT_FILE, index=False)

missingness = (
    featured[care_process_cols]
    .isna()
    .mean()
    .reset_index()
    .rename(columns={"index": "feature", 0: "missing_fraction"})
)
nonzero = []
for col in care_process_cols:
    series = pd.to_numeric(featured[col], errors="coerce")
    nonzero.append({
        "feature": col,
        "nonzero_fraction": float(series.fillna(0).ne(0).mean()),
        "mean": float(series.mean()) if series.notna().any() else np.nan,
        "max": float(series.max()) if series.notna().any() else np.nan,
    })
nonzero = pd.DataFrame(nonzero)
missingness = missingness.merge(nonzero, on="feature", how="left")
missingness.to_csv(MISSINGNESS_FILE, index=False)

row_summary = []
for horizon in [24, 48, 72, 168]:
    target_col = f"future_clabsi_{horizon}h"
    if target_col in featured.columns:
        row_summary.append({
            "horizon_hours": horizon,
            "landmark_rows": int(len(featured)),
            "positive_rows": int(featured[target_col].sum()),
            "positive_fraction": float(featured[target_col].mean()),
        })
pd.DataFrame(row_summary).to_csv(ROW_SUMMARY_FILE, index=False)

print("")
print(f"Saved v0.4G feature matrix: {FEATURE_FILE}")
print(f"Feature audit saved to:     {AUDIT_FILE}")
print(f"Missingness saved to:       {MISSINGNESS_FILE}")
print(f"Row summary saved to:       {ROW_SUMMARY_FILE}")
print("")
print("Feature Engineering 03 v0.4G Care Process complete.")

