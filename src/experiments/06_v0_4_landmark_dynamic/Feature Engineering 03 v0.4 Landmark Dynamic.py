# %% Imports and paths

import os

import numpy as np
import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4")

os.makedirs(DATA_PATH, exist_ok=True)

LANDMARK_HOURS = [48, 72, 96, 120, 144, 168, 240]
LOOKBACK_HOURS = 48
PREDICTION_HORIZON_HOURS = 168


# %% Load v0.3a strict cohort

cohort = pd.read_csv(
    os.path.join(SOURCE_DATA_PATH, "clabsi_cohort_v0_3a.csv"),
    parse_dates=[
        "starttime",
        "endtime",
        "culture_time",
        "pragmatic_culture_time",
        "strict_culture_time",
    ],
)

cohort["culture_time"] = pd.to_datetime(cohort["culture_time"], errors="coerce")
cohort["strict_culture_time"] = pd.to_datetime(cohort["strict_culture_time"], errors="coerce")
cohort["clabsi"] = cohort["clabsi"].astype(int)

print(f"Strict v0.3a cohort loaded: {cohort.shape}")
print(f"Strict CLABSI-positive stays: {cohort['clabsi'].sum():,}")
print(f"Strict CLABSI-negative stays: {(cohort['clabsi'] == 0).sum():,}")


# %% Build landmark rows

rows = []
for _, row in cohort.iterrows():
    starttime = row["starttime"]
    endtime = row["endtime"]
    culture_time = row["culture_time"] if row["clabsi"] == 1 else pd.NaT

    for landmark_hour in LANDMARK_HOURS:
        landmark_time = starttime + pd.Timedelta(hours=landmark_hour)

        # The catheter must still be in place at the landmark.
        if landmark_time >= endtime:
            continue

        # Once strict CLABSI has already occurred, later landmark rows are no longer at risk.
        if pd.notna(culture_time) and culture_time <= landmark_time:
            continue

        prediction_window_end = min(
            endtime,
            landmark_time + pd.Timedelta(hours=PREDICTION_HORIZON_HOURS),
        )

        future_clabsi = int(
            pd.notna(culture_time)
            and culture_time > landmark_time
            and culture_time <= prediction_window_end
        )

        time_to_event_hours = np.nan
        if pd.notna(culture_time):
            time_to_event_hours = (culture_time - landmark_time).total_seconds() / 3600

        new_row = row.to_dict()
        new_row.update({
            "landmark_hour": landmark_hour,
            "landmark_time": landmark_time,
            "lookback_start": landmark_time - pd.Timedelta(hours=LOOKBACK_HOURS),
            "prediction_window_end": prediction_window_end,
            "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
            "dwell_at_landmark_hours": landmark_hour,
            "future_clabsi": future_clabsi,
            "time_to_event_hours": time_to_event_hours,
        })
        rows.append(new_row)

landmarks = pd.DataFrame(rows)

print("")
print("Landmark row construction:")
print(f"  Landmark rows:              {len(landmarks):,}")
print(f"  Unique stays represented:   {landmarks['stay_id'].nunique():,}")
print(f"  Future-positive rows:       {landmarks['future_clabsi'].sum():,} ({landmarks['future_clabsi'].mean() * 100:.2f}%)")
print(f"  Future-positive stays:      {landmarks.loc[landmarks['future_clabsi'] == 1, 'stay_id'].nunique():,}")
print("  Rows by landmark:")
print(landmarks.groupby("landmark_hour")["future_clabsi"].agg(["count", "sum", "mean"]).round(4).to_string())


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

print("")
print("Loading labevents for landmark windows...")
cohort_subjects = set(landmarks["subject_id"].values)
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

print(f"  Total lab rows for represented subjects: {len(labevents):,}")
print("  Rows per lab:")
print(labevents["lab_name"].value_counts().to_string())


# %% Window labs to 48 hours before each landmark

print("")
print("Joining labs to landmark lookback windows...")
labs_merged = labevents.merge(
    landmarks[["subject_id", "stay_id", "landmark_hour", "lookback_start", "landmark_time"]],
    on="subject_id",
    how="inner",
)

labs_windowed = labs_merged[
    (labs_merged["charttime"] >= labs_merged["lookback_start"])
    & (labs_merged["charttime"] <= labs_merged["landmark_time"])
].copy()

labs_windowed["hours_since_lab"] = (
    (labs_windowed["landmark_time"] - labs_windowed["charttime"]).dt.total_seconds() / 3600
)

print(f"  Lab rows before windowing:   {len(labs_merged):,}")
print(f"  Lab rows within lookback:    {len(labs_windowed):,}")
print(f"  Landmark rows with any labs: {labs_windowed[['stay_id', 'landmark_hour']].drop_duplicates().shape[0]:,}")


# %% Aggregate labs for each landmark row

labs_windowed = labs_windowed.sort_values(["stay_id", "landmark_hour", "lab_name", "charttime"])
lab_features = (
    labs_windowed
    .groupby(["stay_id", "landmark_hour", "lab_name"])
    .agg(
        mean_val=("valuenum", "mean"),
        last_val=("valuenum", "last"),
        first_val=("valuenum", "first"),
        lab_count=("valuenum", "size"),
        hours_since_last=("hours_since_lab", "min"),
    )
    .reset_index()
)
lab_features["trend"] = lab_features["last_val"] - lab_features["first_val"]

lab_pivot = lab_features.pivot_table(
    index=["stay_id", "landmark_hour"],
    columns="lab_name",
    values=["mean_val", "last_val", "trend", "lab_count", "hours_since_last"],
)
lab_pivot.columns = [
    f"{lab}_{metric.replace('_val', '')}"
    for metric, lab in lab_pivot.columns
]
lab_pivot = lab_pivot.reset_index()

landmark_featured = landmarks.merge(
    lab_pivot,
    on=["stay_id", "landmark_hour"],
    how="left",
)


# %% Missingness indicators and feature cleanup

for lab in ["wbc", "lactate", "hemoglobin", "platelets", "creatinine", "crp", "albumin"]:
    last_col = f"{lab}_last"
    if last_col in landmark_featured.columns:
        landmark_featured[f"{lab}_measured"] = landmark_featured[last_col].notna().astype(int)

drop_cols = [c for c in landmark_featured.columns if c.startswith("crp") or c.startswith("albumin")]
landmark_featured = landmark_featured.drop(columns=drop_cols)

lab_feature_cols = [
    c for c in landmark_featured.columns
    if any(c.startswith(lab) for lab in ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"])
]
missing_report = (
    landmark_featured[lab_feature_cols].isnull().sum() / len(landmark_featured) * 100
).round(1)

print("")
print("Dynamic lab missing rates:")
print(missing_report.to_string())


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

feature_file = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4.csv")
landmark_encoded.to_csv(feature_file, index=False)

audit = pd.DataFrame([{
    "landmark_hours": ", ".join(str(x) for x in LANDMARK_HOURS),
    "lookback_hours": LOOKBACK_HOURS,
    "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
    "source_cohort_stays": int(cohort["stay_id"].nunique()),
    "source_strict_positive_stays": int(cohort["clabsi"].sum()),
    "landmark_rows": int(len(landmark_encoded)),
    "represented_stays": int(landmark_encoded["stay_id"].nunique()),
    "future_positive_rows": int(landmark_encoded["future_clabsi"].sum()),
    "future_positive_row_rate": float(landmark_encoded["future_clabsi"].mean()),
    "future_positive_stays": int(landmark_encoded.loc[landmark_encoded["future_clabsi"] == 1, "stay_id"].nunique()),
}])
audit_file = os.path.join(DATA_PATH, "v0_4_landmark_feature_audit.csv")
audit.to_csv(audit_file, index=False)

landmark_summary = (
    landmark_encoded
    .groupby("landmark_hour")["future_clabsi"]
    .agg(landmark_rows="size", future_positive_rows="sum", future_positive_rate="mean")
    .reset_index()
)
landmark_summary_file = os.path.join(DATA_PATH, "v0_4_landmark_row_summary.csv")
landmark_summary.to_csv(landmark_summary_file, index=False)

print("")
print(f"Dynamic landmark feature matrix saved to: {feature_file}")
print(f"Audit saved to:                           {audit_file}")
print(f"Landmark row summary saved to:            {landmark_summary_file}")
print(f"Shape: {landmark_encoded.shape}")
print(f"Future-positive rows: {landmark_encoded['future_clabsi'].sum():,} ({landmark_encoded['future_clabsi'].mean() * 100:.2f}%)")
print("")
print("Feature Engineering 03 v0.4 Landmark Dynamic complete.")

