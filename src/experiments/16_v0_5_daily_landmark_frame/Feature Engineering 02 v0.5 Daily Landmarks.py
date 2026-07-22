# %% Imports and paths

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_PATH = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 17 (v0.5 Daily Landmark Frame)"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EXPOSURE_FILE = DATA_PATH / "v0_5_catheter_exposure_periods.csv"

MIN_LANDMARK_HOURS = 48
LANDMARK_STEP_HOURS = 24
PREDICTION_HORIZON_HOURS = 168
LOCKBOX_ANCHOR_YEAR_GROUP = "2020 - 2022"


# %% Helpers

def to_datetime_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def split_role(anchor_year_group):
    if str(anchor_year_group) == LOCKBOX_ANCHOR_YEAR_GROUP:
        return "temporal_lockbox"
    return "development"


def classify_status(event_in_window, window_end, observed_end, end_reason):
    if event_in_window:
        return "strict_proxy_event"
    if pd.isna(observed_end):
        return "censored_unknown_end"
    if observed_end <= window_end:
        return f"competing_or_censoring_{end_reason}"
    return "no_event_full_7d_followup"


def make_episode_landmarks(row):
    rows = []
    exposure_start = row["exposure_start"]
    exposure_end_observed = row["exposure_end_observed"]
    strict_time = row["strict_proxy_culture_time"]
    broad_time = row["broad_proxy_culture_time"]

    if pd.isna(exposure_start) or pd.isna(exposure_end_observed):
        return rows

    if int(row["eligible_48h_observed_exposure"]) != 1:
        return rows

    max_at_risk_end = exposure_end_observed
    if pd.notna(strict_time):
        max_at_risk_end = min(max_at_risk_end, strict_time)

    landmark_hour = MIN_LANDMARK_HOURS
    while True:
        landmark_time = exposure_start + pd.Timedelta(hours=landmark_hour)

        # A prediction row must occur while the episode is still observed and before the strict event.
        if landmark_time >= max_at_risk_end:
            break

        prediction_window_end = landmark_time + pd.Timedelta(hours=PREDICTION_HORIZON_HOURS)
        observed_window_end = min(prediction_window_end, exposure_end_observed)

        future_strict = int(
            pd.notna(strict_time)
            and strict_time > landmark_time
            and strict_time <= prediction_window_end
            and strict_time <= exposure_end_observed
        )
        future_broad = int(
            pd.notna(broad_time)
            and broad_time > landmark_time
            and broad_time <= prediction_window_end
            and broad_time <= exposure_end_observed
        )

        hours_to_strict_event = np.nan
        if pd.notna(strict_time):
            hours_to_strict_event = (strict_time - landmark_time).total_seconds() / 3600

        hours_to_observed_end = (exposure_end_observed - landmark_time).total_seconds() / 3600
        full_7d_followup_observed = int(hours_to_observed_end >= PREDICTION_HORIZON_HOURS)
        status = classify_status(
            future_strict,
            prediction_window_end,
            exposure_end_observed,
            row["end_reason_observed"],
        )

        rows.append({
            "episode_id": row["episode_id"],
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "stay_id": row["stay_id"],
            "episode_number_within_stay": row["episode_number_within_stay"],
            "anchor_year_group": row.get("anchor_year_group", ""),
            "anchor_year": row.get("anchor_year", np.nan),
            "split_role": split_role(row.get("anchor_year_group", "")),
            "exposure_start": exposure_start,
            "exposure_end_observed": exposure_end_observed,
            "end_reason_observed": row["end_reason_observed"],
            "landmark_hour": landmark_hour,
            "landmark_day": int(landmark_hour / 24),
            "landmark_time": landmark_time,
            "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
            "prediction_window_end": prediction_window_end,
            "observed_window_end": observed_window_end,
            "hours_to_observed_end": hours_to_observed_end,
            "full_7d_followup_observed": full_7d_followup_observed,
            "future_strict_cvc_bsi_proxy_7d": future_strict,
            "future_broad_cvc_bsi_proxy_7d": future_broad,
            "landmark_outcome_status": status,
            "hours_to_strict_event": hours_to_strict_event,
            "strict_proxy_culture_time": strict_time,
            "broad_proxy_culture_time": broad_time,
            "observed_exposure_hours": row["observed_exposure_hours"],
            "raw_cvc_event_count": row["raw_cvc_event_count"],
            "cvc_itemids": row["cvc_itemids"],
            "cvc_types": row["cvc_types"],
            "locations": row["locations"],
            "first_careunit": row.get("first_careunit", ""),
            "last_careunit": row.get("last_careunit", ""),
            "admission_type": row.get("admission_type", ""),
            "insurance": row.get("insurance", ""),
            "race": row.get("race", ""),
            "gender": row.get("gender", ""),
            "anchor_age": row.get("anchor_age", np.nan),
            "early_positive_culture": row.get("early_positive_culture", 0),
            "strict_proxy_positive_orgs": row.get("strict_proxy_positive_orgs", ""),
            "strict_proxy_label_reason": row.get("strict_proxy_label_reason", ""),
        })

        landmark_hour += LANDMARK_STEP_HOURS

    return rows


# %% Load v0.5 exposure periods

print("Loading v0.5 catheter exposure periods...")
episodes = pd.read_csv(EXPOSURE_FILE)
episodes = to_datetime_columns(
    episodes,
    [
        "exposure_start",
        "exposure_end",
        "exposure_end_observed",
        "strict_proxy_culture_time",
        "broad_proxy_culture_time",
        "intime",
        "outtime",
        "admittime",
        "dischtime",
        "deathtime",
    ],
)

episodes["eligible_48h_observed_exposure"] = episodes["eligible_48h_observed_exposure"].astype(int)
eligible = episodes[episodes["eligible_48h_observed_exposure"].eq(1)].copy()

print(f"  Total exposure periods: {len(episodes):,}")
print(f"  Eligible exposure periods: {len(eligible):,}")
print(f"  Eligible strict proxy positives: {int(eligible['cvc_bsi_strict_proxy'].sum()):,}")


# %% Build daily landmark rows

print("")
print("Building daily landmark rows...")
all_rows = []
for _, row in eligible.iterrows():
    all_rows.extend(make_episode_landmarks(row))

landmarks = pd.DataFrame(all_rows)
if len(landmarks) == 0:
    raise RuntimeError("No landmark rows were generated. Check eligibility and exposure timing.")

landmarks = landmarks.sort_values(["episode_id", "landmark_hour"]).reset_index(drop=True)
landmarks["landmark_id"] = (
    landmarks["episode_id"].astype(str)
    + "_lm"
    + landmarks["landmark_hour"].astype(int).astype(str).str.zfill(4)
)

landmarks = landmarks[
    ["landmark_id"] + [c for c in landmarks.columns if c != "landmark_id"]
]

print(f"  Landmark rows: {len(landmarks):,}")
print(f"  Episodes represented: {landmarks['episode_id'].nunique():,}")
print(f"  Strict future-positive rows: {int(landmarks['future_strict_cvc_bsi_proxy_7d'].sum()):,}")
print(f"  Strict future-positive row rate: {landmarks['future_strict_cvc_bsi_proxy_7d'].mean() * 100:.2f}%")
print(f"  Episodes with future-positive rows: {landmarks.loc[landmarks['future_strict_cvc_bsi_proxy_7d'].eq(1), 'episode_id'].nunique():,}")


# %% Audits

print("")
print("Building landmark audits...")

summary_audit = pd.DataFrame([{
    "eligible_exposure_periods": int(len(eligible)),
    "eligible_exposure_stays": int(eligible["stay_id"].nunique()),
    "eligible_patients": int(eligible["subject_id"].nunique()),
    "episodes_with_landmarks": int(landmarks["episode_id"].nunique()),
    "stays_with_landmarks": int(landmarks["stay_id"].nunique()),
    "patients_with_landmarks": int(landmarks["subject_id"].nunique()),
    "landmark_rows": int(len(landmarks)),
    "median_landmarks_per_episode": float(landmarks.groupby("episode_id").size().median()),
    "max_landmarks_per_episode": int(landmarks.groupby("episode_id").size().max()),
    "strict_future_positive_rows": int(landmarks["future_strict_cvc_bsi_proxy_7d"].sum()),
    "strict_future_positive_row_rate": float(landmarks["future_strict_cvc_bsi_proxy_7d"].mean()),
    "broad_future_positive_rows": int(landmarks["future_broad_cvc_bsi_proxy_7d"].sum()),
    "broad_future_positive_row_rate": float(landmarks["future_broad_cvc_bsi_proxy_7d"].mean()),
    "episodes_with_strict_future_positive_rows": int(
        landmarks.loc[landmarks["future_strict_cvc_bsi_proxy_7d"].eq(1), "episode_id"].nunique()
    ),
    "episodes_with_broad_future_positive_rows": int(
        landmarks.loc[landmarks["future_broad_cvc_bsi_proxy_7d"].eq(1), "episode_id"].nunique()
    ),
    "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
    "first_landmark_hour": MIN_LANDMARK_HOURS,
    "landmark_step_hours": LANDMARK_STEP_HOURS,
    "temporal_lockbox_anchor_year_group": LOCKBOX_ANCHOR_YEAR_GROUP,
    "primary_target": "future_strict_cvc_bsi_proxy_7d",
}])

by_day_audit = (
    landmarks
    .groupby(["landmark_day", "landmark_hour"], as_index=False)
    .agg(
        rows=("landmark_id", "count"),
        episodes=("episode_id", "nunique"),
        strict_future_positive_rows=("future_strict_cvc_bsi_proxy_7d", "sum"),
        broad_future_positive_rows=("future_broad_cvc_bsi_proxy_7d", "sum"),
        full_7d_followup_observed=("full_7d_followup_observed", "sum"),
    )
)
by_day_audit["strict_future_positive_rate"] = (
    by_day_audit["strict_future_positive_rows"] / by_day_audit["rows"]
)
by_day_audit["broad_future_positive_rate"] = (
    by_day_audit["broad_future_positive_rows"] / by_day_audit["rows"]
)
by_day_audit["full_7d_followup_rate"] = (
    by_day_audit["full_7d_followup_observed"] / by_day_audit["rows"]
)

status_audit = (
    landmarks
    .groupby("landmark_outcome_status", as_index=False)
    .agg(
        rows=("landmark_id", "count"),
        episodes=("episode_id", "nunique"),
        strict_future_positive_rows=("future_strict_cvc_bsi_proxy_7d", "sum"),
        broad_future_positive_rows=("future_broad_cvc_bsi_proxy_7d", "sum"),
    )
    .sort_values("rows", ascending=False)
)
status_audit["row_share"] = status_audit["rows"] / len(landmarks)

split_audit = (
    landmarks
    .groupby(["split_role", "anchor_year_group"], as_index=False)
    .agg(
        rows=("landmark_id", "count"),
        episodes=("episode_id", "nunique"),
        stays=("stay_id", "nunique"),
        patients=("subject_id", "nunique"),
        strict_future_positive_rows=("future_strict_cvc_bsi_proxy_7d", "sum"),
        broad_future_positive_rows=("future_broad_cvc_bsi_proxy_7d", "sum"),
    )
)
split_audit["strict_future_positive_rate"] = (
    split_audit["strict_future_positive_rows"] / split_audit["rows"]
)
split_audit["broad_future_positive_rate"] = (
    split_audit["broad_future_positive_rows"] / split_audit["rows"]
)

episode_landmark_counts = (
    landmarks
    .groupby(["episode_id", "split_role"], as_index=False)
    .agg(
        landmark_rows=("landmark_id", "count"),
        any_strict_future_positive=("future_strict_cvc_bsi_proxy_7d", "max"),
        first_landmark_hour=("landmark_hour", "min"),
        last_landmark_hour=("landmark_hour", "max"),
    )
)

episode_coverage_audit = pd.DataFrame([{
    "eligible_episodes_without_landmarks": int(len(set(eligible["episode_id"]) - set(landmarks["episode_id"]))),
    "landmark_episodes_not_in_eligible": int(len(set(landmarks["episode_id"]) - set(eligible["episode_id"]))),
    "duplicate_landmark_ids": int(landmarks["landmark_id"].duplicated().sum()),
    "duplicate_episode_landmark_hour_rows": int(landmarks.duplicated(["episode_id", "landmark_hour"]).sum()),
    "rows_after_strict_event": int(
        (
            landmarks["strict_proxy_culture_time"].notna()
            & (landmarks["landmark_time"] >= landmarks["strict_proxy_culture_time"])
        ).sum()
    ),
    "rows_after_observed_exposure_end": int(
        (landmarks["landmark_time"] >= landmarks["exposure_end_observed"]).sum()
    ),
    "negative_hours_to_observed_end_rows": int(
        (landmarks["hours_to_observed_end"] < 0).sum()
    ),
}])


# %% Save outputs

landmark_file = DATA_PATH / "v0_5_daily_landmarks.csv"
summary_file = DATA_PATH / "v0_5_daily_landmark_audit.csv"
by_day_file = DATA_PATH / "v0_5_daily_landmark_by_day_audit.csv"
status_file = DATA_PATH / "v0_5_daily_landmark_outcome_status_audit.csv"
split_file = DATA_PATH / "v0_5_daily_landmark_temporal_split_audit.csv"
episode_counts_file = DATA_PATH / "v0_5_daily_landmark_episode_counts.csv"
coverage_file = DATA_PATH / "v0_5_daily_landmark_coverage_audit.csv"

landmarks.to_csv(landmark_file, index=False)
summary_audit.to_csv(summary_file, index=False)
by_day_audit.to_csv(by_day_file, index=False)
status_audit.to_csv(status_file, index=False)
split_audit.to_csv(split_file, index=False)
episode_landmark_counts.to_csv(episode_counts_file, index=False)
episode_coverage_audit.to_csv(coverage_file, index=False)

for source_file in [
    summary_file,
    by_day_file,
    status_file,
    split_file,
    episode_counts_file,
    coverage_file,
]:
    pd.read_csv(source_file).to_csv(OUTPUT_PATH / source_file.name, index=False)

manifest = pd.DataFrame([
    {"artifact": "daily_landmarks", "path": str(landmark_file)},
    {"artifact": "daily_landmark_audit", "path": str(summary_file)},
    {"artifact": "daily_landmark_by_day_audit", "path": str(by_day_file)},
    {"artifact": "daily_landmark_outcome_status_audit", "path": str(status_file)},
    {"artifact": "daily_landmark_temporal_split_audit", "path": str(split_file)},
    {"artifact": "daily_landmark_episode_counts", "path": str(episode_counts_file)},
    {"artifact": "daily_landmark_coverage_audit", "path": str(coverage_file)},
])
manifest_file = OUTPUT_PATH / "v0_5_daily_landmark_frame_manifest.csv"
manifest.to_csv(manifest_file, index=False)


# %% Console summary

print("")
print("v0.5 daily landmark frame summary:")
print(summary_audit.T.to_string(header=False))
print("")
print("Coverage audit:")
print(episode_coverage_audit.to_string(index=False))
print("")
print("Temporal split audit:")
print(split_audit.to_string(index=False))
print("")
print("Outcome status audit:")
print(status_audit.to_string(index=False))
print("")
print(f"Saved daily landmarks to: {landmark_file}")
print(f"Saved Run 17 audit copies to: {OUTPUT_PATH}")
print("")
print("Feature Engineering 02 v0.5 Daily Landmarks complete.")

