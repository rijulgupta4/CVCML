# %% Imports and paths

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


MIMIC_PATH = Path(r"C:\path\to\mimic-iv")
PROJECT_PATH = Path(r"C:\path\to\CVCML")
HOSP = MIMIC_PATH / "hosp"
ICU = MIMIC_PATH / "icu"
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 16 (v0.5 Catheter Episode Redesign)"

DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MIN_ELIGIBLE_EXPOSURE_HOURS = 48
CONTINUOUS_EXPOSURE_GAP_HOURS = 4


# %% Constants and helper functions

CVC_PROCEDURE_IDS = [
    224264,  # PICC Line
    224270,  # Dialysis Catheter
    224273,  # Presep Catheter
    224560,  # PA Catheter
    225203,  # Pheresis Catheter
    229517,  # Multi Lumen Cooling Catheter
]

BLOOD_CULTURE_SPEC_TYPES = [
    "BLOOD CULTURE",
    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
]

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


def join_unique(values):
    cleaned = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return "; ".join(cleaned)


def min_ignore_na(values):
    valid = [v for v in values if pd.notna(v)]
    if not valid:
        return pd.NaT
    return min(valid)


def summarize_strict_episode_label(group):
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
        positive_orgs = join_unique(strict_rows["org_name"])
    else:
        culture_time = pd.NaT
        positive_orgs = ""

    return pd.Series({
        "cvc_bsi_strict_proxy": strict_positive,
        "strict_proxy_culture_time": culture_time,
        "strict_proxy_positive_orgs": positive_orgs,
        "strict_proxy_label_reason": strict_reason,
        "strict_proxy_qualifying_culture_rows": int(len(strict_rows)),
        "strict_proxy_clear_pathogen_rows": int(len(clear_pathogens)),
        "strict_proxy_commensal_rows": int(len(common_commensals)),
        "strict_proxy_distinct_commensal_times": int(common_commensals["charttime"].dropna().nunique()),
    })


def merge_exposure_intervals(group, subject_id, hadm_id, stay_id):
    group = group.sort_values(["starttime", "endtime"]).copy()
    max_gap = pd.Timedelta(hours=CONTINUOUS_EXPOSURE_GAP_HOURS)
    periods = []
    current = None

    for _, row in group.iterrows():
        if current is None:
            current = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "exposure_start": row["starttime"],
                "exposure_end": row["endtime"],
                "raw_cvc_event_count": 1,
                "cvc_itemids": [row["itemid"]],
                "cvc_types": [row["cvc_type"]],
                "locations": [row.get("location", np.nan)],
            }
            continue

        starts_before_gap = row["starttime"] <= current["exposure_end"] + max_gap
        if starts_before_gap:
            current["exposure_end"] = max(current["exposure_end"], row["endtime"])
            current["raw_cvc_event_count"] += 1
            current["cvc_itemids"].append(row["itemid"])
            current["cvc_types"].append(row["cvc_type"])
            current["locations"].append(row.get("location", np.nan))
        else:
            periods.append(current)
            current = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "exposure_start": row["starttime"],
                "exposure_end": row["endtime"],
                "raw_cvc_event_count": 1,
                "cvc_itemids": [row["itemid"]],
                "cvc_types": [row["cvc_type"]],
                "locations": [row.get("location", np.nan)],
            }

    if current is not None:
        periods.append(current)

    out = pd.DataFrame(periods)
    if len(out) == 0:
        return out

    out["cvc_itemids"] = out["cvc_itemids"].apply(lambda x: "; ".join(str(int(v)) for v in sorted(set(x))))
    out["cvc_types"] = out["cvc_types"].apply(join_unique)
    out["locations"] = out["locations"].apply(join_unique)
    return out


# %% Load lightweight tables

print("Loading patients, admissions, icustays, and d_items...")
patients = pd.read_csv(HOSP / "patients.csv.gz")
admissions = pd.read_csv(
    HOSP / "admissions.csv.gz",
    parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
)
icustays = pd.read_csv(
    ICU / "icustays.csv.gz",
    parse_dates=["intime", "outtime"],
)
d_items = pd.read_csv(ICU / "d_items.csv.gz")

print(f"  patients:   {len(patients):,}")
print(f"  admissions: {len(admissions):,}")
print(f"  icustays:   {len(icustays):,}")
print(f"  d_items:    {len(d_items):,}")


# %% Extract all CVC procedure events

print("")
print("Loading procedureevents and extracting CVC records...")
procedureevents = pd.read_csv(ICU / "procedureevents.csv.gz", low_memory=False)
print(f"  procedureevents rows: {len(procedureevents):,}")

cvc_events = procedureevents[procedureevents["itemid"].isin(CVC_PROCEDURE_IDS)].copy()
cvc_events["starttime"] = pd.to_datetime(cvc_events["starttime"], errors="coerce")
cvc_events["endtime"] = pd.to_datetime(cvc_events["endtime"], errors="coerce")
cvc_events = cvc_events[cvc_events["starttime"].notna() & cvc_events["endtime"].notna()].copy()
cvc_events = cvc_events[cvc_events["endtime"] > cvc_events["starttime"]].copy()

cvc_events = cvc_events.merge(
    d_items[["itemid", "label"]],
    on="itemid",
    how="left",
).rename(columns={"label": "cvc_type"})

cvc_events["raw_event_duration_hours"] = (
    (cvc_events["endtime"] - cvc_events["starttime"]).dt.total_seconds() / 3600
)
cvc_events["raw_event_ge_48h"] = (
    cvc_events["raw_event_duration_hours"] >= MIN_ELIGIBLE_EXPOSURE_HOURS
).astype(int)

keep_event_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "caregiver_id",
    "itemid",
    "cvc_type",
    "starttime",
    "endtime",
    "raw_event_duration_hours",
    "raw_event_ge_48h",
    "location",
    "locationcategory",
    "orderid",
    "linkorderid",
    "statusdescription",
]
keep_event_cols = [c for c in keep_event_cols if c in cvc_events.columns]
cvc_events = cvc_events[keep_event_cols].sort_values(["stay_id", "starttime", "endtime"]).reset_index(drop=True)
cvc_events["raw_cvc_event_id"] = np.arange(1, len(cvc_events) + 1)

print(f"  CVC procedure records retained: {len(cvc_events):,}")
print(f"  Unique ICU stays with any CVC record: {cvc_events['stay_id'].nunique():,}")
print(f"  Raw CVC records >=48h: {int(cvc_events['raw_event_ge_48h'].sum()):,}")


# %% Reconstruct continuous CVC exposure periods

print("")
print("Reconstructing continuous CVC exposure periods...")
period_chunks = []
for (subject_id, hadm_id, stay_id), group in cvc_events.groupby(["subject_id", "hadm_id", "stay_id"]):
    period_chunks.append(merge_exposure_intervals(group, subject_id, hadm_id, stay_id))

periods = (
    pd.concat(period_chunks, ignore_index=True)
    if period_chunks
    else pd.DataFrame(columns=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "exposure_start",
        "exposure_end",
        "raw_cvc_event_count",
        "cvc_itemids",
        "cvc_types",
        "locations",
    ])
)

periods = periods.merge(
    icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit", "last_careunit"]],
    on=["subject_id", "hadm_id", "stay_id"],
    how="left",
)

admission_cols = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "insurance",
    "race",
    "hospital_expire_flag",
]
periods = periods.merge(admissions[admission_cols], on=["subject_id", "hadm_id"], how="left")

patient_cols = ["subject_id", "gender", "anchor_age"]
if "anchor_year_group" in patients.columns:
    patient_cols.append("anchor_year_group")
if "anchor_year" in patients.columns:
    patient_cols.append("anchor_year")
periods = periods.merge(patients[patient_cols], on="subject_id", how="left")

clip_candidates = ["exposure_end", "outtime", "dischtime", "deathtime"]
periods["exposure_end_observed"] = periods[clip_candidates].apply(min_ignore_na, axis=1)
periods["exposure_hours"] = (
    (periods["exposure_end"] - periods["exposure_start"]).dt.total_seconds() / 3600
)
periods["observed_exposure_hours"] = (
    (periods["exposure_end_observed"] - periods["exposure_start"]).dt.total_seconds() / 3600
)
periods["observed_exposure_hours"] = periods["observed_exposure_hours"].clip(lower=0)
periods["eligible_48h_observed_exposure"] = (
    periods["observed_exposure_hours"] >= MIN_ELIGIBLE_EXPOSURE_HOURS
).astype(int)

periods["episode_number_within_stay"] = periods.groupby("stay_id").cumcount() + 1
periods["episode_id"] = (
    periods["stay_id"].astype(str)
    + "_ep"
    + periods["episode_number_within_stay"].astype(str).str.zfill(2)
)

periods["end_reason_observed"] = "procedure_end_or_line_removal"
periods.loc[
    periods["deathtime"].notna()
    & (periods["exposure_end_observed"] == periods["deathtime"]),
    "end_reason_observed",
] = "death"
periods.loc[
    periods["dischtime"].notna()
    & (periods["exposure_end_observed"] == periods["dischtime"]),
    "end_reason_observed",
] = "hospital_discharge"
periods.loc[
    periods["outtime"].notna()
    & (periods["exposure_end_observed"] == periods["outtime"]),
    "end_reason_observed",
] = "icu_outtime"

periods = periods.sort_values(["subject_id", "hadm_id", "stay_id", "exposure_start"]).reset_index(drop=True)

print(f"  Continuous exposure periods: {len(periods):,}")
print(f"  Eligible periods >=48 observed hours: {int(periods['eligible_48h_observed_exposure'].sum()):,}")
print(f"  Stays with multiple exposure periods: {int((periods.groupby('stay_id').size() > 1).sum()):,}")


# %% Load positive blood cultures

print("")
print("Loading microbiologyevents and extracting positive blood cultures...")
micro = pd.read_csv(HOSP / "microbiologyevents.csv.gz", low_memory=False)
print(f"  microbiologyevents rows: {len(micro):,}")

blood_cultures = micro[micro["spec_type_desc"].isin(BLOOD_CULTURE_SPEC_TYPES)].copy()
positive_cultures = blood_cultures[
    blood_cultures["org_name"].notna()
    & ~blood_cultures["org_name"].str.contains("CANCELLED", case=False, na=False)
].copy()
positive_cultures["charttime"] = pd.to_datetime(positive_cultures["charttime"], errors="coerce")
positive_cultures = positive_cultures[positive_cultures["charttime"].notna()].copy()
positive_cultures["is_common_commensal"] = positive_cultures["org_name"].apply(is_common_commensal)

culture_cols = [
    "subject_id",
    "hadm_id",
    "charttime",
    "spec_type_desc",
    "test_name",
    "org_name",
    "is_common_commensal",
]
culture_cols = [c for c in culture_cols if c in positive_cultures.columns]
positive_cultures = positive_cultures[culture_cols].copy()

print(f"  Positive blood culture rows retained: {len(positive_cultures):,}")


# %% Associate cultures with exposure periods

print("")
print("Associating cultures with CVC exposure periods...")
eligible_periods = periods[periods["eligible_48h_observed_exposure"].eq(1)].copy()
eligible_periods["earliest_eligible_culture_time"] = (
    eligible_periods["exposure_start"] + pd.Timedelta(hours=MIN_ELIGIBLE_EXPOSURE_HOURS)
)

culture_merged = eligible_periods[
    [
        "episode_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "exposure_start",
        "exposure_end_observed",
        "earliest_eligible_culture_time",
    ]
].merge(
    positive_cultures,
    on=["subject_id", "hadm_id"],
    how="left",
)

culture_merged["culture_while_observed_exposure"] = (
    (culture_merged["charttime"] >= culture_merged["exposure_start"])
    & (culture_merged["charttime"] <= culture_merged["exposure_end_observed"])
)
culture_merged["early_positive_culture"] = (
    culture_merged["culture_while_observed_exposure"]
    & (culture_merged["charttime"] < culture_merged["earliest_eligible_culture_time"])
)
culture_merged["qualifying_cvc_associated_culture"] = (
    (culture_merged["charttime"] >= culture_merged["earliest_eligible_culture_time"])
    & (culture_merged["charttime"] <= culture_merged["exposure_end_observed"])
)
culture_merged["hours_from_exposure_start"] = (
    (culture_merged["charttime"] - culture_merged["exposure_start"]).dt.total_seconds() / 3600
)

qualifying_cultures = culture_merged[culture_merged["qualifying_cvc_associated_culture"]].copy()
early_culture_flags = (
    culture_merged
    .groupby("episode_id")["early_positive_culture"]
    .max()
    .fillna(False)
    .astype(int)
    .reset_index()
)

broader_labels = (
    culture_merged
    .groupby("episode_id")["qualifying_cvc_associated_culture"]
    .max()
    .fillna(False)
    .astype(int)
    .reset_index()
    .rename(columns={"qualifying_cvc_associated_culture": "cvc_bsi_broad_proxy"})
)

broader_culture_times = (
    qualifying_cultures
    .sort_values("charttime")
    .groupby("episode_id")["charttime"]
    .first()
    .reset_index()
    .rename(columns={"charttime": "broad_proxy_culture_time"})
)

if len(qualifying_cultures) > 0:
    strict_labels = (
        qualifying_cultures
        .groupby("episode_id")
        .apply(summarize_strict_episode_label)
        .reset_index()
    )
else:
    strict_labels = pd.DataFrame(columns=[
        "episode_id",
        "cvc_bsi_strict_proxy",
        "strict_proxy_culture_time",
        "strict_proxy_positive_orgs",
        "strict_proxy_label_reason",
        "strict_proxy_qualifying_culture_rows",
        "strict_proxy_clear_pathogen_rows",
        "strict_proxy_commensal_rows",
        "strict_proxy_distinct_commensal_times",
    ])

periods = periods.merge(broader_labels, on="episode_id", how="left")
periods = periods.merge(broader_culture_times, on="episode_id", how="left")
periods = periods.merge(strict_labels, on="episode_id", how="left")
periods = periods.merge(early_culture_flags, on="episode_id", how="left")

periods["cvc_bsi_broad_proxy"] = periods["cvc_bsi_broad_proxy"].fillna(0).astype(int)
periods["cvc_bsi_strict_proxy"] = periods["cvc_bsi_strict_proxy"].fillna(0).astype(int)
periods["early_positive_culture"] = periods["early_positive_culture"].fillna(0).astype(int)
periods["strict_proxy_positive_orgs"] = periods["strict_proxy_positive_orgs"].fillna("")
periods["strict_proxy_label_reason"] = periods["strict_proxy_label_reason"].fillna("no_qualifying_positive_culture")
periods["broad_proxy_culture_time"] = pd.to_datetime(periods["broad_proxy_culture_time"], errors="coerce")
periods["strict_proxy_culture_time"] = pd.to_datetime(periods["strict_proxy_culture_time"], errors="coerce")

for col in [
    "strict_proxy_qualifying_culture_rows",
    "strict_proxy_clear_pathogen_rows",
    "strict_proxy_commensal_rows",
    "strict_proxy_distinct_commensal_times",
]:
    periods[col] = periods[col].fillna(0).astype(int)

periods["broad_downgraded_by_strict_organism_rule"] = (
    periods["cvc_bsi_broad_proxy"].eq(1)
    & periods["cvc_bsi_strict_proxy"].eq(0)
).astype(int)


# %% Audits

print("")
print("Building v0.5 audit tables...")
eligible = periods[periods["eligible_48h_observed_exposure"].eq(1)].copy()

summary_audit = pd.DataFrame([{
    "raw_cvc_procedure_events": int(len(cvc_events)),
    "raw_cvc_events_ge_48h": int(cvc_events["raw_event_ge_48h"].sum()),
    "stays_with_any_cvc_record": int(cvc_events["stay_id"].nunique()),
    "continuous_exposure_periods": int(len(periods)),
    "eligible_48h_exposure_periods": int(len(eligible)),
    "eligible_48h_exposure_stays": int(eligible["stay_id"].nunique()),
    "stays_with_multiple_exposure_periods": int((periods.groupby("stay_id").size() > 1).sum()),
    "broad_proxy_positive_episodes": int(eligible["cvc_bsi_broad_proxy"].sum()),
    "broad_proxy_positive_rate": float(eligible["cvc_bsi_broad_proxy"].mean()) if len(eligible) else np.nan,
    "strict_proxy_positive_episodes": int(eligible["cvc_bsi_strict_proxy"].sum()),
    "strict_proxy_positive_rate": float(eligible["cvc_bsi_strict_proxy"].mean()) if len(eligible) else np.nan,
    "broad_downgraded_by_strict_organism_rule": int(eligible["broad_downgraded_by_strict_organism_rule"].sum()),
    "early_positive_culture_episodes": int(eligible["early_positive_culture"].sum()),
    "continuous_exposure_gap_hours": CONTINUOUS_EXPOSURE_GAP_HOURS,
    "minimum_eligible_exposure_hours": MIN_ELIGIBLE_EXPOSURE_HOURS,
    "label_name_recommended": "strict CVC-associated BSI proxy",
    "key_limitation": "Secondary-source infection and MBI-LCBI exclusions are not adjudicated in this proxy.",
}])

censoring_audit = (
    eligible
    .groupby("end_reason_observed")
    .agg(
        episodes=("episode_id", "count"),
        strict_proxy_positive=("cvc_bsi_strict_proxy", "sum"),
        broad_proxy_positive=("cvc_bsi_broad_proxy", "sum"),
        median_observed_exposure_hours=("observed_exposure_hours", "median"),
    )
    .reset_index()
)

duration_audit = pd.DataFrame([{
    "period_set": "all_continuous_exposure_periods",
    "episodes": int(len(periods)),
    "median_observed_exposure_hours": float(periods["observed_exposure_hours"].median()),
    "p25_observed_exposure_hours": float(periods["observed_exposure_hours"].quantile(0.25)),
    "p75_observed_exposure_hours": float(periods["observed_exposure_hours"].quantile(0.75)),
    "max_observed_exposure_hours": float(periods["observed_exposure_hours"].max()),
}, {
    "period_set": "eligible_48h_observed_exposure_periods",
    "episodes": int(len(eligible)),
    "median_observed_exposure_hours": float(eligible["observed_exposure_hours"].median()) if len(eligible) else np.nan,
    "p25_observed_exposure_hours": float(eligible["observed_exposure_hours"].quantile(0.25)) if len(eligible) else np.nan,
    "p75_observed_exposure_hours": float(eligible["observed_exposure_hours"].quantile(0.75)) if len(eligible) else np.nan,
    "max_observed_exposure_hours": float(eligible["observed_exposure_hours"].max()) if len(eligible) else np.nan,
}])

if "anchor_year_group" in eligible.columns:
    temporal_audit = (
        eligible
        .groupby("anchor_year_group")
        .agg(
            eligible_episodes=("episode_id", "count"),
            eligible_stays=("stay_id", "nunique"),
            patients=("subject_id", "nunique"),
            strict_proxy_positive=("cvc_bsi_strict_proxy", "sum"),
            broad_proxy_positive=("cvc_bsi_broad_proxy", "sum"),
        )
        .reset_index()
    )
    temporal_audit["strict_proxy_rate"] = (
        temporal_audit["strict_proxy_positive"] / temporal_audit["eligible_episodes"]
    )
    temporal_audit["broad_proxy_rate"] = (
        temporal_audit["broad_proxy_positive"] / temporal_audit["eligible_episodes"]
    )
    latest_group = sorted(temporal_audit["anchor_year_group"].dropna().unique())[-1] if len(temporal_audit) else None
    temporal_audit["candidate_split_role"] = np.where(
        temporal_audit["anchor_year_group"].eq(latest_group),
        "candidate_temporal_lockbox",
        "candidate_development",
    )
else:
    temporal_audit = pd.DataFrame([{
        "note": "patients.csv.gz did not contain anchor_year_group; temporal lockbox audit not available.",
    }])

if len(qualifying_cultures):
    organism_counts = (
        qualifying_cultures
        .assign(organism_type=lambda x: x["is_common_commensal"].map({True: "common_commensal", False: "clear_pathogen"}))
        .groupby(["org_name", "organism_type"])
        .agg(
            culture_rows=("org_name", "size"),
            episodes=("episode_id", "nunique"),
            stays=("stay_id", "nunique"),
        )
        .reset_index()
        .sort_values(["organism_type", "episodes", "culture_rows"], ascending=[True, False, False])
    )
else:
    organism_counts = pd.DataFrame(columns=["org_name", "organism_type", "culture_rows", "episodes", "stays"])

label_reason_counts = (
    eligible
    .groupby("strict_proxy_label_reason")
    .agg(
        episodes=("episode_id", "count"),
        strict_proxy_positive=("cvc_bsi_strict_proxy", "sum"),
        broad_proxy_positive=("cvc_bsi_broad_proxy", "sum"),
    )
    .reset_index()
    .sort_values("episodes", ascending=False)
)

episode_per_stay_audit = (
    periods
    .groupby("stay_id")
    .agg(
        subject_id=("subject_id", "first"),
        hadm_id=("hadm_id", "first"),
        exposure_periods=("episode_id", "count"),
        eligible_48h_periods=("eligible_48h_observed_exposure", "sum"),
        strict_proxy_positive_periods=("cvc_bsi_strict_proxy", "sum"),
        broad_proxy_positive_periods=("cvc_bsi_broad_proxy", "sum"),
    )
    .reset_index()
)


# %% Save data and audit outputs

raw_event_file = DATA_PATH / "v0_5_cvc_procedure_events.csv"
period_file = DATA_PATH / "v0_5_catheter_exposure_periods.csv"
culture_detail_file = DATA_PATH / "v0_5_episode_culture_detail.csv"
summary_file = DATA_PATH / "v0_5_episode_label_audit.csv"
organism_file = DATA_PATH / "v0_5_qualifying_organism_counts.csv"
temporal_file = DATA_PATH / "v0_5_temporal_lockbox_candidate_audit.csv"
censoring_file = DATA_PATH / "v0_5_censoring_audit.csv"
duration_file = DATA_PATH / "v0_5_duration_audit.csv"
label_reason_file = DATA_PATH / "v0_5_label_reason_counts.csv"
episode_per_stay_file = DATA_PATH / "v0_5_episode_per_stay_audit.csv"

cvc_events.to_csv(raw_event_file, index=False)
periods.to_csv(period_file, index=False)
culture_merged.to_csv(culture_detail_file, index=False)
summary_audit.to_csv(summary_file, index=False)
organism_counts.to_csv(organism_file, index=False)
temporal_audit.to_csv(temporal_file, index=False)
censoring_audit.to_csv(censoring_file, index=False)
duration_audit.to_csv(duration_file, index=False)
label_reason_counts.to_csv(label_reason_file, index=False)
episode_per_stay_audit.to_csv(episode_per_stay_file, index=False)

for source_file in [
    summary_file,
    temporal_file,
    censoring_file,
    duration_file,
    label_reason_file,
    episode_per_stay_file,
]:
    out_file = OUTPUT_PATH / source_file.name
    pd.read_csv(source_file).to_csv(out_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "raw_cvc_procedure_events", "path": str(raw_event_file)},
    {"artifact": "catheter_exposure_periods", "path": str(period_file)},
    {"artifact": "episode_culture_detail", "path": str(culture_detail_file)},
    {"artifact": "episode_label_audit", "path": str(summary_file)},
    {"artifact": "qualifying_organism_counts", "path": str(organism_file)},
    {"artifact": "temporal_lockbox_candidate_audit", "path": str(temporal_file)},
    {"artifact": "censoring_audit", "path": str(censoring_file)},
    {"artifact": "duration_audit", "path": str(duration_file)},
    {"artifact": "label_reason_counts", "path": str(label_reason_file)},
    {"artifact": "episode_per_stay_audit", "path": str(episode_per_stay_file)},
])
manifest_file = OUTPUT_PATH / "v0_5_catheter_episode_redesign_manifest.csv"
manifest.to_csv(manifest_file, index=False)


# %% Console summary

print("")
print("v0.5 catheter episode redesign summary:")
print(summary_audit.T.to_string(header=False))
print("")
print("Censoring/end-reason audit:")
print(censoring_audit.to_string(index=False))
print("")
print("Temporal lockbox candidate audit:")
print(temporal_audit.to_string(index=False))
print("")
print("Saved v0.5 data files to:")
print(f"  {DATA_PATH}")
print("Saved Run 16 audit copies to:")
print(f"  {OUTPUT_PATH}")
print("")
print("Data Extraction 01 v0.5 Catheter Episodes complete.")

