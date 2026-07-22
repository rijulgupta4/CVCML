# %% Imports and paths

import re
from pathlib import Path

import numpy as np
import pandas as pd


MIMIC_PATH = Path(r"C:\path\to\mimic-iv")
PROJECT_PATH = Path(r"C:\path\to\CVCML")
HOSP = MIMIC_PATH / "hosp"
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 22 (v0.5 Secondary Source Label Audit)"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

EPISODE_FILE = DATA_PATH / "v0_5_catheter_exposure_periods.csv"
LANDMARK_FILE = DATA_PATH / "v0_5_daily_landmarks.csv"

SOURCE_WINDOW_DAYS = 3
SOURCE_WINDOW_HOURS = SOURCE_WINDOW_DAYS * 24


# %% Source definitions

BLOOD_SPEC_RE = re.compile(r"\bBLOOD\b|BLOOD CULTURE|SEROLOGY/BLOOD|Blood \(Toxo\)", flags=re.IGNORECASE)

SOURCE_PATTERNS = {
    "urinary": re.compile(r"URINE|UROGENITAL|RENAL|KIDNEY|BLADDER", flags=re.IGNORECASE),
    "respiratory": re.compile(
        r"SPUTUM|RESPIRATORY|TRACHEAL|BRONCH|BAL|BRONCHOALVEOLAR|PLEURAL|LUNG|NASAL|THROAT",
        flags=re.IGNORECASE,
    ),
    "wound_skin_soft_tissue": re.compile(
        r"WOUND|ABSCESS|SKIN|SOFT TISSUE|TISSUE|ULCER|DRAINAGE|DRAIN|PUS|ASPIRATE",
        flags=re.IGNORECASE,
    ),
    "abdominal_gi_biliary": re.compile(
        r"ABDOM|PERITONEAL|ASCITES|BILE|BILIARY|GALLBLADDER|LIVER|PANCREA|STOOL|FECAL|RECTAL|GI",
        flags=re.IGNORECASE,
    ),
    "csf_cns": re.compile(r"CSF|CEREBROSPINAL|BRAIN|CNS", flags=re.IGNORECASE),
    "other_sterile_site": re.compile(r"FLUID|JOINT|SYNOVIAL|BONE|STERILE BODY", flags=re.IGNORECASE),
    "line_support": re.compile(r"CATHETER TIP|CENTRAL LINE|LINE TIP|IV CATHETER", flags=re.IGNORECASE),
}

ICD_SOURCE_PATTERNS = {
    "urinary": re.compile(r"urinary tract infection|pyelonephritis|cystitis|urosepsis|kidney infection", flags=re.IGNORECASE),
    "respiratory": re.compile(r"pneumonia|empyema|lung abscess|respiratory infection|bronchitis", flags=re.IGNORECASE),
    "wound_skin_soft_tissue": re.compile(
        r"cellulitis|abscess|wound infection|soft tissue infection|skin infection|necrotizing fasciitis",
        flags=re.IGNORECASE,
    ),
    "abdominal_gi_biliary": re.compile(
        r"peritonitis|cholangitis|cholecystitis|diverticulitis|appendicitis|intra-abdominal|abdominal abscess|biliary infection",
        flags=re.IGNORECASE,
    ),
    "surgical_procedure_related": re.compile(r"postoperative infection|surgical site|infected prosthesis|device infection", flags=re.IGNORECASE),
    "endocarditis_or_deep_focus": re.compile(r"endocarditis|osteomyelitis|septic arthritis|meningitis", flags=re.IGNORECASE),
}


# %% Helpers

def join_unique(values):
    cleaned = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    return "; ".join(cleaned)


def first_matching_source(text, pattern_dict):
    if pd.isna(text):
        return ""
    text = str(text)
    for source, pattern in pattern_dict.items():
        if pattern.search(text):
            return source
    return ""


def source_priority(source):
    priorities = {
        "line_support": 99,
        "urinary": 1,
        "respiratory": 2,
        "wound_skin_soft_tissue": 3,
        "abdominal_gi_biliary": 4,
        "csf_cns": 5,
        "other_sterile_site": 6,
        "surgical_procedure_related": 7,
        "endocarditis_or_deep_focus": 8,
    }
    return priorities.get(source, 50)


def normalize_organism_name(value):
    if pd.isna(value):
        return ""
    text = str(value).upper()
    replacements = {
        "STAPH AUREUS COAG +": "STAPHYLOCOCCUS AUREUS",
        "STAPH AUREUS COAG POS": "STAPHYLOCOCCUS AUREUS",
        "STAPHYLOCOCCUS, COAGULASE NEGATIVE": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
        "ESCHERICHIA COLI": "E COLI",
        "ENTEROBACTER CLOACAE COMPLEX": "ENTEROBACTER CLOACAE",
        "ACINETOBACTER BAUMANNII COMPLEX": "ACINETOBACTER BAUMANNII",
        "CANDIDA ALBICANS, PRESUMPTIVE IDENTIFICATION": "CANDIDA ALBICANS",
        "YEAST, PRESUMPTIVELY NOT C. ALBICANS": "YEAST",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def organism_tokens(value):
    normalized = normalize_organism_name(value)
    if not normalized:
        return set()
    parts = [p.strip() for p in normalized.split(";") if p.strip()]
    tokens = set()
    for part in parts:
        tokens.add(part)
        words = part.split()
        if len(words) >= 2:
            tokens.add(" ".join(words[:2]))
        if len(words) >= 1 and words[0] in {"CANDIDA", "ENTEROCOCCUS", "KLEBSIELLA", "PSEUDOMONAS", "SERRATIA", "STAPHYLOCOCCUS", "ENTEROBACTER"}:
            tokens.add(words[0])
    return {t for t in tokens if t and t not in {"GRAM", "ROD", "COCCUS", "YEAST"}}


def organisms_concordant(blood_orgs, source_org):
    blood_tokens = organism_tokens(blood_orgs)
    source_tokens = organism_tokens(source_org)
    if not blood_tokens or not source_tokens:
        return False
    return bool(blood_tokens & source_tokens)


def source_bucket_list(values):
    buckets = sorted({v for v in values if pd.notna(v) and str(v).strip()}, key=source_priority)
    return "; ".join(buckets)


def classify_source_screen(row):
    if int(row["cvc_bsi_strict_proxy"]) != 1:
        return "strict_negative"
    if int(row.get("concordant_nonblood_source_culture_count", 0)) > 0:
        return "strict_proxy_secondary_possible_concordant_culture"
    if int(row.get("nearby_nonblood_source_culture_count", 0)) > 0:
        return "strict_proxy_uncertain_nonconcordant_source_culture"
    if int(row.get("hadm_source_icd_count", 0)) > 0:
        return "strict_proxy_uncertain_icd_only"
    return "strict_proxy_primary_likely"


def derive_horizon_target(landmarks, culture_col, target_col, horizon_hours=168):
    event_time = pd.to_datetime(landmarks[culture_col], errors="coerce")
    landmark_time = pd.to_datetime(landmarks["landmark_time"], errors="coerce")
    exposure_end = pd.to_datetime(landmarks["exposure_end_observed"], errors="coerce")
    window_end = landmark_time + pd.to_timedelta(horizon_hours, unit="h")
    landmarks[target_col] = (
        event_time.notna()
        & (event_time > landmark_time)
        & (event_time <= window_end)
        & (event_time <= exposure_end)
    ).astype(int)
    return landmarks


# %% Load v0.5 episodes

print("Loading v0.5 catheter exposure episodes...")
episodes = pd.read_csv(
    EPISODE_FILE,
    parse_dates=[
        "exposure_start",
        "exposure_end_observed",
        "strict_proxy_culture_time",
        "broad_proxy_culture_time",
        "admittime",
        "dischtime",
    ],
)
episodes["cvc_bsi_strict_proxy"] = episodes["cvc_bsi_strict_proxy"].fillna(0).astype(int)
strict_positive = episodes[episodes["cvc_bsi_strict_proxy"].eq(1)].copy()

print(f"  Total episodes: {len(episodes):,}")
print(f"  Strict proxy positive episodes: {len(strict_positive):,}")


# %% Load positive microbiology cultures and screen non-blood sources

print("")
print("Loading positive microbiology cultures for secondary-source screen...")
micro_cols = [
    "microevent_id",
    "subject_id",
    "hadm_id",
    "charttime",
    "chartdate",
    "spec_type_desc",
    "test_name",
    "org_name",
]
micro = pd.read_csv(HOSP / "microbiologyevents.csv.gz", usecols=micro_cols, low_memory=False)
micro = micro[micro["hadm_id"].notna() & micro["org_name"].notna()].copy()
micro["hadm_id"] = micro["hadm_id"].astype(int)
micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
micro["chartdate"] = pd.to_datetime(micro["chartdate"], errors="coerce")
micro["culture_time"] = micro["charttime"].fillna(micro["chartdate"])
micro = micro[micro["culture_time"].notna()].copy()
micro = micro.drop_duplicates([
    "microevent_id",
    "subject_id",
    "hadm_id",
    "culture_time",
    "spec_type_desc",
    "test_name",
    "org_name",
]).copy()
micro["source_bucket"] = micro["spec_type_desc"].apply(lambda x: first_matching_source(x, SOURCE_PATTERNS))
micro["is_blood_specimen"] = micro["spec_type_desc"].fillna("").str.contains(BLOOD_SPEC_RE, regex=True)
micro["is_nonblood_source_candidate"] = (
    micro["source_bucket"].ne("")
    & ~micro["is_blood_specimen"]
    & micro["source_bucket"].ne("line_support")
)

strict_hadm_ids = set(strict_positive["hadm_id"].dropna().astype(int))
source_cultures = micro[
    micro["hadm_id"].isin(strict_hadm_ids)
    & micro["is_nonblood_source_candidate"]
].copy()

print(f"  Positive microbiology rows retained: {len(micro):,}")
print(f"  Non-blood source-candidate rows in strict-positive admissions: {len(source_cultures):,}")

source_matches = strict_positive[[
    "episode_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "strict_proxy_culture_time",
    "strict_proxy_positive_orgs",
    "strict_proxy_label_reason",
]].merge(
    source_cultures,
    on=["subject_id", "hadm_id"],
    how="left",
)
source_matches["hours_from_blood_culture"] = (
    (source_matches["culture_time"] - source_matches["strict_proxy_culture_time"]).dt.total_seconds() / 3600
)
source_matches = source_matches[
    source_matches["hours_from_blood_culture"].abs() <= SOURCE_WINDOW_HOURS
].copy()
source_matches["organism_concordant_with_blood"] = source_matches.apply(
    lambda row: organisms_concordant(row["strict_proxy_positive_orgs"], row["org_name"]),
    axis=1,
)

source_detail_cols = [
    "episode_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "strict_proxy_culture_time",
    "strict_proxy_positive_orgs",
    "strict_proxy_label_reason",
    "microevent_id",
    "culture_time",
    "hours_from_blood_culture",
    "source_bucket",
    "organism_concordant_with_blood",
    "spec_type_desc",
    "test_name",
    "org_name",
]
source_matches = source_matches[source_detail_cols].sort_values([
    "episode_id",
    "source_bucket",
    "culture_time",
    "org_name",
])

source_summary = (
    source_matches
    .groupby("episode_id", as_index=False)
    .agg(
        nearby_nonblood_source_culture_count=("microevent_id", "count"),
        concordant_nonblood_source_culture_count=("organism_concordant_with_blood", "sum"),
        nearby_nonblood_source_buckets=("source_bucket", source_bucket_list),
        nearby_nonblood_source_specimens=("spec_type_desc", join_unique),
        nearby_nonblood_source_orgs=("org_name", join_unique),
        nearest_source_culture_abs_hours=("hours_from_blood_culture", lambda s: float(s.abs().min()) if len(s) else np.nan),
    )
)
source_summary["nonconcordant_nonblood_source_culture_count"] = (
    source_summary["nearby_nonblood_source_culture_count"]
    - source_summary["concordant_nonblood_source_culture_count"]
)


# %% Load source-related ICD evidence

print("")
print("Loading ICD source-evidence screen...")
diagnoses = pd.read_csv(HOSP / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
d_icd = pd.read_csv(HOSP / "d_icd_diagnoses.csv.gz")
diagnoses = diagnoses.merge(d_icd, on=["icd_code", "icd_version"], how="left")
diagnoses = diagnoses[diagnoses["hadm_id"].isin(strict_hadm_ids)].copy()
diagnoses["icd_source_bucket"] = diagnoses["long_title"].apply(lambda x: first_matching_source(x, ICD_SOURCE_PATTERNS))
source_icd = diagnoses[diagnoses["icd_source_bucket"].ne("")].copy()

icd_summary = (
    source_icd
    .groupby("hadm_id", as_index=False)
    .agg(
        hadm_source_icd_count=("icd_code", "count"),
        hadm_source_icd_buckets=("icd_source_bucket", source_bucket_list),
        hadm_source_icd_titles=("long_title", join_unique),
    )
)

print(f"  Source-related ICD rows in strict-positive admissions: {len(source_icd):,}")


# %% Build episode-level source-screened labels

print("")
print("Building episode-level source-screened label hierarchy...")
label_cols = [
    "episode_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "anchor_year_group",
    "exposure_start",
    "exposure_end_observed",
    "eligible_48h_observed_exposure",
    "cvc_bsi_broad_proxy",
    "broad_proxy_culture_time",
    "cvc_bsi_strict_proxy",
    "strict_proxy_culture_time",
    "strict_proxy_positive_orgs",
    "strict_proxy_label_reason",
    "strict_proxy_qualifying_culture_rows",
    "strict_proxy_clear_pathogen_rows",
    "strict_proxy_commensal_rows",
    "early_positive_culture",
]
episode_labels = episodes[label_cols].copy()
episode_labels = episode_labels.merge(source_summary, on="episode_id", how="left")
episode_labels = episode_labels.merge(icd_summary, on="hadm_id", how="left")

for col in [
    "nearby_nonblood_source_culture_count",
    "concordant_nonblood_source_culture_count",
    "nonconcordant_nonblood_source_culture_count",
    "hadm_source_icd_count",
]:
    episode_labels[col] = episode_labels[col].fillna(0).astype(int)

for col in [
    "nearby_nonblood_source_buckets",
    "nearby_nonblood_source_specimens",
    "nearby_nonblood_source_orgs",
    "hadm_source_icd_buckets",
    "hadm_source_icd_titles",
]:
    episode_labels[col] = episode_labels[col].fillna("")

episode_labels["source_screen_class"] = episode_labels.apply(classify_source_screen, axis=1)
episode_labels["cvc_bsi_strict_primary_likely_proxy"] = (
    episode_labels["source_screen_class"].eq("strict_proxy_primary_likely")
).astype(int)
episode_labels["cvc_bsi_strict_secondary_possible_proxy"] = (
    episode_labels["source_screen_class"].eq("strict_proxy_secondary_possible_concordant_culture")
).astype(int)
episode_labels["cvc_bsi_strict_uncertain_source_proxy"] = (
    episode_labels["source_screen_class"].str.startswith("strict_proxy_uncertain")
).astype(int)
episode_labels["cvc_bsi_strict_primary_or_uncertain_proxy"] = (
    episode_labels["cvc_bsi_strict_primary_likely_proxy"].eq(1)
    | episode_labels["cvc_bsi_strict_uncertain_source_proxy"].eq(1)
).astype(int)
episode_labels["strict_primary_likely_culture_time"] = episode_labels["strict_proxy_culture_time"].where(
    episode_labels["cvc_bsi_strict_primary_likely_proxy"].eq(1),
    pd.NaT,
)
episode_labels["strict_primary_or_uncertain_culture_time"] = episode_labels["strict_proxy_culture_time"].where(
    episode_labels["cvc_bsi_strict_primary_or_uncertain_proxy"].eq(1),
    pd.NaT,
)
episode_labels["strict_secondary_possible_culture_time"] = episode_labels["strict_proxy_culture_time"].where(
    episode_labels["cvc_bsi_strict_secondary_possible_proxy"].eq(1),
    pd.NaT,
)


# %% Propagate source-screened label to daily landmarks

print("")
print("Propagating source-screened labels to daily landmarks...")
landmarks = pd.read_csv(LANDMARK_FILE, parse_dates=["landmark_time", "exposure_end_observed"])
landmark_label_cols = [
    "episode_id",
    "source_screen_class",
    "cvc_bsi_strict_primary_likely_proxy",
    "cvc_bsi_strict_secondary_possible_proxy",
    "cvc_bsi_strict_uncertain_source_proxy",
    "cvc_bsi_strict_primary_or_uncertain_proxy",
    "strict_primary_likely_culture_time",
    "strict_primary_or_uncertain_culture_time",
    "strict_secondary_possible_culture_time",
    "nearby_nonblood_source_culture_count",
    "concordant_nonblood_source_culture_count",
    "nonconcordant_nonblood_source_culture_count",
    "nearby_nonblood_source_buckets",
    "hadm_source_icd_count",
    "hadm_source_icd_buckets",
]
landmark_labels = landmarks.merge(
    episode_labels[landmark_label_cols],
    on="episode_id",
    how="left",
)
landmark_labels = derive_horizon_target(
    landmark_labels,
    "strict_primary_likely_culture_time",
    "future_strict_primary_likely_cvc_bsi_proxy_7d",
    horizon_hours=168,
)
landmark_labels = derive_horizon_target(
    landmark_labels,
    "strict_primary_or_uncertain_culture_time",
    "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d",
    horizon_hours=168,
)
landmark_labels = derive_horizon_target(
    landmark_labels,
    "strict_secondary_possible_culture_time",
    "future_strict_secondary_possible_cvc_bsi_proxy_7d",
    horizon_hours=168,
)


# %% Audits

label_audit = pd.DataFrame([
    {
        "metric": "episodes_total",
        "value": int(len(episode_labels)),
    },
    {
        "metric": "eligible_48h_observed_exposure_episodes",
        "value": int(episode_labels["eligible_48h_observed_exposure"].sum()),
    },
    {
        "metric": "broad_cvc_associated_bsi_proxy_episodes",
        "value": int(episode_labels["cvc_bsi_broad_proxy"].sum()),
    },
    {
        "metric": "strict_cvc_associated_bsi_proxy_episodes",
        "value": int(episode_labels["cvc_bsi_strict_proxy"].sum()),
    },
    {
        "metric": "strict_primary_likely_proxy_episodes",
        "value": int(episode_labels["cvc_bsi_strict_primary_likely_proxy"].sum()),
    },
    {
        "metric": "strict_primary_or_uncertain_proxy_episodes",
        "value": int(episode_labels["cvc_bsi_strict_primary_or_uncertain_proxy"].sum()),
    },
    {
        "metric": "strict_secondary_possible_concordant_proxy_episodes",
        "value": int(episode_labels["cvc_bsi_strict_secondary_possible_proxy"].sum()),
    },
    {
        "metric": "strict_positive_with_nearby_nonblood_source_culture",
        "value": int((episode_labels["cvc_bsi_strict_proxy"].eq(1) & episode_labels["nearby_nonblood_source_culture_count"].gt(0)).sum()),
    },
    {
        "metric": "strict_positive_with_concordant_nonblood_source_culture",
        "value": int((episode_labels["cvc_bsi_strict_proxy"].eq(1) & episode_labels["concordant_nonblood_source_culture_count"].gt(0)).sum()),
    },
    {
        "metric": "strict_positive_with_source_icd",
        "value": int((episode_labels["cvc_bsi_strict_proxy"].eq(1) & episode_labels["hadm_source_icd_count"].gt(0)).sum()),
    },
    {
        "metric": "landmark_rows",
        "value": int(len(landmark_labels)),
    },
    {
        "metric": "future_strict_original_7d_rows",
        "value": int(landmark_labels["future_strict_cvc_bsi_proxy_7d"].sum()),
    },
    {
        "metric": "future_strict_primary_likely_7d_rows",
        "value": int(landmark_labels["future_strict_primary_likely_cvc_bsi_proxy_7d"].sum()),
    },
    {
        "metric": "future_strict_primary_or_uncertain_7d_rows",
        "value": int(landmark_labels["future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"].sum()),
    },
    {
        "metric": "future_strict_secondary_possible_concordant_7d_rows",
        "value": int(landmark_labels["future_strict_secondary_possible_cvc_bsi_proxy_7d"].sum()),
    },
])

class_counts = (
    episode_labels
    .groupby("source_screen_class", as_index=False)
    .agg(
        episodes=("episode_id", "count"),
        strict_proxy_episodes=("cvc_bsi_strict_proxy", "sum"),
        broad_proxy_episodes=("cvc_bsi_broad_proxy", "sum"),
    )
)
class_counts["episode_fraction"] = class_counts["episodes"] / len(episode_labels)

source_bucket_counts = (
    source_matches
    .groupby(["source_bucket", "spec_type_desc"], as_index=False)
    .agg(
        culture_rows=("microevent_id", "count"),
        episodes=("episode_id", "nunique"),
        organisms=("org_name", join_unique),
    )
    .sort_values(["episodes", "culture_rows"], ascending=False)
)

strict_org_counts = (
    episode_labels[episode_labels["cvc_bsi_strict_proxy"].eq(1)]
    .assign(strict_proxy_positive_orgs=lambda x: x["strict_proxy_positive_orgs"].fillna(""))
    .groupby(["source_screen_class", "strict_proxy_positive_orgs"], as_index=False)
    .agg(episodes=("episode_id", "count"))
    .sort_values(["source_screen_class", "episodes"], ascending=[True, False])
)


# %% Save outputs

episode_label_file = DATA_PATH / "v0_5_run22_source_screened_episode_labels.csv"
landmark_label_file = DATA_PATH / "v0_5_run22_source_screened_daily_landmarks.csv"
source_detail_file = OUTPUT_PATH / "v0_5_run22_secondary_source_culture_detail.csv"
icd_detail_file = OUTPUT_PATH / "v0_5_run22_source_icd_detail.csv"
label_audit_file = OUTPUT_PATH / "v0_5_run22_source_screen_label_audit.csv"
class_counts_file = OUTPUT_PATH / "v0_5_run22_source_screen_class_counts.csv"
source_bucket_file = OUTPUT_PATH / "v0_5_run22_secondary_source_bucket_counts.csv"
organism_counts_file = OUTPUT_PATH / "v0_5_run22_strict_organism_by_source_class.csv"
manifest_file = OUTPUT_PATH / "v0_5_run22_manifest.csv"

episode_labels.to_csv(episode_label_file, index=False)
landmark_labels.to_csv(landmark_label_file, index=False)
source_matches.to_csv(source_detail_file, index=False)
source_icd.to_csv(icd_detail_file, index=False)
label_audit.to_csv(label_audit_file, index=False)
class_counts.to_csv(class_counts_file, index=False)
source_bucket_counts.to_csv(source_bucket_file, index=False)
strict_org_counts.to_csv(organism_counts_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "episode_source_screened_labels", "path": str(episode_label_file)},
    {"artifact": "daily_landmarks_with_source_screened_targets", "path": str(landmark_label_file)},
    {"artifact": "secondary_source_culture_detail", "path": str(source_detail_file)},
    {"artifact": "source_icd_detail", "path": str(icd_detail_file)},
    {"artifact": "label_audit", "path": str(label_audit_file)},
    {"artifact": "source_screen_class_counts", "path": str(class_counts_file)},
    {"artifact": "secondary_source_bucket_counts", "path": str(source_bucket_file)},
    {"artifact": "strict_organism_by_source_class", "path": str(organism_counts_file)},
])
manifest.to_csv(manifest_file, index=False)


# %% Console summary

print("")
print("Run 22 source-screen label audit:")
print(label_audit.to_string(index=False))
print("")
print("Source-screen classes:")
print(class_counts.to_string(index=False))
print("")
print(f"Saved source-screened episode labels to: {episode_label_file}")
print(f"Saved source-screened daily landmarks to: {landmark_label_file}")
print(f"Saved Run 22 outputs to: {OUTPUT_PATH}")
print("Data Extraction 02 v0.5 Secondary Source Label Audit complete.")

