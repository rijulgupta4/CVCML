# %% Imports and paths

import os
import re

import pandas as pd

MIMIC_PATH = r"C:\path\to\mimic-iv"
PROJECT_PATH = r"C:\path\to\CVCML"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU = os.path.join(MIMIC_PATH, "icu")
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4d")

os.makedirs(DATA_PATH, exist_ok=True)

ANTIBIOTIC_CACHE_FILE = os.path.join(DATA_PATH, "v0_4d_antibiotics_long.pkl")
VASOPRESSOR_CACHE_FILE = os.path.join(DATA_PATH, "v0_4d_vasopressors_long.pkl")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4d_therapy_extraction_audit.csv")
COUNTS_FILE = os.path.join(DATA_PATH, "v0_4d_therapy_extraction_counts.csv")


# %% Therapy definitions

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
    r"("
    r"vancomycin|cefepime|ceftazidime|ceftaroline|piperacillin|tazobactam|zosyn|"
    r"meropenem|imipenem|ertapenem|aztreonam|ciprofloxacin|levofloxacin|"
    r"moxifloxacin|linezolid|daptomycin|colistin|polymyxin"
    r")",
    flags=re.IGNORECASE,
)
ANTI_MRSA_PATTERN = re.compile(r"(vancomycin|linezolid|daptomycin|ceftaroline)", flags=re.IGNORECASE)
ANTIPSEUDOMONAL_PATTERN = re.compile(
    r"(cefepime|ceftazidime|piperacillin|tazobactam|zosyn|meropenem|imipenem|aztreonam|ciprofloxacin|levofloxacin)",
    flags=re.IGNORECASE,
)
CARBAPENEM_PATTERN = re.compile(r"(meropenem|imipenem|ertapenem)", flags=re.IGNORECASE)
ANAEROBE_PATTERN = re.compile(r"(metronidazole|clindamycin|piperacillin|tazobactam|zosyn|meropenem|imipenem)", flags=re.IGNORECASE)

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


# %% Extract antibiotic prescriptions

if os.path.exists(ANTIBIOTIC_CACHE_FILE):
    print("")
    print(f"Antibiotic extraction already exists, reading: {ANTIBIOTIC_CACHE_FILE}")
    antibiotics = pd.read_pickle(ANTIBIOTIC_CACHE_FILE)
else:
    print("")
    print("Extracting antibiotic prescriptions...")
    antibiotic_chunks = []
    matched_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(HOSP, "prescriptions.csv.gz"),
            usecols=["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"],
            chunksize=500000,
            low_memory=False,
            parse_dates=["starttime", "stoptime"],
        ),
        start=1,
    ):
        filtered = chunk[
            chunk["hadm_id"].isin(cohort_hadm)
            & chunk["starttime"].notna()
            & chunk["drug"].fillna("").str.contains(ANTIBIOTIC_PATTERN)
        ].copy()
        if len(filtered) > 0:
            filtered = add_antibiotic_flags(filtered)
            antibiotic_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} prescription chunks | matched rows: {matched_rows:,}")

    antibiotics = (
        pd.concat(antibiotic_chunks, ignore_index=True)
        if antibiotic_chunks
        else pd.DataFrame()
    )
    antibiotics.to_pickle(ANTIBIOTIC_CACHE_FILE)
    print(f"Saved extracted antibiotics: {ANTIBIOTIC_CACHE_FILE}")

print(f"Total extracted antibiotic rows: {len(antibiotics):,}")


# %% Extract vasopressor/vasoactive medication inputevents

d_items = pd.read_csv(os.path.join(ICU, "d_items.csv.gz"), usecols=["itemid", "label", "category"])
d_items["label_lc"] = d_items["label"].fillna("").str.lower()
pressor_item_map = d_items[
    d_items["label_lc"].apply(lambda value: any(term in value for term in VASOPRESSOR_TERMS))
    & ~d_items["label_lc"].str.contains("intubation", na=False)
].copy()
pressor_item_map["vasopressor_name"] = pressor_item_map["label"].apply(classify_pressor)
pressor_itemids = set(pressor_item_map["itemid"].astype(int).values)

if os.path.exists(VASOPRESSOR_CACHE_FILE):
    print("")
    print(f"Vasopressor extraction already exists, reading: {VASOPRESSOR_CACHE_FILE}")
    vasopressors = pd.read_pickle(VASOPRESSOR_CACHE_FILE)
else:
    print("")
    print("Extracting vasopressor inputevents...")
    print(pressor_item_map[["itemid", "label", "vasopressor_name"]].to_string(index=False))

    pressor_chunks = []
    matched_rows = 0
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            os.path.join(ICU, "inputevents.csv.gz"),
            usecols=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "amountuom",
                "rate",
                "rateuom",
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
            & chunk["itemid"].isin(pressor_itemids)
            & chunk["starttime"].notna()
        ].copy()
        if len(filtered) > 0:
            filtered = filtered.merge(
                pressor_item_map[["itemid", "label", "vasopressor_name"]],
                on="itemid",
                how="left",
            )
            pressor_chunks.append(filtered)
            matched_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"  scanned {chunk_idx:,} input chunks | matched rows: {matched_rows:,}")

    vasopressors = (
        pd.concat(pressor_chunks, ignore_index=True)
        if pressor_chunks
        else pd.DataFrame()
    )
    vasopressors.to_pickle(VASOPRESSOR_CACHE_FILE)
    print(f"Saved extracted vasopressors: {VASOPRESSOR_CACHE_FILE}")

print(f"Total extracted vasopressor rows: {len(vasopressors):,}")


# %% Save audit and counts

audit = pd.DataFrame([{
    "source_cohort_stays": int(cohort["stay_id"].nunique()),
    "source_cohort_hadm": int(cohort["hadm_id"].nunique()),
    "source_strict_positive_stays": int(cohort["clabsi"].sum()),
    "antibiotic_regex": ANTIBIOTIC_PATTERN.pattern,
    "broad_antibiotic_regex": BROAD_PATTERN.pattern,
    "vasopressor_itemids": ", ".join(str(x) for x in sorted(pressor_itemids)),
    "extracted_antibiotic_rows": int(len(antibiotics)),
    "extracted_vasopressor_rows": int(len(vasopressors)),
    "antibiotic_cache_file": ANTIBIOTIC_CACHE_FILE,
    "vasopressor_cache_file": VASOPRESSOR_CACHE_FILE,
}])
audit.to_csv(AUDIT_FILE, index=False)

count_rows = []
if len(antibiotics):
    for name, count in antibiotics["antibiotic_name"].value_counts().head(100).items():
        count_rows.append({"source": "prescriptions", "name": name, "rows": int(count)})
if len(vasopressors):
    for name, count in vasopressors["vasopressor_name"].value_counts().items():
        count_rows.append({"source": "inputevents", "name": name, "rows": int(count)})
pd.DataFrame(count_rows).to_csv(COUNTS_FILE, index=False)

print("")
print(f"Therapy extraction audit saved to:  {AUDIT_FILE}")
print(f"Therapy extraction counts saved to: {COUNTS_FILE}")
print("")
print("Data Extraction 01 v0.4D Therapy Context complete.")

