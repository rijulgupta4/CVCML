# %% Imports and paths

import os
import re

import pandas as pd

PROJECT_PATH = r"C:\path\to\CVCML"
SOURCE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4d")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4e")

os.makedirs(DATA_PATH, exist_ok=True)

SOURCE_ANTIBIOTIC_FILE = os.path.join(SOURCE_DATA_PATH, "v0_4d_antibiotics_long.pkl")
SOURCE_VASOPRESSOR_FILE = os.path.join(SOURCE_DATA_PATH, "v0_4d_vasopressors_long.pkl")

CLEAN_ANTIBIOTIC_FILE = os.path.join(DATA_PATH, "v0_4e_systemic_antibiotics_long.pkl")
CLEAN_VASOPRESSOR_FILE = os.path.join(DATA_PATH, "v0_4e_vasopressors_long.pkl")
AUDIT_FILE = os.path.join(DATA_PATH, "v0_4e_cleaned_therapy_extraction_audit.csv")
COUNTS_FILE = os.path.join(DATA_PATH, "v0_4e_cleaned_therapy_counts.csv")


# %% Cleaning definitions

SYSTEMIC_ROUTES = {
    "IV",
    "IV INFUSION",
    "IV BOLUS",
    "IV DRIP",
    "PO",
    "PO/NG",
    "NG",
    "ORAL",
    "G TUBE",
    "J TUBE",
    "IM",
}

NON_SYSTEMIC_DRUG_PATTERN = re.compile(
    r"("
    r"ophth|ophthalmic|otic|both eyes|right eye|left eye|both ears|right ear|left ear|"
    r"topical|cream|ointment|gel|inhalation|intravitreal|enema|oral liquid|"
    r"antibiotic lock|lock|desensitization|graded challenge|challenge|"
    r"neomycin|bacitracin|polymyxin b sulfate opht|polymyxin b -trimethoprim"
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


# %% Helpers

def clean_antibiotics(df):
    df = df.copy()
    df["route_clean"] = df["route"].fillna("").str.upper().str.strip()
    df["antibiotic_name"] = df["drug"].fillna("").str.lower().str.strip()

    route_keep = df["route_clean"].isin(SYSTEMIC_ROUTES)
    drug_keep = ~df["drug"].fillna("").str.contains(NON_SYSTEMIC_DRUG_PATTERN)
    cleaned = df.loc[route_keep & drug_keep].copy()

    drug = cleaned["drug"].fillna("")
    cleaned["systemic_antibiotic_any"] = 1
    cleaned["broad_antibiotic"] = drug.str.contains(BROAD_PATTERN).astype(int)
    cleaned["anti_mrsa_antibiotic"] = drug.str.contains(ANTI_MRSA_PATTERN).astype(int)
    cleaned["antipseudomonal_antibiotic"] = drug.str.contains(ANTIPSEUDOMONAL_PATTERN).astype(int)
    cleaned["carbapenem_antibiotic"] = drug.str.contains(CARBAPENEM_PATTERN).astype(int)
    cleaned["anaerobe_antibiotic"] = drug.str.contains(ANAEROBE_PATTERN).astype(int)
    return cleaned


# %% Load Run 10 cached therapy extracts

if not os.path.exists(SOURCE_ANTIBIOTIC_FILE):
    raise FileNotFoundError(
        f"Missing Run 10 antibiotic extract: {SOURCE_ANTIBIOTIC_FILE}\n"
        "Run v0.4D therapy extraction before v0.4E cleaning."
    )
if not os.path.exists(SOURCE_VASOPRESSOR_FILE):
    raise FileNotFoundError(
        f"Missing Run 10 vasopressor extract: {SOURCE_VASOPRESSOR_FILE}\n"
        "Run v0.4D therapy extraction before v0.4E cleaning."
    )

antibiotics_raw = pd.read_pickle(SOURCE_ANTIBIOTIC_FILE)
vasopressors = pd.read_pickle(SOURCE_VASOPRESSOR_FILE)

print(f"Run 10 antibiotic rows loaded:  {len(antibiotics_raw):,}")
print(f"Run 10 vasopressor rows loaded: {len(vasopressors):,}")


# %% Clean antibiotic therapy

systemic_antibiotics = clean_antibiotics(antibiotics_raw)

systemic_antibiotics.to_pickle(CLEAN_ANTIBIOTIC_FILE)
vasopressors.to_pickle(CLEAN_VASOPRESSOR_FILE)

print("")
print(f"Systemic antibiotic rows retained: {len(systemic_antibiotics):,}")
print(f"Rows removed by cleaning:          {len(antibiotics_raw) - len(systemic_antibiotics):,}")
print("")
print("Top retained systemic antibiotics:")
print(systemic_antibiotics["antibiotic_name"].value_counts().head(25).to_string())


# %% Save audits

route_counts_before = antibiotics_raw["route"].fillna("MISSING").value_counts()
route_counts_after = systemic_antibiotics["route"].fillna("MISSING").value_counts()

audit = pd.DataFrame([{
    "raw_antibiotic_rows": int(len(antibiotics_raw)),
    "systemic_antibiotic_rows": int(len(systemic_antibiotics)),
    "removed_antibiotic_rows": int(len(antibiotics_raw) - len(systemic_antibiotics)),
    "retained_fraction": float(len(systemic_antibiotics) / max(len(antibiotics_raw), 1)),
    "raw_vasopressor_rows": int(len(vasopressors)),
    "systemic_routes": ", ".join(sorted(SYSTEMIC_ROUTES)),
    "non_systemic_drug_pattern": NON_SYSTEMIC_DRUG_PATTERN.pattern,
    "clean_antibiotic_file": CLEAN_ANTIBIOTIC_FILE,
    "clean_vasopressor_file": CLEAN_VASOPRESSOR_FILE,
}])
audit.to_csv(AUDIT_FILE, index=False)

count_rows = []
for name, count in systemic_antibiotics["antibiotic_name"].value_counts().head(100).items():
    count_rows.append({"source": "systemic_antibiotics", "name": name, "rows": int(count)})
for route, count in route_counts_before.items():
    count_rows.append({"source": "route_before_cleaning", "name": route, "rows": int(count)})
for route, count in route_counts_after.items():
    count_rows.append({"source": "route_after_cleaning", "name": route, "rows": int(count)})
if len(vasopressors):
    for name, count in vasopressors["vasopressor_name"].value_counts().items():
        count_rows.append({"source": "vasopressors", "name": name, "rows": int(count)})
pd.DataFrame(count_rows).to_csv(COUNTS_FILE, index=False)

print("")
print(f"Cleaned therapy audit saved to:  {AUDIT_FILE}")
print(f"Cleaned therapy counts saved to: {COUNTS_FILE}")
print("")
print("Data Extraction 01 v0.4E Cleaned Therapy complete.")

