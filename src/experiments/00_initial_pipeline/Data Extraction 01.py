# %% Imports and paths

import pandas as pd
import os

# â”€â”€ Set your MIMIC path here â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MIMIC_PATH = r"C:\path\to\mimic-iv"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU  = os.path.join(MIMIC_PATH, "icu")

# %% Load lightweight tables

print("Loading patients...")
patients = pd.read_csv(os.path.join(HOSP, "patients.csv.gz"))
print(f"  {len(patients):,} rows | columns: {list(patients.columns)}")

print("Loading admissions...")
admissions = pd.read_csv(os.path.join(HOSP, "admissions.csv.gz"))
print(f"  {len(admissions):,} rows | columns: {list(admissions.columns)}")

print("Loading icustays...")
icustays = pd.read_csv(os.path.join(ICU, "icustays.csv.gz"))
print(f"  {len(icustays):,} rows | columns: {list(icustays.columns)}")

print("Loading d_items...")
d_items = pd.read_csv(os.path.join(ICU, "d_items.csv.gz"))
print(f"  {len(d_items):,} rows | columns: {list(d_items.columns)}")

print("\nDone. All tables loaded successfully.")

# %% Search d_items for CVC-related entries

cvc_items = d_items[d_items['label'].str.contains(
    'catheter|central|CVC|PICC|central line|multilumen',
    case=False, na=False
)]

print(f"Found {len(cvc_items)} CVC-related items:\n")
print(cvc_items[['itemid', 'label', 'category', 'linksto']].to_string(index=False))

# %% Define CVC item IDs and load procedureevents

CVC_PROCEDURE_IDS = [
    224264,  # PICC Line
    224270,  # Dialysis Catheter
    224273,  # Presep Catheter
    224560,  # PA Catheter
    225203,  # Pheresis Catheter
    229517,  # Multi Lumen Cooling Catheter
]

CVC_INSERTION_DATE_IDS = [
    224184,  # PICC Line Insertion Date
    225322,  # Dialysis Catheter Insertion Date
    225354,  # PA Catheter Insertion Date
    225370,  # Pheresis Catheter Insertion Date
    225386,  # Presep Catheter Insertion Date
]

print("Loading procedureevents...")
procedureevents = pd.read_csv(os.path.join(ICU, "procedureevents.csv.gz"))
print(f"  {len(procedureevents):,} total rows")

# Filter to only CVC-related procedures
cvc_procedures = procedureevents[
    procedureevents['itemid'].isin(CVC_PROCEDURE_IDS)
].copy()

print(f"  {len(cvc_procedures):,} CVC procedure rows")
print(f"\n  Breakdown by catheter type:")
print(
    cvc_procedures.merge(
        d_items[['itemid','label']], on='itemid', how='left'
    )['label'].value_counts().to_string()
)

# %% Inspect cvc_procedures columns and sample rows

print("Columns:", list(cvc_procedures.columns))
print(f"\nSample rows:")
print(cvc_procedures.head(5).to_string(index=False))

# %% Build CVC cohort with dwell time

# Convert timestamps
cvc_procedures['starttime'] = pd.to_datetime(cvc_procedures['starttime'])
cvc_procedures['endtime']   = pd.to_datetime(cvc_procedures['endtime'])

# Calculate dwell time in hours (value column is in minutes)
cvc_procedures['dwell_hours'] = cvc_procedures['value'] / 60

# Keep only stays with CVC in place >= 48 hours (CDC CLABSI definition requirement)
cvc_cohort = cvc_procedures[cvc_procedures['dwell_hours'] >= 48].copy()

# Add catheter type label
cvc_cohort = cvc_cohort.merge(
    d_items[['itemid', 'label']], on='itemid', how='left'
).rename(columns={'label': 'cvc_type'})

# Keep only columns we need
cvc_cohort = cvc_cohort[[
    'subject_id', 'hadm_id', 'stay_id', 'caregiver_id',
    'starttime', 'endtime', 'dwell_hours',
    'cvc_type', 'location'
]].reset_index(drop=True)

print(f"Total CVC events:              {len(cvc_procedures):,}")
print(f"CVC events >= 48hr dwell time: {len(cvc_cohort):,}")
print(f"Unique ICU stays:              {cvc_cohort['stay_id'].nunique():,}")
print(f"Unique patients:               {cvc_cohort['subject_id'].nunique():,}")
print(f"\nInsertion site breakdown:")
print(cvc_cohort['location'].value_counts().head(10).to_string())
print(f"\nCatheter type breakdown:")
print(cvc_cohort['cvc_type'].value_counts().to_string())

# %% Deduplicate to one CVC event per stay (longest dwell time)
cvc_cohort = cvc_cohort.sort_values(
    'dwell_hours', ascending=False
).drop_duplicates(
    subset='stay_id', keep='first'
).reset_index(drop=True)

print(f"After deduplication:")
print(f"  Unique ICU stays: {len(cvc_cohort):,}")
print(f"  Unique patients:  {cvc_cohort['subject_id'].nunique():,}")
print(f"  Avg dwell time:   {cvc_cohort['dwell_hours'].mean():.1f} hours")
print(f"  Max dwell time:   {cvc_cohort['dwell_hours'].max():.1f} hours")
print(f"  Min dwell time:   {cvc_cohort['dwell_hours'].min():.1f} hours")

# %% Load microbiologyevents and identify CLABSI cases

print("Loading microbiologyevents...")
micro = pd.read_csv(os.path.join(HOSP, "microbiologyevents.csv.gz"))
print(f"  {len(micro):,} total rows")

# Keep only blood cultures
blood_cultures = micro[
    micro['spec_type_desc'].str.contains('blood', case=False, na=False)
].copy()

print(f"  {len(blood_cultures):,} blood culture rows")
print(f"\n  Specimen type breakdown:")
print(blood_cultures['spec_type_desc'].value_counts().to_string())
print(f"\n  Top 10 organisms cultured:")
print(blood_cultures['org_name'].value_counts().head(10).to_string())

# %% Create CLABSI labels and save culture timestamp

micro = pd.read_csv(
    os.path.join(HOSP, "microbiologyevents.csv.gz"),
    low_memory=False
)

# Keep only relevant blood culture types
blood_cultures = micro[
    micro['spec_type_desc'].isin([
        'BLOOD CULTURE',
        'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)'
    ])
].copy()

# Keep only positive cultures
positive_cultures = blood_cultures[
    blood_cultures['org_name'].notna() &
    ~blood_cultures['org_name'].str.contains('CANCELLED', case=False, na=False)
].copy()

# Convert timestamps
positive_cultures['charttime'] = pd.to_datetime(positive_cultures['charttime'])
cvc_cohort['starttime']        = pd.to_datetime(cvc_cohort['starttime'])
cvc_cohort['endtime']          = pd.to_datetime(cvc_cohort['endtime'])

# Join cultures to cohort on subject_id
merged = cvc_cohort.merge(
    positive_cultures[['subject_id', 'charttime', 'org_name']],
    on='subject_id',
    how='left'
)

# CLABSI = positive blood culture drawn while CVC was in place
merged['clabsi'] = (
    (merged['charttime'] >= merged['starttime']) &
    (merged['charttime'] <= merged['endtime'])
).astype(int)

# For CLABSI+ stays: save the FIRST positive culture time as reference time
# For CLABSI- stays: reference time will be CVC endtime (set in Phase 2)
culture_times = (
    merged[merged['clabsi'] == 1]
    .sort_values('charttime')
    .groupby('stay_id')['charttime']
    .first()
    .reset_index()
    .rename(columns={'charttime': 'culture_time'})
)

# Collapse to one row per stay
clabsi_labels = merged.groupby('stay_id')['clabsi'].max().reset_index()

# Merge labels and culture times back to cohort
cvc_cohort = cvc_cohort.merge(clabsi_labels, on='stay_id', how='left')
cvc_cohort = cvc_cohort.merge(culture_times, on='stay_id', how='left')
cvc_cohort['clabsi'] = cvc_cohort['clabsi'].fillna(0).astype(int)

# Report
n_clabsi    = cvc_cohort['clabsi'].sum()
n_total     = len(cvc_cohort)
clabsi_rate = n_clabsi / n_total * 100

print(f"Total CVC stays:       {n_total:,}")
print(f"CLABSI positive:       {n_clabsi:,}  ({clabsi_rate:.1f}%)")
print(f"CLABSI negative:       {n_total - n_clabsi:,}  ({100 - clabsi_rate:.1f}%)")
print(f"Culture time captured: {cvc_cohort['culture_time'].notna().sum():,} stays")
print(f"Missing culture time:  {cvc_cohort['culture_time'].isna().sum():,} stays (CLABSI- expected)")
# %% FLAG: Sensitivity analysis â€” strict CLABSI labeling (implement after pragmatic pipeline complete)
#
# Current approach: pragmatic labeling
# Any positive blood culture drawn while CVC in place = CLABSI
# Expected rate: 5-15% (Goto et al., Infect Control Hosp Epidemiol, 2014)
#
# STRICT APPROACH TO IMPLEMENT LATER:
#
# Filter 1 â€” Exclude common skin contaminants with only one positive culture
#   CDC NHSN definition requires >= 2 positive cultures for:
#   Coagulase-negative staph, Bacillus spp., Corynebacterium spp.,
#   Micrococcus spp., Propionibacterium spp.
#   Source: CDC NHSN Protocol, 2024
#   https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf
#
# Filter 2 â€” Exclude patients with documented alternative infection source
#   Check microbiologyevents for concurrent positive urine or respiratory cultures
#   within +/- 48hrs of blood culture draw
#   Source: Magill et al., NEJM, 2014 (HAI prevalence methodology)
#
# Filter 3 â€” Require clinical signs of infection in chartevents
#   Fever > 38C OR hypothermia < 36C OR hypotension (SBP < 90mmHg)
#   within 48hrs of positive culture
#   Source: CDC NHSN CLABSI criteria, 2024
#
# Expected outcome: CLABSI rate drops to ~1-3%, ~200-350 positive cases
# Purpose: sensitivity analysis comparing pragmatic vs strict labeling
# Value: adds methodological robustness not present in Frontiers 2025 paper

# %% Merge patient demographics
patients['anchor_age'] = pd.to_numeric(patients['anchor_age'], errors='coerce')

admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

# Keep only columns we need from each table
patients_slim = patients[[
    'subject_id', 'gender', 'anchor_age'
]].copy()

admissions_slim = admissions[[
    'subject_id', 'hadm_id', 'admission_type',
    'insurance', 'marital_status', 'race'
]].copy()

# Merge into cohort
cvc_cohort = cvc_cohort.merge(patients_slim, on='subject_id', how='left')
cvc_cohort = cvc_cohort.merge(admissions_slim, on=['subject_id', 'hadm_id'], how='left')

print(f"Cohort shape after demographics merge: {cvc_cohort.shape}")
print(f"\nColumns now available:")
print(list(cvc_cohort.columns))
print(f"\nMissing values per column:")
print(cvc_cohort.isnull().sum().to_string())

# %% Clean missing values and finalize cohort
# Drop caregiver_id - 78% missing, will derive better metric from chartevents in Phase 2
cvc_cohort = cvc_cohort.drop(columns=['caregiver_id'])

# Convert location to binary site_known flag
# location is meaningful but 79% missing - preserve as documentation quality indicator
# NOTE: specific site (IJ vs femoral vs subclavian) to be revisited if missingness improves
cvc_cohort['site_known'] = cvc_cohort['location'].notna().astype(int)
cvc_cohort = cvc_cohort.drop(columns=['location'])

# Fill minor missing values
cvc_cohort['insurance']      = cvc_cohort['insurance'].fillna('Unknown')
cvc_cohort['marital_status'] = cvc_cohort['marital_status'].fillna('Unknown')

# Final check
print(f"Final cohort shape: {cvc_cohort.shape}")
print(f"\nColumns: {list(cvc_cohort.columns)}")
print(f"\nMissing values:")
print(cvc_cohort.isnull().sum().to_string())
print(f"\nCLABSI distribution:")
print(cvc_cohort['clabsi'].value_counts().to_string())
print(f"\nSample rows:")
print(cvc_cohort.head(3).to_string(index=False))

# %% Save cohort to file
import os

# Create output directory if it doesn't exist
output_path = r"C:\path\to\CVCML\data"
os.makedirs(output_path, exist_ok=True)

# Save as CSV
cohort_file = os.path.join(output_path, "clabsi_cohort.csv")
cvc_cohort.to_csv(cohort_file, index=False)

print(f"Cohort saved to: {cohort_file}")
print(f"Shape: {cvc_cohort.shape}")
print(f"File size: {os.path.getsize(cohort_file) / 1024:.1f} KB")

# %% Verify saved cohort loads correctly
test = pd.read_csv(
    r"C:\path\to\CVCML\data\clabsi_cohort.csv",
    parse_dates=['starttime', 'endtime']
)
print(test.dtypes)
print(test[['starttime', 'endtime']].head(3))
