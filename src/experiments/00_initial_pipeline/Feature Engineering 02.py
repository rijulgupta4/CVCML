# %% Imports and load cohort

import pandas as pd
import numpy as np
import os

# Paths
MIMIC_PATH = r"C:\path\to\mimic-iv"
HOSP = os.path.join(MIMIC_PATH, "hosp")
ICU  = os.path.join(MIMIC_PATH, "icu")
DATA_PATH  = r"C:\path\to\CVCML\data"

# Load cohort from Phase 1
cohort = pd.read_csv(
    os.path.join(DATA_PATH, "clabsi_cohort.csv"),
    parse_dates=['starttime', 'endtime']
)

# Explicitly convert culture_time â€” contains NaT so needs errors='coerce'
cohort['culture_time'] = pd.to_datetime(cohort['culture_time'], errors='coerce')

print(f"Cohort loaded: {cohort.shape}")
print(f"CLABSI positive:      {cohort['clabsi'].sum():,}")
print(f"CLABSI negative:      {(cohort['clabsi']==0).sum():,}")
print(f"Culture times loaded: {cohort['culture_time'].notna().sum():,}")
print(f"\nDtypes:")
print(cohort[['starttime','endtime','culture_time']].dtypes)

# %% Define reference time and compute 48-hour lookback window
# CLABSI+ â†’ reference time = first positive culture draw time
# CLABSI- â†’ reference time = CVC removal time (endtime)
# Source: consistent with temporal validation approach in Shrestha et al., Frontiers 2025

cohort['ref_time'] = np.where(
    cohort['clabsi'] == 1,
    cohort['culture_time'],
    cohort['endtime']
)
cohort['ref_time']     = pd.to_datetime(cohort['ref_time'])
cohort['window_start'] = cohort['ref_time'] - pd.Timedelta(hours=48)

print("Reference time logic:")
print(f"  CLABSI+ uses culture_time: {(cohort['clabsi'] == 1).sum():,} stays")
print(f"  CLABSI- uses endtime:      {(cohort['clabsi'] == 0).sum():,} stays")
print(f"\nWindow duration check (should all be 48hrs):")
window_hrs = (cohort['ref_time'] - cohort['window_start']).dt.total_seconds() / 3600
print(f"  Min: {window_hrs.min():.0f}hrs  Max: {window_hrs.max():.0f}hrs  Mean: {window_hrs.mean():.0f}hrs")
print(f"\nSample:")
print(cohort[['stay_id', 'clabsi', 'window_start', 'ref_time']].head(5).to_string(index=False))

# %% Load d_labitems and search for key labs

d_labitems = pd.read_csv(os.path.join(HOSP, "d_labitems.csv.gz"))

# Search for our target labs
target_labs = ['white blood cell', 'wbc', 'lactate', 'c-reactive', 'crp',
               'temperature', 'hemoglobin', 'platelet', 'creatinine', 'albumin']

lab_search = d_labitems[
    d_labitems['label'].str.contains(
        '|'.join(target_labs), case=False, na=False
    )
]

print(f"Found {len(lab_search)} matching lab items:\n")
print(lab_search[['itemid', 'label', 'fluid', 'category']].to_string(index=False))

# %% Define lab item IDs for CLABSI-relevant features
# Source: MIMIC-IV d_labitems, filtered to blood specimens only

LAB_ITEMS = {
    'wbc':        [51301],           # White Blood Cells, Blood, Hematology
    'lactate':    [50813, 52442],    # Lactate, Blood, Blood Gas (two MIMIC periods)
    'crp':        [50889, 51652],    # C-Reactive Protein + High-Sensitivity CRP
    'hemoglobin': [51222],           # Hemoglobin, Blood, Hematology
    'platelets':  [51265],           # Platelet Count, Blood, Hematology
    'creatinine': [50912],           # Creatinine, Blood, Chemistry
    'albumin':    [50862],           # Albumin, Blood, Chemistry
}

# Flatten to a single list for filtering labevents
all_lab_ids = [itemid for ids in LAB_ITEMS.values() for itemid in ids]

print("Lab item IDs to extract:")
for lab, ids in LAB_ITEMS.items():
    print(f"  {lab:<12}: {ids}")
print(f"\nTotal unique item IDs: {len(all_lab_ids)}")

# %% Load labevents filtered to cohort patients and target labs
# labevents is large - filter on load to manage memory
# Source: MIMIC-IV labevents table, hosp module

print("Loading labevents (this may take a minute)...")

# Get set of subject_ids in our cohort for filtering
cohort_subjects = set(cohort['subject_id'].values)

# Load in chunks, keeping only rows we need
chunks = []
chunk_size = 500000

for chunk in pd.read_csv(
    os.path.join(HOSP, "labevents.csv.gz"),
    chunksize=chunk_size,
    low_memory=False,
    parse_dates=['charttime']
):
    # Filter to cohort patients AND target lab items only
    filtered = chunk[
        chunk['subject_id'].isin(cohort_subjects) &
        chunk['itemid'].isin(all_lab_ids)
    ]
    if len(filtered) > 0:
        chunks.append(filtered)

labevents = pd.concat(chunks, ignore_index=True)

print(f"  Total lab rows for cohort: {len(labevents):,}")
print(f"  Unique patients:           {labevents['subject_id'].nunique():,}")
print(f"\n  Rows per lab type:")
# Map itemid back to lab name for readability
id_to_lab = {itemid: lab for lab, ids in LAB_ITEMS.items() for itemid in ids}
labevents['lab_name'] = labevents['itemid'].map(id_to_lab)
print(labevents['lab_name'].value_counts().to_string())
print(f"\n  Sample rows:")
print(labevents[['subject_id', 'charttime', 'lab_name', 'valuenum']].head(5).to_string(index=False))

# %% Join labevents to cohort 48-hour window
print("Joining labs to 48-hour window...")

# Merge labs to cohort on subject_id to get window times
labs_merged = labevents.merge(
    cohort[['subject_id', 'stay_id', 'window_start', 'ref_time']],
    on='subject_id',
    how='inner'
)

# Keep only lab values falling within the 48-hour lookback window
labs_windowed = labs_merged[
    (labs_merged['charttime'] >= labs_merged['window_start']) &
    (labs_merged['charttime'] <= labs_merged['ref_time'])
].copy()

print(f"  Lab rows before windowing:   {len(labs_merged):,}")
print(f"  Lab rows within 48hr window: {len(labs_windowed):,}")
print(f"  Unique stays with lab data:  {labs_windowed['stay_id'].nunique():,}")
print(f"  Stays with NO lab data:      {cohort['stay_id'].nunique() - labs_windowed['stay_id'].nunique():,}")
print(f"\n  Lab counts within window:")
print(labs_windowed['lab_name'].value_counts().to_string())
print(f"\n  Sample:")
print(labs_windowed[['stay_id', 'charttime', 'lab_name', 'valuenum']].head(5).to_string(index=False))

# %% Aggregate lab values into per-stay features
# For each lab compute: mean, last value, and trend (last - first)
# Trend captures direction of change which is clinically more meaningful than snapshot
# Source: feature engineering approach consistent with Shrestha et al., Frontiers 2025

print("Aggregating lab features per stay...")

# Sort by time so first/last are meaningful
labs_windowed = labs_windowed.sort_values(['stay_id', 'lab_name', 'charttime'])

# Aggregate
lab_features = labs_windowed.groupby(['stay_id', 'lab_name'])['valuenum'].agg(
    mean_val  = 'mean',
    last_val  = 'last',
    first_val = 'first'
).reset_index()

# Compute trend = last - first (positive = rising, negative = falling)
lab_features['trend'] = lab_features['last_val'] - lab_features['first_val']

# Pivot to wide format â€” one row per stay, one column per lab per metric
lab_pivot = lab_features.pivot_table(
    index   = 'stay_id',
    columns = 'lab_name',
    values  = ['mean_val', 'last_val', 'trend']
)

# Flatten column names â€” e.g. ('mean_val', 'wbc') â†’ 'wbc_mean'
lab_pivot.columns = [f"{lab}_{metric.replace('_val','')}" 
                     for metric, lab in lab_pivot.columns]
lab_pivot = lab_pivot.reset_index()

print(f"  Lab feature columns generated: {len(lab_pivot.columns) - 1}")
print(f"  Stays with lab features:       {len(lab_pivot):,}")
print(f"\n  Columns:")
print([c for c in lab_pivot.columns if c != 'stay_id'])
print(f"\n  Sample row:")
print(lab_pivot.head(2).to_string(index=False))

# %% Merge lab features into cohort
cohort_featured = cohort.merge(lab_pivot, on='stay_id', how='left')

print(f"Cohort shape after lab features: {cohort_featured.shape}")
print(f"\nMissing values per lab feature:")
lab_cols = [c for c in cohort_featured.columns if any(
    c.startswith(lab) for lab in ['wbc','lactate','crp','hemoglobin',
                                   'platelets','creatinine','albumin']
)]
print(cohort_featured[lab_cols].isnull().sum().to_string())
print(f"\nOverall missing rate per feature:")
print((cohort_featured[lab_cols].isnull().sum() / len(cohort_featured) * 100).round(1).to_string())

# %% Drop unusable features and handle informative missingness
# CRP dropped: 97.9% missing - not routinely ordered in US ICUs
# Source: Yeh et al., Critical Care Medicine, 2023 - CRP underutilized in US vs Europe
# Albumin dropped: 73% missing - insufficient coverage for reliable prediction

# Add informative missingness flag for lactate before dropping anything
# Lactate missingness is clinically meaningful - ordered when infection suspected
# Source: Mikkelsen et al., Critical Care Medicine, 2009
cohort_featured['lactate_measured'] = (
    cohort_featured['lactate_last'].notna()
).astype(int)

# Drop CRP and albumin columns
drop_cols = [c for c in cohort_featured.columns
             if c.startswith('crp') or c.startswith('albumin')]
cohort_featured = cohort_featured.drop(columns=drop_cols)

print(f"Dropped {len(drop_cols)} columns: {drop_cols}")
print(f"\nCohort shape: {cohort_featured.shape}")
print(f"\nLactate measured flag:")
print(cohort_featured['lactate_measured'].value_counts().to_string())
print(f"\nRemaining missing rates:")
lab_cols = [c for c in cohort_featured.columns if any(
    c.startswith(lab) for lab in ['wbc','lactate','hemoglobin',
                                   'platelets','creatinine']
)]
print((cohort_featured[lab_cols].isnull().sum() / 
       len(cohort_featured) * 100).round(1).to_string())

# %% Check how many patients we'd lose with listwise deletion
# Core lab columns only (excluding lactate which we know is 62% missing)
core_lab_cols = [c for c in cohort_featured.columns if any(
    c.startswith(lab) for lab in ['wbc', 'hemoglobin', 'platelets', 'creatinine']
)]

# How many stays have ANY missing core lab value
missing_any_core = cohort_featured[core_lab_cols].isnull().any(axis=1).sum()
missing_all_core = cohort_featured[core_lab_cols].isnull().all(axis=1).sum()

# How many would survive complete case analysis (no missing at all)
complete_cases = cohort_featured[core_lab_cols].dropna().shape[0]

# Check CLABSI distribution in missing vs non-missing
missing_mask = cohort_featured[core_lab_cols].isnull().any(axis=1)
print(f"Total stays:                    {len(cohort_featured):,}")
print(f"Stays missing ANY core lab:     {missing_any_core:,} ({missing_any_core/len(cohort_featured)*100:.1f}%)")
print(f"Stays missing ALL core labs:    {missing_all_core:,}")
print(f"Stays surviving deletion:       {complete_cases:,}")
print(f"\nCLABSI rate in complete cases:")
complete_mask = ~missing_mask
print(f"  Complete cases CLABSI rate:   {cohort_featured[complete_mask]['clabsi'].mean()*100:.1f}%")
print(f"  Missing cases CLABSI rate:    {cohort_featured[missing_mask]['clabsi'].mean()*100:.1f}%")
print(f"\nIf we also drop missing lactate:")
lactate_mask = cohort_featured['lactate_last'] == 0
full_complete = cohort_featured[~missing_mask & ~lactate_mask]
print(f"  Stays remaining: {len(full_complete):,} ({len(full_complete)/len(cohort_featured)*100:.1f}%)")
print(f"  CLABSI rate:     {full_complete['clabsi'].mean()*100:.1f}%")

# %% Listwise deletion for missing core lab values
# Missing core labs (2.5%) show CLABSI rate of 1.1% vs 5.8% in complete cases
# Missingness is informative (not MAR) â€” imputation would introduce selection bias
# Dropping 274 patients is negligible loss with improved data integrity
# Source: Sterne et al., BMJ 2009 â€” listwise deletion appropriate for informative missingness

before = len(cohort_featured)
clabsi_before = cohort_featured['clabsi'].sum()

core_lab_cols = [c for c in cohort_featured.columns if any(
    c.startswith(lab) for lab in ['wbc', 'hemoglobin', 'platelets', 'creatinine']
)]

cohort_featured = cohort_featured.dropna(subset=core_lab_cols).reset_index(drop=True)

after = len(cohort_featured)
clabsi_after = cohort_featured['clabsi'].sum()

print(f"Before deletion: {before:,} stays | {clabsi_before} CLABSI positive")
print(f"After deletion:  {after:,} stays  | {clabsi_after} CLABSI positive")
print(f"Dropped:         {before - after:,} stays")
print(f"\nCLABSI rate before: {clabsi_before/before*100:.1f}%")
print(f"CLABSI rate after:  {clabsi_after/after*100:.1f}%")

# Zero imputation for lactate only
# Lactate missingness is informative â€” binary flag already captures clinical signal
# Source: Mikkelsen et al., Critical Care Medicine, 2009
lactate_cols = [c for c in cohort_featured.columns if c.startswith('lactate')]
for col in lactate_cols:
    cohort_featured[col] = cohort_featured[col].fillna(0)

print(f"\nLactate zero imputation applied")
print(f"Final missing values: {cohort_featured.isnull().sum().sum()}")

# %% Diagnose remaining missing values
print("Missing values by column:")
missing = cohort_featured.isnull().sum()
print(missing[missing > 0].to_string())

# %% Encode categorical variables
# Convert text categories to numeric for XGBoost
# Using pandas get_dummies (one-hot encoding) for nominal variables
# Source: standard approach for nominal clinical variables without ordinal relationship

print("Columns before encoding:")
cat_cols = cohort_featured.select_dtypes(include='object').columns.tolist()
# Exclude datetime and identifier columns
cat_cols = [c for c in cat_cols if c not in ['culture_time']]
print(cat_cols)

cohort_encoded = pd.get_dummies(
    cohort_featured,
    columns=['gender', 'cvc_type', 'admission_type', 
             'insurance', 'marital_status', 'race'],
    drop_first=False
)

print(f"\nShape before encoding: {cohort_featured.shape}")
print(f"Shape after encoding:  {cohort_encoded.shape}")
print(f"\nNew columns added:")
original_cols = set(cohort_featured.columns)
new_cols = [c for c in cohort_encoded.columns if c not in original_cols]
print(new_cols)

# %% Consolidate race categories to reduce sparse dummy variables
# Granular race subcategories create sparse columns that add noise
# Consolidating to 6 broad groups is standard in clinical ML literature
# Source: Chen et al., NEJM AI, 2024 - race consolidation in clinical prediction models

race_map = {
    'WHITE': 'White',
    'WHITE - BRAZILIAN': 'White',
    'WHITE - EASTERN EUROPEAN': 'White',
    'WHITE - OTHER EUROPEAN': 'White',
    'WHITE - RUSSIAN': 'White',
    'BLACK/AFRICAN AMERICAN': 'Black',
    'BLACK/AFRICAN': 'Black',
    'BLACK/CAPE VERDEAN': 'Black',
    'BLACK/CARIBBEAN ISLAND': 'Black',
    'ASIAN': 'Asian',
    'ASIAN - ASIAN INDIAN': 'Asian',
    'ASIAN - CHINESE': 'Asian',
    'ASIAN - KOREAN': 'Asian',
    'ASIAN - SOUTH EAST ASIAN': 'Asian',
    'HISPANIC OR LATINO': 'Hispanic',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic',
    'HISPANIC/LATINO - COLUMBIAN': 'Hispanic',
    'HISPANIC/LATINO - CUBAN': 'Hispanic',
    'HISPANIC/LATINO - DOMINICAN': 'Hispanic',
    'HISPANIC/LATINO - GUATEMALAN': 'Hispanic',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic',
    'HISPANIC/LATINO - SALVADORAN': 'Hispanic',
    'SOUTH AMERICAN': 'Hispanic',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other',
    'AMERICAN INDIAN/ALASKA NATIVE': 'Other',
    'MULTIPLE RACE/ETHNICITY': 'Other',
    'PORTUGUESE': 'Other',
    'OTHER': 'Other',
    'PATIENT DECLINED TO ANSWER': 'Unknown',
    'UNABLE TO OBTAIN': 'Unknown',
    'UNKNOWN': 'Unknown',
}

# Apply consolidation to original cohort_featured before re-encoding
cohort_featured['race_consolidated'] = cohort_featured['race'].map(race_map).fillna('Unknown')

# Drop old race column
cohort_featured = cohort_featured.drop(columns=['race'])

# Re-encode everything cleanly
cohort_encoded = pd.get_dummies(
    cohort_featured,
    columns=['gender', 'cvc_type', 'admission_type',
             'insurance', 'marital_status', 'race_consolidated'],
    drop_first=False
)

print(f"Shape after race consolidation: {cohort_encoded.shape}")
print(f"\nRace distribution:")
print(cohort_featured['race_consolidated'].value_counts().to_string())
print(f"\nRace columns in encoded dataset:")
print([c for c in cohort_encoded.columns if c.startswith('race')])

# %% Save final feature matrix
# Drop metadata columns not used in modeling
# Keep subject_id, hadm_id, stay_id for reference but exclude from features later

# Columns to exclude from modeling (identifiers and metadata)
meta_cols = ['subject_id', 'hadm_id', 'stay_id', 
             'starttime', 'endtime', 'culture_time',
             'ref_time', 'window_start']

# Confirm all meta cols exist before dropping
meta_cols = [c for c in meta_cols if c in cohort_encoded.columns]

# Save full dataset including metadata â€” we separate features in Phase 3
output_file = os.path.join(DATA_PATH, "clabsi_features.csv")
cohort_encoded.to_csv(output_file, index=False)

print(f"Feature matrix saved to: {output_file}")
print(f"Shape: {cohort_encoded.shape}")
print(f"\nFinal column list:")
feature_cols = [c for c in cohort_encoded.columns if c not in meta_cols]
print(f"  Metadata columns: {len(meta_cols)}")
print(f"  Feature columns:  {len([c for c in feature_cols if c != 'clabsi'])}")
print(f"  Target column:    clabsi")
print(f"\nCLABSI distribution in final dataset:")
print(f"  Positive: {cohort_encoded['clabsi'].sum():,} ({cohort_encoded['clabsi'].mean()*100:.1f}%)")
print(f"  Negative: {(cohort_encoded['clabsi']==0).sum():,} ({(1-cohort_encoded['clabsi'].mean())*100:.1f}%)")
print(f"\nFeature Engineering 02.py complete.")



