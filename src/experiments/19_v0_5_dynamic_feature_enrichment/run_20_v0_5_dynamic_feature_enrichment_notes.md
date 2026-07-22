# Run 20 - v0.5 Dynamic Feature Enrichment

## Purpose

Run 20 tests whether the weak Run 19 validation ranking was mainly a feature-layer problem. It keeps the v0.5 catheter-episode denominator, daily seven-day landmark target, and temporal lockbox discipline unchanged, then adds dynamic physiology and therapy context.

No 2020-2022 lockbox predictions were generated.

## Inputs

- Run 18 v0.5 feature matrix: `data/v0_5/v0_5_run18_development_features.csv`
- v0.5 daily landmarks and strict seven-day target inherited from Runs 16-18.

## New Dynamic Features

Run 20 extracts v0.5-specific raw source rows and caches them:

- `chartevents`: vital signs for HR, RR, SpO2, temperature, SBP, DBP, MAP.
- `prescriptions`: systemic antibiotic exposure and antibiotic class flags.
- `inputevents`: vasopressor/vasoactive medication exposure.

Feature windows:

- 24-hour lookback.
- 48-hour lookback.

Feature families:

- Vital mean/min/max/last/trend/count/hours-since-last.
- Active antibiotics and new antibiotic starts by class.
- Active vasopressor rows, vasopressor starts, and rate summaries.

## Feature Engineering Output

- Enriched matrix: `data/v0_5/v0_5_run20_dynamic_enriched_features.csv`
- Shape: 64,752 rows x 207 columns.
- Cleaned vital rows cached: 18,394,028.
- Antibiotic rows cached: 135,371.
- Vasopressor rows cached: 440,692.

## Modeling Protocol

The Run 19 temporal protocol was preserved:

- Train core: 2008-2013.
- Calibration: 2014-2016.
- Validation: 2017-2019.
- Temporal lockbox: 2020-2022, audit only.

Models compared:

- XGBoost static + labs.
- XGBoost static + labs + vitals.
- XGBoost static + labs + vitals + therapy.

Each model was evaluated as:

- Raw probability.
- Platt-calibrated probability.
- Isotonic-calibrated probability.

## Key Validation Results

Validation prevalence: 5.34%.

Best PR-AUC:

- Static + labs raw/Platt: PR-AUC 0.063, PR lift 1.18x.
- Static + labs + vitals raw/Platt: PR-AUC 0.071, PR lift 1.32x.
- Static + labs + vitals + therapy raw/Platt: PR-AUC 0.071, PR lift 1.34x.

Best top-risk review yield:

- Static + labs top 5%: PPV 7.0%, 20 true-positive rows, 21.4% episode recall.
- Static + labs + vitals top 5%: PPV 9.1%, 26 true-positive rows, 27.1% episode recall.
- Static + labs + vitals + therapy top 5%: PPV 9.5%, 27 true-positive rows, 28.6% episode recall.
- Static + labs + vitals + therapy top 100 rows: PPV 8.0%, 8 true-positive rows.
- Isotonic static + labs + vitals + therapy top 1% had PPV 14.0%, but only captured 8 true-positive rows and should be interpreted cautiously.

## Interpretation

Run 20 shows that richer dynamic features help, but only modestly. The validation PR-AUC improved from about 0.063 in Run 19 to about 0.071 with vitals and therapy, and top-5% PPV improved from 7.0% to 9.5%.

This suggests the feature layer was part of the problem, but not the whole problem. Under the stricter v0.5 temporal protocol, the current seven-day daily landmark target remains difficult.

## Decision

Do not open the 2020-2022 lockbox yet.

Run 20 is directionally encouraging, but validation-era ranking remains below the level needed for a final temporal test. The next step should focus on improving the prediction frame or feature design while preserving the temporal development protocol.

## Recommended Next Step

Candidate Run 21 directions:

1. Test alternate target framing inside v0.5 development years only, such as 72-hour or event-adjacent daily review windows.
2. Add care-process/documentation features from caregiver-linked charting, line-care documentation, and measurement density.
3. Try episode-level aggregation of daily risk into a "top-risk review list" model rather than judging only row-level PR-AUC.
4. Evaluate whether the seven-day horizon is too diffuse for the available MIMIC signal.

## Main Artifacts

- `data/v0_5/v0_5_run20_dynamic_enriched_features.csv`
- `data/v0_5/v0_5_run20_vitals_long.pkl`
- `data/v0_5/v0_5_run20_antibiotics_long.pkl`
- `data/v0_5/v0_5_run20_vasopressors_long.pkl`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_dynamic_feature_audit.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_dynamic_model_comparison.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_validation_topk_review.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_validation_threshold_policy.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_validation_first_alert_policy.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_validation_calibration_deciles.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_xgboost_feature_importance.csv`
- `Outputs/Run 20 (v0.5 Dynamic Feature Enrichment)/v0_5_run20_validation_predictions.csv`


