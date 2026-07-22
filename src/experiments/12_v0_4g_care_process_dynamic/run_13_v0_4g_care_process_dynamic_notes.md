# Run 13 - v0.4G Care-Process Dynamic Model

## Purpose

Run 13 tests whether care-process exposure adds CLABSI-specific dynamic signal beyond physiology, labs, vitals, and cleaned therapy context.

Run 12 showed that calibration and alert-policy framing make the dynamic branch more operationally honest, but the model is still not ready as a bedside alert. Run 13 therefore moves from alert-policy characterization to a true feature-improvement question:

> Do caregiver exposure, invasive-line maintenance documentation, and fluid-balance context improve 7-day CLABSI risk prediction?

## Clinical Rationale

CLABSI is not only a marker of patient instability. It is directly related to catheter presence, catheter handling, dressing integrity, line access, and care processes around the line. A model that only learns lactate, platelets, vasopressors, and antibiotics may be detecting sepsis-like deterioration rather than catheter-associated infection risk. Care-process variables are therefore more mechanistically aligned with the outcome.

## New Data Sources

1. `chartevents`
   - Uses `caregiver_id` from selected bedside documentation rows.
   - Includes routine vitals and invasive-line documentation itemids.
   - Produces caregiver exposure and continuity features.

2. `datetimeevents`
   - Pulls invasive-line maintenance documentation from `Access Lines - Invasive`.
   - Includes dressing change, cap change, tubing change, site assessment, biopatch, securement, insertion-date documentation, and change-over-wire documentation.
   - Excludes culture/tip-culture/discontinued/removal labels to reduce diagnostic leakage.

3. `inputevents` and `outputevents`
   - Adds fluid input, output, urine output, and net fluid balance context.
   - This is secondary severity/context information, not the primary novelty.

## Feature Families

- `caregiver_*`: event counts, unique caregiver counts, handoff count, dominant caregiver fraction.
- `linecare_*`: any line-care documentation, event counts, event-type counts, unique event types, caregiver counts, hours since last line-care event.
- `fluid_*`: input volume, output volume, urine output, net fluid balance, event counts, caregiver counts.

Windows are 24h, 48h, and 72h before each landmark.

## Run Order

1. `Data Extraction 01 v0.4G Care Process.py`
2. `Feature Engineering 03 v0.4G Care Process.py`
3. `Modeling 10 v0.4G Care Process Dynamic.py`

## Expected Outputs

Data folder:

- `data/v0_4g/clabsi_landmark_features_v0_4g.csv`
- `data/v0_4g/v0_4g_care_process_extraction_audit.csv`
- `data/v0_4g/v0_4g_care_process_extraction_counts.csv`
- `data/v0_4g/v0_4g_care_process_feature_audit.csv`
- `data/v0_4g/v0_4g_care_process_feature_missingness.csv`
- `data/v0_4g/v0_4g_landmark_row_summary.csv`

Output folder:

- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_model_comparison.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_best_by_frame_horizon.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_threshold_table.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_alert_policy_summary.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_calibration_deciles.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/v0_4g_care_process_dynamic_top_risk_review_table.csv`
- `Outputs/Run 13 (v0.4G Care Process Dynamic)/plots/v0_4g_care_process_dynamic_best_model_shap.png`

## How To Interpret

The most important comparison is:

- `v0.4G baseline full cleaned therapy`
- versus `v0.4G full cleaned therapy plus care process`
- versus `v0.4G full cleaned therapy plus care process no site`

If care-process features improve PR-AUC, top-risk PPV, or false-alert burden at the 168h horizon, the project has moved toward a more mechanistically CLABSI-specific dynamic model.

If care-process features do not improve performance, inspect feature sparsity first. A null result may mean MIMIC line-care documentation is too sparse/noisy, not that the clinical idea is weak.

## Primary Success Criteria

Run 13 is promising if it does at least one of the following:

- improves 168h PR-AUC versus Run 12 or Run 11;
- improves top-1% or top-5% stay-level PPV;
- reduces false alerts per true positive under alert-cap thresholds;
- introduces clinically interpretable SHAP features related to line care or caregiver continuity.


