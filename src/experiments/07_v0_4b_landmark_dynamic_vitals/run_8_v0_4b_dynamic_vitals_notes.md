# Run 8 - v0.4B Dynamic Vitals + Horizon Sensitivity

## Purpose

Run 7 showed that a landmark dynamic model using mostly rolling labs did not improve on the leakage-audited static model. Run 8 tests whether the dynamic framing becomes more clinically meaningful when it includes bedside physiology and shorter prediction horizons.

This run is motivated by the comparison to the Frontiers 2025 CLABSI paper. That paper used richer physiologic and treatment-context features, but its reported performance may be inflated by leakage-prone variables such as ICU length of stay. Run 8 closes part of the legitimate feature gap while preserving this project's leakage controls.

## What changed from Run 7

- Adds a dedicated extraction step for target labs and vitals.
- Adds prospective vital-sign features from `chartevents`.
- Keeps patient-level splitting.
- Keeps strict organism labeling from v0.3a.
- Avoids outcome-dependent features such as total ICU length of stay, total catheter dwell, future culture information, and audit-only flags.
- Tests multiple prediction horizons from the same landmark feature matrix:
  - 24 hours
  - 48 hours
  - 72 hours
  - 168 hours

## Feature logic

For each landmark row, the feature engineering script summarizes only measurements charted before the landmark time.

Vitals:

- temperature
- heart rate
- respiratory rate
- SpO2
- systolic blood pressure
- diastolic blood pressure
- MAP

For each vital and lab, the script computes 24-hour and 48-hour summaries:

- mean
- min
- max
- last value
- trend
- count
- hours since last measurement
- measured indicator

It also adds simple clinical abnormality flags:

- fever in last 24 hours
- hypothermia in last 24 hours
- tachycardia in last 24 hours
- hypotension by MAP in last 24 hours
- tachypnea in last 24 hours

## Outputs to inspect

After running the data extraction script:

- `data/v0_4b/v0_4b_labs_long.pkl`
- `data/v0_4b/v0_4b_vitals_long.pkl`
- `data/v0_4b/v0_4b_vitals_extraction_audit.csv`
- `data/v0_4b/v0_4b_vitals_extraction_counts.csv`

After running the feature engineering script:

- `data/v0_4b/clabsi_landmark_features_v0_4b.csv`
- `data/v0_4b/v0_4b_landmark_feature_audit.csv`
- `data/v0_4b/v0_4b_landmark_row_summary.csv`
- `data/v0_4b/v0_4b_dynamic_feature_missingness.csv`

The `.pkl` files are intermediate extracted long tables. They are not the final model matrix; they exist so feature engineering reruns do not need to rescan the full raw MIMIC `labevents` and `chartevents` tables after extraction succeeds once.

After running the modeling script:

- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_model_comparison.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_best_by_horizon.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_threshold_table.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_performance_by_landmark.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_stay_level_summary.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/v0_4b_dynamic_vitals_calibration_deciles.csv`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/plots/v0_4b_dynamic_vitals_best_pr_curves.png`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/plots/v0_4b_dynamic_vitals_performance_by_horizon.png`
- `Outputs/Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)/plots/v0_4b_dynamic_vitals_best_model_shap.png`

## What we are hoping to learn

The main question is not just whether Run 8 has a higher AUC. The better question is whether the dynamic model becomes more clinically coherent when it predicts closer events.

Promising signs:

- 24-hour, 48-hour, or 72-hour models outperform the 168-hour model on PR-AUC.
- Vitals or routine labs plus vitals outperform static-only features.
- SHAP highlights recent temperature, heart rate, MAP, respiratory rate, WBC, platelets, or creatinine changes instead of mostly administrative context.
- Alert burden improves at reasonable thresholds.

Concerning signs:

- All horizons stay close to the base rate.
- Performance remains driven mostly by `site_known`, age, or catheter type.
- The model requires excessive alerts per true positive.
- Calibration remains severely overconfident.

