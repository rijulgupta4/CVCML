# Run 9 - v0.4C Dynamic Proxy Robustness

## Purpose

Run 8 showed that adding vitals improved the first dynamic landmark model, but the best model still relied heavily on monitoring/documentation intensity and performed best at the 168-hour horizon. Run 9 is a diagnostic modeling-only run that asks whether the dynamic model still has useful signal after removing measurement-intensity proxy features.

This run does not perform new raw data extraction. It uses the Run 8 feature matrix:

- `data/v0_4b/clabsi_landmark_features_v0_4b.csv`

## What Run 9 Tests

### 1. Measurement-Intensity Proxy Removal

Run 8 SHAP and feature importance showed high importance for features such as:

- measurement counts
- hours since last measurement
- measured/not measured indicators

These can be clinically meaningful, but they may also act as proxies for acuity, staffing, monitoring intensity, or documentation workflow. Run 9 creates value-only feature sets that remove:

- `_count_`
- `_hours_since_last_`
- `_measured_`

This tests whether actual physiologic values carry signal without care-intensity proxies.

### 2. Site Documentation Ablation

`site_known` has repeatedly been predictive, but it likely reflects documentation/workflow and patient context rather than direct biology. Run 9 includes no-site variants to test whether performance persists when this feature is removed.

### 3. Gray-Zone Label Frame

For short-horizon prediction, a row can be labeled negative even if the patient develops CLABSI soon after the horizon. For example, under a 48-hour target, a row 72 hours before CLABSI is technically negative, but clinically it may already be in a pre-event state.

Run 9 therefore evaluates two label frames:

- `standard`: same labels as Run 8.
- `gray_zone_excluded`: removes rows with future CLABSI outside the current horizon but within 168 hours.

This asks whether short-horizon performance improves when ambiguous pre-event rows are not forced into the negative class.

## Horizons Tested

- 48 hours
- 72 hours
- 168 hours

The 24-hour horizon is omitted because Run 8 showed extremely sparse positives and unstable performance.

## Feature Sets

- full values no intensity
- no site no intensity
- routine labs + vitals values
- routine labs + vitals values no site
- vitals values only
- labs values only
- context only

## Outputs to Inspect

- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_proxy_feature_audit.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_dynamic_proxy_model_comparison.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_dynamic_proxy_best_by_frame_horizon.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_dynamic_proxy_threshold_table.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_dynamic_proxy_stay_level_summary.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/v0_4c_dynamic_proxy_calibration_deciles.csv`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/plots/v0_4c_standard_best_pr_curves.png`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/plots/v0_4c_gray_zone_excluded_best_pr_curves.png`
- `Outputs/Run 9 (v0.4C Dynamic Proxy Robustness)/plots/v0_4c_dynamic_proxy_best_model_shap.png`

## What We Hope to Learn

Promising signs:

- Value-only vitals/labs retain meaningful discrimination.
- Removing `site_known` does not collapse performance.
- Gray-zone exclusion improves 48-hour or 72-hour performance.
- SHAP shifts from measurement frequency toward physiologic values such as temperature, heart rate, MAP, respiratory rate, platelets, WBC, creatinine, or lactate.

Concerning signs:

- Context-only remains competitive with physiology.
- Removing intensity features sharply reduces performance.
- Gray-zone exclusion does not improve short-horizon models.
- Alert burden remains excessive even after proxy removal.

## Interpretation Boundary

Run 9 is not meant to be the final dynamic model. It is an audit run to decide whether the next major dynamic improvement should focus on better physiologic modeling or on adding treatment/context features such as antibiotics, vasopressors, ventilation, and fluid balance.

