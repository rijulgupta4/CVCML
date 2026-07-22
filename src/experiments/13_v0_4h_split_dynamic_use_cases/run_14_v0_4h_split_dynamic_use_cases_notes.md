# Run 14 / v0.4H - Split Dynamic Use Cases

## Purpose

Run 14 is a modeling-only refinement that reuses the v0.4G landmark feature matrix. It does not require new extraction or feature engineering.

The goal is to stop treating the dynamic model as one generic classifier and instead evaluate two clinically different jobs:

1. `168h_surveillance_review`
   - Horizon: 168 hours / 7 days
   - Label frame: gray-zone excluded
   - Clinical role: infection-prevention surveillance, rounding list review, and literature comparison.
   - Recall target for thresholding: 0.70, because this use case should avoid unmanageable alert burden.

2. `72h_near_term_workflow`
   - Horizon: 72 hours
   - Label frame: gray-zone excluded
   - Clinical role: near-term workflow-aware monitoring using physiology plus care-process intensity.
   - Recall target for thresholding: 0.80, because this branch is closer to active monitoring.

## What changed from Run 13

- Reuses `data/v0_4g/clabsi_landmark_features_v0_4g.csv`.
- Limits modeling to the two clinically motivated branches.
- Keeps Run 13 alert-policy, calibration, top-risk, and SHAP machinery.
- Adds validation-based selection summaries:
  - `best_validation_calibration`
  - `best_validation_ranking`
  - `best_validation_ppv_at_recall`

This makes the model selection more honest and more clinically interpretable than choosing a single global winner from all horizons.

## Run order

Run only:

`13_v0_4h_split_dynamic_use_cases/Modeling 11 v0.4H Split Dynamic Use Cases.py`

## Expected outputs

Main CSVs:

- `v0_4h_split_dynamic_model_comparison.csv`
- `v0_4h_split_dynamic_use_case_selection_summary.csv`
- `v0_4h_split_dynamic_threshold_table.csv`
- `v0_4h_split_dynamic_alert_policy_summary.csv`
- `v0_4h_split_dynamic_top_risk_review_table.csv`
- `v0_4h_split_dynamic_calibration_deciles.csv`
- `v0_4h_split_dynamic_feature_audit.csv`

Plots:

- `plots/v0_4h_split_dynamic_selected_model_performance.png`
- `plots/v0_4h_split_dynamic_selected_alert_burden.png`
- `plots/v0_4h_split_dynamic_best_ranking_model_shap.png`

## How to interpret

The 168h branch should be judged mainly as a 7-day risk-review model: discrimination, calibration, top-risk PPV, and one-alert-per-stay burden matter more than immediate bedside alerting.

The 72h branch should be judged as a workflow-aware model: care-process features should be useful if they improve PPV or reduce false alerts at comparable recall.

