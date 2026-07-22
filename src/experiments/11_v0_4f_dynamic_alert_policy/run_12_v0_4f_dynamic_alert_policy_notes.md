# Run 12 - v0.4F Dynamic Alert Policy

## Purpose

Run 11 improved dynamic PR-AUC after cleaning therapy-context features, but the model still behaved poorly as a direct clinical alert: PPV was low, calibration was poor, and alert burden remained high. Run 12 asks a more clinical question:

Can the dynamic model become useful when evaluated as an alert/risk-ranking policy rather than only as a repeated-row classifier?

## What Changed

- Reuses `data/v0_4e/clabsi_landmark_features_v0_4e.csv`; no new raw MIMIC extraction is required.
- Adds derived physiology burden features in memory:
  - fever and hypothermia excess
  - temperature range
  - tachycardia and tachypnea excess
  - MAP/SBP hypotension depth
  - lactate excess and rise
  - platelet drop and thrombocytopenia depth
  - WBC high/low abnormality depth
  - composite instability rank
- Compares therapy-inclusive and therapy-excluded models.
- Tests site-documentation ablations.
- Adds calibration variants:
  - raw XGBoost probability
  - isotonic calibration
  - Platt/logistic calibration
- Evaluates alert policies:
  - row-level recall threshold
  - validation alert-budget thresholds
  - first alert per stay
  - maximum-risk one alert per stay
  - 48h and 72h cooldown alert policies
  - top-risk review at 1%, 2%, 5%, and 10%

## Run Order

1. `Modeling 09 v0.4F Dynamic Alert Policy.py`

This is modeling-only and should be faster than runs that reload raw `chartevents` or `inputevents`.

## Expected Outputs

- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_model_comparison.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_best_by_frame_horizon.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_threshold_table.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_alert_policy_summary.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_calibration_deciles.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/v0_4f_dynamic_alert_policy_top_risk_review_table.csv`
- `Outputs/Run 12 (v0.4F Dynamic Alert Policy)/plots/`

## How To Interpret

- If therapy-excluded physiology models perform close to therapy-inclusive models, the dynamic signal is more likely to represent patient trajectory rather than clinician suspicion.
- If therapy-inclusive models dominate, therapy context is useful but should be framed carefully as treatment-context/risk-state signal.
- If calibrated probabilities remain overconfident, use the model for ranking and triage rather than absolute risk.
- If first-alert or cooldown policies substantially reduce false alerts per true positive, the dynamic model becomes more defensible for clinical workflow discussion.
- If top-risk review PPV improves meaningfully, the model may be better framed as a daily high-risk list rather than a bedside interruptive alert.

