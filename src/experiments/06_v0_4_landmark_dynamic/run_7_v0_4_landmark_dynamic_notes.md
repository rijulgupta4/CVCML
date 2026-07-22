# Run 7 - v0.4 Landmark Dynamic Model

Purpose:
- Move beyond static retrospective reference-time modeling.
- Ask a prospective-style question at fixed catheter dwell landmarks: using only data available up to this time, will strict CLABSI occur later?

Landmark design:
- Landmark hours: 48, 72, 96, 120, 144, 168, and 240 hours after catheter placement.
- Lookback window: 48 hours before each landmark.
- Prediction horizon: future strict CLABSI within the next 168 hours or before catheter removal, whichever comes first.
- Rows after a strict CLABSI event are excluded because the patient is no longer at risk for a future first event.

Feature design:
- Static demographics and catheter/admission metadata.
- Landmark time and dwell-at-landmark exposure.
- Lab mean, last value, trend, count, and recency within the 48-hour lookback window.
- Missing labs are left as missing for XGBoost and accompanied by measured indicators.

Run order:
1. `SRC\06_v0_4_landmark_dynamic\Feature Engineering 03 v0.4 Landmark Dynamic.py`
2. `SRC\06_v0_4_landmark_dynamic\Modeling 05 v0.4 Landmark Dynamic.py`

Expected data outputs:
- `data\v0_4\clabsi_landmark_features_v0_4.csv`
- `data\v0_4\v0_4_landmark_feature_audit.csv`
- `data\v0_4\v0_4_landmark_row_summary.csv`

Expected model outputs:
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_model_comparison.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_threshold_table.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_performance_by_landmark.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_stay_level_summary.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_top_100_rows.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\v0_4_landmark_dynamic_calibration_deciles.csv`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\plots\v0_4_landmark_dynamic_pr_curves.png`
- `Outputs\Run 7 (v0.4 Landmark Dynamic Model)\plots\v0_4_landmark_dynamic_best_model_shap.png`

Interpretation focus:
- Does prospective-style landmark prediction retain useful ranking signal?
- Which landmark times are most/least predictive?
- Is dynamic alert burden lower or more clinically actionable than the static score?
- Does model logic shift away from outcome-relative timing features and toward available labs/context?
- Does calibration improve or remain an issue?

