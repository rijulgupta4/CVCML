# Run 19 - v0.5 Calibration and Review Policy

## Purpose

Run 19 tests whether the best Run 18 v0.5 model family can be turned into a more honest calibrated risk score and operational review policy without touching the 2020-2022 temporal lockbox.

The run directly addresses the critique that the project needs a true development workflow:

- Train model parameters on earlier years.
- Calibrate probabilities on a later-but-still-development era.
- Evaluate model selection and policy choices on a validation era.
- Keep the final temporal lockbox untouched until the model and policy are frozen.

## Temporal Split

- Train core: 2008-2013.
- Calibration: 2014-2016.
- Validation: 2017-2019.
- Temporal lockbox: 2020-2022, audited only with no predictions generated.

This is deliberately not a holistic/random evaluation. The lockbox is meant to simulate the harder question: if the model were built on earlier hospital years, would it still work in a later hospital period?

## Methods

Run 19 reuses the Run 18 feature matrix:

- `data/v0_5/v0_5_run18_development_features.csv`

Model:

- XGBoost with static context plus 48-hour lab features.

Calibration methods:

- Raw XGBoost probabilities.
- Platt calibration fit on 2014-2016.
- Isotonic calibration fit on 2014-2016.

Outputs:

- Validation discrimination and calibration summaries.
- Top-risk review policies.
- Threshold policies.
- First-alert-per-episode policies.
- Validation predictions.
- Lockbox audit only.

Plot generation was disabled in the local execution environment because Matplotlib exited without a Python traceback. The CSV tables were produced successfully and can be plotted separately.

## Key Output Summary

Development split audit:

- Train core: 9,228 rows, 837 positives, 9.07% prevalence.
- Calibration: 4,737 rows, 273 positives, 5.76% prevalence.
- Validation: 5,694 rows, 304 positives, 5.34% prevalence.
- Temporal lockbox audit only: 5,580 rows, 162 positives, 2.90% prevalence.

Validation performance:

- Raw XGBoost: ROC-AUC 0.559, PR-AUC 0.063, PR lift 1.18x, E:O 7.42.
- Platt-calibrated XGBoost: ROC-AUC 0.559, PR-AUC 0.063, PR lift 1.18x, E:O 1.12.
- Isotonic-calibrated XGBoost: ROC-AUC 0.556, PR-AUC 0.063, PR lift 1.18x, E:O 1.14.

Validation top-risk review:

- Raw/Platt top 5%: 285 rows reviewed, 20 true-positive rows, PPV 7.0%, row recall 6.6%, episode recall 21.4%.
- Isotonic top 5%: 285 rows reviewed, 11 true-positive rows, PPV 3.9%, row recall 3.6%, episode recall 11.4%.
- Raw/Platt top 1% and top 100 rows captured no true-positive rows on validation.

## Interpretation

Run 19 improved probability totals but weakened the apparent model utility compared with Run 18. This likely reflects a combination of:

- Reduced training data after reserving 2014-2016 for calibration.
- Temporal drift between early development years and 2017-2019 validation.
- A very difficult seven-day proxy prediction task under the stricter v0.5 cohort design.

The key methodological gain is that Run 19 gives a much cleaner development protocol. The key modeling warning is that the current v0.5 static-plus-lab XGBoost is not strong enough, under this stricter temporal workflow, to justify lockbox evaluation yet.

## Recommended Next Step

Before opening the 2020-2022 lockbox:

1. Keep the Run 19 temporal development structure.
2. Improve the feature layer or model training strategy on development years only.
3. Retest validation-era ranking and calibration.
4. Freeze the candidate only after validation performance returns to a clinically meaningful threshold.

Likely next experiments:

- Refit on 2008-2016 after selecting calibration approach, then validate only if using a nested calibration strategy.
- Add vitals/therapy/care-process features into the v0.5 episode/landmark frame.
- Evaluate whether a 168-hour daily review model remains too sparse and whether a narrower landmark target is more stable.

## Main Artifacts

- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_calibration_model_comparison.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_validation_topk_review.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_validation_threshold_policy.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_validation_first_alert_policy.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_validation_calibration_deciles.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_development_split_audit.csv`
- `Outputs/Run 19 (v0.5 Calibration Review Policy)/v0_5_run19_validation_predictions.csv`


