# Run 18 - v0.5 Development Models

## Purpose

Run 18 is the first constrained v0.5 modeling experiment after the catheter-episode redesign and daily landmark frame. It intentionally uses only the development era for modeling:

- Train: 2008-2016 anchor-year groups.
- Validation: 2017-2019 anchor-year group.
- Temporal lockbox held out: 2020-2022, audited only with no predictions generated.

The purpose is to test whether a fixed daily seven-day prediction task can produce useful risk stratification once the cohort avoids longest-line selection, near-event static reference leakage, and repeated tuning on the old test split.

## Data and Feature Design

Primary modeling frame:

- Included rows with either a future strict CVC-associated BSI proxy within seven days or complete observed seven-day follow-up.
- Excluded rows censored before seven days without an event from model training and validation.

Feature sets:

- `static_context`: landmark timing plus demographic/admission context.
- `static_context_labs_48h`: static context plus raw MIMIC-IV `labevents` aggregated over the prior 48 hours.

Raw lab extraction was kept intentionally faithful to the v0.5 cohort. The script scans raw `labevents.csv.gz`, filters to the v0.5 cohort and target lab itemids, caches the filtered table at `data/v0_5/v0_5_run18_labs_long.pkl`, then aggregates by admission/window to avoid a large all-landmark merge.

## Modeling

Constrained model comparison:

- Logistic regression.
- XGBoost.

No lockbox model predictions are generated in this run.

## Key Output Summary

Feature construction:

- Landmark rows: 64,752.
- Filtered source lab rows: 1,335,467.
- Windowed 48-hour lab rows: 959,991.
- Landmarks with any 48-hour lab: 64,192.
- Feature matrix shape: 64,752 rows x 73 columns.

Development modeling frame:

- Train: 13,965 rows, 1,110 positives, 7.95% prevalence.
- Validation: 5,694 rows, 304 positives, 5.34% prevalence.
- Lockbox held out: 5,580 rows; 2,262 primary-frame rows; 162 positive rows.

Validation performance:

- Logistic, static context: ROC-AUC 0.548, PR-AUC 0.064, PR lift 1.20x.
- XGBoost, static context: ROC-AUC 0.526, PR-AUC 0.069, PR lift 1.29x.
- Logistic, static + 48h labs: ROC-AUC 0.600, PR-AUC 0.098, PR lift 1.84x.
- XGBoost, static + 48h labs: ROC-AUC 0.596, PR-AUC 0.116, PR lift 2.16x.

Top-k review on validation:

- Best top 1% review policy: XGBoost static + labs, PPV 31.6%, recall 5.9%, about 2.2 false alerts per true positive.
- Best top 5% review policy: XGBoost static + labs, PPV 15.4%, recall 14.5%, about 5.5 false alerts per true positive.
- Best top 100 rows: XGBoost static + labs, PPV 25.0%, recall 8.2%, about 3.0 false alerts per true positive.

## Interpretation

Run 18 supports the idea that recent lab context adds signal to the v0.5 daily seven-day prediction task. The absolute PR-AUC remains modest, but the best model clears the prespecified "2x prevalence" signal threshold on validation while preserving the new lockbox discipline.

Calibration is not yet acceptable. Expected/observed ratios around 7 indicate predicted probabilities are too high, so the next development step should address calibration and review-policy framing before the temporal lockbox is touched.

## Main Artifacts

- `data/v0_5/v0_5_run18_development_features.csv`
- `data/v0_5/v0_5_run18_labs_long.pkl`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_development_model_comparison.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_topk_review_table.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_feature_audit.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_modeling_frame_audit.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_lockbox_holdout_audit.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_xgboost_feature_importance.csv`
- `Outputs/Run 18 (v0.5 Development Models)/v0_5_run18_development_predictions.csv`


