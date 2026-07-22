# Run 17 - v0.5 Daily Landmark Frame

## Purpose

Run 17 builds the prediction-row frame for v0.5. It does not fit a model and does not extract new clinical features.

This run answers:

- Can the v0.5 catheter-exposure episodes be converted into daily prediction rows?
- Does the 7-day future strict CVC-associated BSI proxy target behave sensibly?
- Are censoring and competing events explicitly represented?
- Is the temporal lockbox preserved before modeling?

## Inputs

Primary input:

`data/v0_5/v0_5_catheter_exposure_periods.csv`

This file was produced by Run 16.

## Landmark Rules

- Include only exposure periods with at least 48 observed exposure hours.
- First landmark occurs at 48 hours after exposure start.
- Subsequent landmarks occur every 24 hours.
- Stop generating landmarks once the strict proxy event has occurred or observed exposure has ended.
- Use one prediction horizon: 168 hours / 7 days.
- Primary target: `future_strict_cvc_bsi_proxy_7d`.
- Sensitivity target: `future_broad_cvc_bsi_proxy_7d`.
- Preserve the 2020-2022 anchor-year group as `temporal_lockbox`.

## Main Results

- Eligible exposure periods: 11,602
- Episodes represented with landmarks: 11,591
- Landmark rows: 64,752
- Median landmarks per episode: 3
- Maximum landmarks per episode: 155
- Strict future-positive rows: 1,576
- Strict future-positive row rate: 2.43%
- Broad future-positive rows: 2,261
- Broad future-positive row rate: 3.49%
- Episodes with strict future-positive rows: 371
- Episodes with broad future-positive rows: 528

## Coverage Audit

- Eligible episodes without landmarks: 11
- Duplicate landmark IDs: 0
- Duplicate episode/landmark-hour rows: 0
- Rows after strict event: 0
- Rows after observed exposure end: 0
- Negative hours-to-observed-end rows: 0

The 11 eligible episodes without landmarks likely have observed exposure at or just above 48 hours but no at-risk time strictly after the 48-hour landmark.

## Temporal Split

Development rows:

- 2008-2010: 18,954 rows
- 2011-2013: 11,961 rows
- 2014-2016: 13,741 rows
- 2017-2019: 14,516 rows

Temporal lockbox rows:

- 2020-2022: 5,580 rows
- 810 episodes
- 783 stays
- 691 patients
- 162 strict future-positive rows
- Strict future-positive row rate: 2.90%

The temporal lockbox should not be used for model selection, feature selection, calibration selection, or threshold selection.

## Outputs

Data folder:

`data/v0_5`

Key files:

- `v0_5_daily_landmarks.csv`
- `v0_5_daily_landmark_audit.csv`
- `v0_5_daily_landmark_by_day_audit.csv`
- `v0_5_daily_landmark_outcome_status_audit.csv`
- `v0_5_daily_landmark_temporal_split_audit.csv`
- `v0_5_daily_landmark_episode_counts.csv`
- `v0_5_daily_landmark_coverage_audit.csv`

Output folder:

`Outputs/Run 17 (v0.5 Daily Landmark Frame)`

## Next Step

Run 18 should be the first constrained v0.5 model run.

Recommended modeling discipline:

- Use development rows only.
- Do not use the 2020-2022 temporal lockbox.
- Compare logistic regression versus one tree model.
- Start with simple feature sets.
- Report PR-AUC lift over prevalence, ROC-AUC, Brier Skill Score, calibration intercept/slope, E:O ratio, and top-k daily review burden.

