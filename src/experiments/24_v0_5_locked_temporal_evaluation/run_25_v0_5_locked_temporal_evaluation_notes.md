# Run 25 - v0.5 Locked Temporal Evaluation

## Purpose

Run 25 opens the 2020-2022 temporal lockbox once for the frozen v0.5 candidate. It loads the saved Run 23 XGBoost model and Platt calibrator, scores eligible lockbox landmark rows, and reports the same discrimination, calibration, and review-list metrics used during development.

## Frozen Specification

- Label: `future_strict_primary_or_uncertain_cvc_bsi_proxy_7d` (`primary_or_uncertain`)
- Horizon: 7 days / 168 hours
- Feature set: static + labs + vitals + therapy context
- Model: Run 23 XGBoost, no refitting
- Calibration: Run 23 Platt calibrator, no recalibration
- Use case: daily infection-prevention review list for active CVC episodes

## Locked Test Performance

- Rows: 2,262; positive rows: 134; prevalence: 5.9%
- Episodes: 270; positive episodes: 26
- ROC-AUC: 0.623
- PR-AUC: 0.110
- PR-AUC lift over prevalence: 1.85x
- Brier Skill Score: 0.013
- Expected:Observed ratio: 0.89

## Review-List Policies

- Top 5% rows: 114 reviews, PPV 14.9%, row recall 12.7%, episode recall 30.8%, false reviews/TP 5.71.
- Top 10% rows: 227 reviews, PPV 13.7%, row recall 23.1%, episode recall 50.0%, false reviews/TP 6.32.
- Top 150 episodes: 150 reviews, PPV 10.0%, episode recall 57.7%, false reviews/TP 9.00.
- Top 250 episodes: 250 reviews, PPV 8.4%, episode recall 80.8%, false reviews/TP 10.90.

## Initial Interpretation Template

Use the lockbox result to decide whether the project is strong, modest-but-useful, or weak. Do not tune on these results. If performance is modest but top-k yield remains above prevalence, the defensible claim is review-list prioritization rather than bedside alerting.
