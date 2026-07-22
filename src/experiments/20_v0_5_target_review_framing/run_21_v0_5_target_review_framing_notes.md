# Run 21 - v0.5 Target / Review Framing

## Purpose

Run 21 asks whether the v0.5 model is being evaluated in the right clinical frame. Instead of adding another broad feature set, this run reuses the Run 20 enriched matrix and compares target horizons and review-list interpretations:

- 48-hour future strict CVC-associated BSI proxy
- 72-hour future strict CVC-associated BSI proxy
- 168-hour / 7-day future strict CVC-associated BSI proxy

This follows the lesson from earlier runs: use previous experiments to guide the next question, but only treat results as manuscript-relevant after re-evaluating them under the v0.5 catheter-episode and temporal-development protocol.

## Inputs

- Feature matrix: `data/v0_5/v0_5_run20_dynamic_enriched_features.csv`
- Development rows only: `split_role == development`
- Primary model frame: `run18_primary_model_frame == 1`
- Temporal split:
  - Train core: 2008-2013
  - Calibration: 2014-2016
  - Validation: 2017-2019
  - Lockbox: 2020-2022 held out, audit only

## Model Comparisons

Two feature sets were evaluated for each horizon:

1. `static_labs`
   - Static demographics/admission/context plus lab summaries.
   - Used as the v0.5 comparator.

2. `static_labs_vitals_therapy`
   - Static + labs + 24h/48h vitals + antibiotic/vasopressor context.
   - Used as the current enriched candidate from Run 20.

Each model was trained as XGBoost and evaluated raw plus Platt calibrated. The main interpretation should use Platt-calibrated validation performance.

## Validation Results

Platt-calibrated validation performance:

| Horizon | Feature set | Prevalence | ROC-AUC | PR-AUC | PR-AUC lift | E:O |
|---|---|---:|---:|---:|---:|---:|
| 48h | static_labs | 2.20% | 0.6065 | 0.0509 | 2.32x | 1.18 |
| 48h | static_labs_vitals_therapy | 2.20% | 0.6685 | 0.0651 | 2.97x | 1.19 |
| 72h | static_labs | 3.00% | 0.5867 | 0.0457 | 1.52x | 1.15 |
| 72h | static_labs_vitals_therapy | 3.00% | 0.6322 | 0.0571 | 1.90x | 1.18 |
| 168h | static_labs | 5.34% | 0.5588 | 0.0631 | 1.18x | 1.12 |
| 168h | static_labs_vitals_therapy | 5.34% | 0.5799 | 0.0714 | 1.34x | 1.09 |

## Review-List Interpretation

Top 5% row-level review policy:

| Horizon | Feature set | Rows reviewed | TP rows | PPV | Row recall | Episode recall | False reviews / TP |
|---|---|---:|---:|---:|---:|---:|---:|
| 48h | static_labs | 285 | 20 | 7.02% | 16.00% | 24.29% | 13.25 |
| 48h | static_labs_vitals_therapy | 285 | 23 | 8.07% | 18.40% | 25.71% | 11.39 |
| 72h | static_labs | 285 | 21 | 7.37% | 12.28% | 21.43% | 12.57 |
| 72h | static_labs_vitals_therapy | 285 | 21 | 7.37% | 12.28% | 21.43% | 12.57 |
| 168h | static_labs | 285 | 20 | 7.02% | 6.58% | 21.43% | 13.25 |
| 168h | static_labs_vitals_therapy | 285 | 27 | 9.47% | 8.88% | 28.57% | 9.56 |

## Interpretation

Run 21 clarifies that model usefulness depends on the intended clinical workflow:

- The 48-hour enriched model has the strongest discrimination and PR-AUC lift over baseline prevalence. This suggests vitals and therapy context help most when the question is short-term deterioration or near-event risk.
- The 168-hour enriched model has the best top-5% review-list PPV and episode capture. This better supports an infection-prevention surveillance list than a bedside alert.
- The 72-hour frame does not currently add a clear advantage.
- Dynamic enrichment consistently improves over static + labs, but the absolute PPV remains modest.
- The temporal lockbox remains closed. No Run 21 lockbox predictions were generated.

## Practical Conclusion

The most defensible use case is not high-frequency bedside alerting. The stronger framing is:

> A daily infection-prevention review list using an enriched dynamic model, with the 7-day horizon for surveillance and the 48-hour horizon as a near-term acuity sensitivity analysis.

## Outputs

Stored in `Outputs/Run 21 (v0.5 Target Review Framing)`:

- `v0_5_run21_target_framing_model_comparison.csv`
- `v0_5_run21_target_framing_topk_row_review.csv`
- `v0_5_run21_target_framing_topk_episode_review.csv`
- `v0_5_run21_target_framing_threshold_policy.csv`
- `v0_5_run21_target_framing_first_alert_policy.csv`
- `v0_5_run21_target_framing_calibration_deciles.csv`
- `v0_5_run21_target_framing_feature_importance.csv`
- `v0_5_run21_target_framing_validation_predictions.csv`
- `plots/v0_5_run21_pr_auc_by_horizon.png`
- `plots/v0_5_run21_pr_auc_lift_by_horizon.png`
- `plots/v0_5_run21_enriched_pr_curves_by_horizon.png`
- `plots/v0_5_run21_top5_ppv_by_horizon.png`

## Next Step

Run 22 should avoid broad retesting. The best next experiment is an episode-level operating policy run:

- choose one or two candidate horizons,
- evaluate daily or per-episode review-list rules,
- reduce repeat alerts,
- report patient/episode-level workload and yield,
- and decide whether the 7-day surveillance framing or 48-hour near-term risk framing should become the final v0.5 candidate.

