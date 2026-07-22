# Run 23 - v0.5 Label Sensitivity Modeling

## Purpose

Run 23 tests whether the Run 22 secondary-source label refinement improves model behavior without changing features, model type, or the temporal development protocol.

This is a label-sensitivity run, not a feature-engineering run.

## Inputs

- Features: `data/v0_5/v0_5_run20_dynamic_enriched_features.csv`
- Source-screened labels: `data/v0_5/v0_5_run22_source_screened_daily_landmarks.csv`
- Feature set: `static_labs_vitals_therapy`
- Model: XGBoost with Platt calibration
- Development protocol:
  - Train core: 2008-2013
  - Calibration: 2014-2016
  - Validation: 2017-2019
  - Temporal lockbox: 2020-2022 held out, no predictions

## Target Definitions

Three 7-day future-positive targets were compared:

1. `original_strict`
   - Current v0.5 strict CVC-associated BSI proxy.

2. `primary_or_uncertain`
   - Source-screened candidate primary target.
   - Keeps strict positives unless they have concordant non-blood source-culture evidence.
   - Intended to remove the clearest likely secondary BSIs while preserving uncertain ICU cases.

3. `primary_likely`
   - High-specificity source-screened subset.
   - No nearby source-culture evidence and no source-related ICD evidence.
   - Included only as a sensitivity analysis because it is very sparse.

## Target Prevalence

Development-frame target counts:

| Target | Positive rows | Prevalence | Positive episodes |
|---|---:|---:|---:|
| Original strict | 1,414 | 7.19% | 338 |
| Primary-or-uncertain | 1,151 | 5.85% | 265 |
| Primary-likely | 115 | 0.59% | 26 |

Validation target counts:

| Target | Positive rows | Prevalence |
|---|---:|---:|
| Original strict | 304 | 5.34% |
| Primary-or-uncertain | 242 | 4.25% |
| Primary-likely | 29 | 0.51% |

## Validation Results

Platt-calibrated validation performance:

| Target | ROC-AUC | PR-AUC | PR-AUC lift | Brier skill | E:O |
|---|---:|---:|---:|---:|---:|
| Original strict | 0.580 | 0.071 | 1.34x | -0.005 | 1.09 |
| Primary-or-uncertain | 0.619 | 0.069 | 1.63x | 0.007 | 1.15 |
| Primary-likely | 0.614 | 0.0066 | 1.30x | -0.003 | 0.70 |

## Review-List Yield

Top 5% row-level review policy:

| Target | Rows reviewed | TP rows | PPV | Row recall | Episode recall | False reviews / TP |
|---|---:|---:|---:|---:|---:|---:|
| Original strict | 285 | 27 | 9.47% | 8.88% | 28.57% | 9.56 |
| Primary-or-uncertain | 285 | 29 | 10.18% | 11.98% | 29.63% | 8.83 |
| Primary-likely | 285 | 1 | 0.35% | 3.45% | 16.67% | 284.00 |

## Interpretation

The source-screened `primary_or_uncertain` label improves the model's relative usefulness:

- ROC-AUC increased from 0.580 to 0.619.
- PR-AUC lift increased from 1.34x to 1.63x.
- Top-5% PPV increased from 9.47% to 10.18%.
- Row recall and episode recall also improved modestly.
- Brier skill became slightly positive.

The absolute PR-AUC is similar because the source-screened label has lower prevalence, but the lift and review yield are better. This suggests that secondary-source label refinement is helping the model focus on a more coherent target.

The `primary_likely` label should not become the primary modeling target. It has only 29 validation-positive rows and produces poor review-list yield despite superficially acceptable ROC-AUC. It is best reserved as a high-specificity sensitivity analysis.

## Practical Conclusion

The best current v0.5 candidate outcome is:

> `future_strict_primary_or_uncertain_cvc_bsi_proxy_7d`

This is more defensible than the original strict proxy because it excludes concordant secondary-source culture evidence, but it is not so conservative that it destroys modelability.

## Outputs

Stored in `Outputs/Run 23 (v0.5 Label Sensitivity Modeling)`:

- `v0_5_run23_label_sensitivity_model_comparison.csv`
- `v0_5_run23_label_sensitivity_target_audit.csv`
- `v0_5_run23_label_sensitivity_split_audit.csv`
- `v0_5_run23_label_sensitivity_topk_row_review.csv`
- `v0_5_run23_label_sensitivity_topk_episode_review.csv`
- `v0_5_run23_label_sensitivity_threshold_policy.csv`
- `v0_5_run23_label_sensitivity_first_alert_policy.csv`
- `v0_5_run23_label_sensitivity_calibration_deciles.csv`
- `v0_5_run23_label_sensitivity_feature_importance.csv`
- `v0_5_run23_label_sensitivity_validation_predictions.csv`
- `plots/v0_5_run23_pr_auc_by_label.png`
- `plots/v0_5_run23_pr_auc_lift_by_label.png`
- `plots/v0_5_run23_top5_ppv_by_label.png`

## Next Step

Run 24 should freeze the candidate label as `primary_or_uncertain` and return to operating-policy characterization:

- compare 48h versus 168h framing under the refined label if needed,
- evaluate episode-level first-review policies,
- reduce repeat reviews,
- quantify workload and PPV,
- and decide whether a final pre-lockbox candidate can be specified.

