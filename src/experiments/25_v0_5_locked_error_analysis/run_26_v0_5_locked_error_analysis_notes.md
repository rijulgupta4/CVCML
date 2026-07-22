# Run 26 - v0.5 Locked Error Analysis

## Purpose

Run 26 characterizes the locked Run 25 temporal evaluation. It does not refit, recalibrate, tune thresholds, or revise labels. It joins the frozen lockbox predictions to source-screen labels, episode context, culture detail, and selected feature values.

## Review-List Error Groups

- Top 5% rows reviewed 114 rows: 17 true-positive rows and 97 false-positive rows.
- Top 10% rows reviewed 227 rows: 31 true-positive rows, 196 false-positive rows, and left 103 positive rows unreviewed.
- Top 150 episode review captured 69.2% of positive episodes with reviewed-episode PPV 12.0%.

## Feature Contrast Signal

Largest standardized contrasts between high-risk negatives and missed positive rows:
- platelets_lab_count: FP - FN standardized mean difference 0.74
- wbc_lab_count: FP - FN standardized mean difference 0.70
- lactate_lab_count: FP - FN standardized mean difference 0.69
- platelets_mean: FP - FN standardized mean difference -0.53
- platelets_last: FP - FN standardized mean difference -0.49
- vital_respiratory_rate_max_24h: FP - FN standardized mean difference -0.37
- abx_antibiotic_any_starts_24h: FP - FN standardized mean difference -0.36
- abx_broad_antibiotic_starts_24h: FP - FN standardized mean difference -0.29

## Interpretation

- This run should be used to decide whether high-risk false positives are clinically plausible review candidates or mostly noise.
- If high-risk negatives show infection-like physiology, therapy exposure, source-culture context, or care intensity, the modest PPV is less damaging because the review list may still enrich for clinically concerning CVC episodes.
- Missed positives define the next label/feature-improvement target, but the lockbox itself should not be used for tuning.
