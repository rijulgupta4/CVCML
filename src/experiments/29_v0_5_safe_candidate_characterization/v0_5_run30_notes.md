# Run 30: Leakage-safe candidate characterization

## Decision question
How uncertain and heterogeneous is the frozen Run 29 leakage-safe candidate on the untouched 2017-2019 development-validation period, before external validation?

## Locked design
- Predictions: Run 29 `safe_exclude_early_positive`, Platt calibrated.
- Target: `future_strict_primary_or_uncertain_cvc_bsi_proxy_7d` (7-day strict primary-or-uncertain CVC-associated BSI proxy).
- Evaluation period: 2017-2019 validation only.
- No model refitting, retuning, threshold optimization, or 2020-2022 scoring.
- Uncertainty: 2,000 patient-clustered bootstrap replicates. All landmarks and episodes for a sampled patient move together.

## Overall validation performance
- ROC-AUC: 0.6122 (95% CI 0.5184-0.7032).
- PR-AUC: 0.0653 (95% CI 0.0407-0.1126).
- PR-AUC lift over 4.25% prevalence: 1.54x (95% CI 1.13-2.48x).
- Brier Skill Score: 0.0049 (95% CI -0.0186-0.0209).
- Calibration intercept: -0.712; slope: 0.807; E:O: 1.159.

## Episode review policy
- Top 10% review PPV: 15.4% (95% CI 7.8%-26.9%).
- Top 10% positive-episode recall: 18.5% (95% CI 10.6%-31.5%).
- False reviews per true positive at top 10%: 5.50 (95% CI 2.72-11.80).

## Subgroup interpretation
- Subgroups were prespecified from available demographic and episode context: sex, age, race, first ICU type, admission type, insurance, and catheter context.
- 7 subgroup levels met the descriptive `more_stable` rule (at least 20 positive patients and 100 patients).
- Cells with fewer than 10 positive patients retain descriptive point estimates, but their confidence intervals are suppressed and they are excluded from the forest plot. These are heterogeneity checks, not formal fairness claims or evidence of causal differences.
- No multiplicity-adjusted hypothesis testing was performed.

## Scope and limitations
- The confidence intervals measure sampling uncertainty conditional on the already fitted model and calibration map; they do not include model-development uncertainty.
- Repeated landmarks are handled by patient-clustered resampling, but single-center label error and transportability remain unresolved.
- Subgroup prevalence differs, so PR-AUC is interpreted alongside PR-AUC lift and event counts.
- The 2020-2022 period remains a post-hoc sensitivity cohort for the revised safe pipeline, not a pristine lockbox.
- External validation remains the required confirmation step.

## Data inventory
- 5,694 landmark rows, 590 patients, and 648 episodes.
- 242 positive landmark rows (4.25%).
- Calibration table: 10 equal-frequency risk groups.

