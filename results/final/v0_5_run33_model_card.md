# CVCML v0.5 Leakage-Safe Model Card

**Status:** Research prototype; not for clinical use.

## Intended use

Bounded infection-prevention review-list prioritization for seven-day strict primary-or-uncertain CVC-associated BSI proxy risk. Not an autonomous diagnosis, calibrated bedside risk tool, or interruptive alert.

## Frozen safe candidate

- XGBoost with Platt calibration.
- Train 2008-2013; calibration 2014-2016; leakage-safe evaluation 2017-2019.
- `early_positive_culture` excluded after Run 29 outcome-validity audit.
- Target: `future_strict_primary_or_uncertain_cvc_bsi_proxy_7d`.

## Performance

- ROC-AUC: 0.612 (95% CI 0.518-0.703).
- PR-AUC: 0.065 (0.041-0.113); prevalence 4.25%.
- Brier Skill Score: 0.005 (-0.019-0.021).
- Top 10% episode review PPV: 15.4%; recall: 18.5%.

## External evidence

- ARMD-MGB: 32,887 positive blood-culture accessions; organism and partial source logic transported.
- eICU: failed exact-validation feasibility gate.
- Full external model validation: not achieved.

## Decision

Credible as a transparent retrospective review-prioritization research model. Not ready for clinical deployment.
