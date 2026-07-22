# Run 29: Outcome validity and leakage audit

## Decision question
Does the frozen development model use knowledge that an early blood specimen eventually grew an organism before that result was available, and is performance robust to removing that feature?

## Locked design
- Primary target: `future_strict_primary_or_uncertain_cvc_bsi_proxy_7d`.
- Train: 2008-2013; Platt calibration: 2014-2016; validation: 2017-2019.
- Model and hyperparameters copied from Run 23 without tuning.
- The 2020-2022 temporal lockbox was excluded before audit/modeling; 5,580 lockbox rows were not scored.

## Main findings
- 42.4% of development landmark rows carrying an early-positive specimen flag did not yet have an organism-positive `storetime` by the landmark.
- Original-feature validation PR-AUC: 0.0691 (1.63x prevalence).
- Safe exclusion validation PR-AUC: 0.0653 (1.54x prevalence).
- PR-AUC change after exclusion: -0.0038.
- The adjudication queue contains 100 stratified episodes and remains pending manual review.

## Interpretation
`early_positive_culture` is an outcome/result-derived episode flag, not a prospectively available specimen-order feature. `charttime` is specimen collection time, whereas `storetime` is the last known microbiology result update. Because the feature was copied onto all daily landmarks, it can reveal eventual culture positivity before the organism result was available. The manuscript-safe primary pipeline should exclude it. The storetime-aware replacement is exploratory because MIMIC documents `storetime` as the last known update rather than the first preliminary notification.

## Scope and limitations
- Run 29 is a development-only validity audit, not a new lockbox evaluation.
- `storetime` is an imperfect proxy for clinical notification time.
- MIMIC-IV-Note is not available locally, so narrative source adjudication is not yet performed.
- The balanced review sample supports failure-mode analysis, not unweighted population PPV estimation.

## Sources
- MIMIC-IV microbiologyevents documentation: https://mimic.mit.edu/docs/IV/modules/hosp/microbiologyevents.html
- CDC NHSN Bloodstream Infection Event guidance: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf

