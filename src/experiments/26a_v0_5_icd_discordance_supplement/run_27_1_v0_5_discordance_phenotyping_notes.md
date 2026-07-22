# Run 27.1 - ICD Discordance Supplement

## Purpose

Run 27.1 reconciles the grain mismatch identified after Run 27, assigns transparent structured reasons to proxy/ICD discordance, and creates deterministic review queues for later chart validation. It does not refit the model, change thresholds, or redefine the primary outcome.

## Grain Reconciliation

- Episode-level comparison: 22,812 episodes, 60 both-positive, positive-set Jaccard 6.6%.
- Admission-level comparison: 19,346 admissions, 56 both-positive, positive-set Jaccard 8.5%.
- ICD-specific CVC-BSI codes appeared in 434 admissions; 167 (38.5%) contained multiple reconstructed catheter episodes.
- Because diagnoses_icd is admission-level and assigned at discharge, it cannot identify which catheter episode generated the code. Admission-level agreement is therefore the fairer external agreement analysis.

## Discordance Phenotypes

- No structured qualifying blood culture during episode: 288 episodes (46.7% within proxy_neg_icd_pos).
- No observed 48h eligible CVC exposure: 267 episodes (43.3% within proxy_neg_icd_pos).
- Positive culture before 48h eligibility: 37 episodes (6.0% within proxy_neg_icd_pos).
- Broad culture failed strict organism logic: 13 episodes (2.1% within proxy_neg_icd_pos).
- Strict event excluded as possible secondary BSI: 12 episodes (1.9% within proxy_neg_icd_pos).
- Uncertain: nonconcordant source culture, no CVC-BSI ICD code: 146 episodes (63.2% within proxy_pos_icd_neg).
- Uncertain: source ICD evidence, no CVC-BSI ICD code: 66 episodes (28.6% within proxy_pos_icd_neg).
- Primary-likely proxy, no CVC-BSI ICD code: 19 episodes (8.2% within proxy_pos_icd_neg).

## Interpretation

The dominant ICD-positive/proxy-negative phenotype is absence of a structured qualifying blood culture during the reconstructed episode. This does not establish that the ICD code is wrong: possible explanations include incomplete procedureevents line documentation, an infection tied to a different catheter episode in the same admission, culture timing outside the proxy window, or administrative coding that cannot be temporally localized.

Proxy-positive/ICD-negative episodes are mostly source-screened uncertain events rather than the small primary-likely subset. They remain appropriate for sensitivity analysis and targeted review, but they should not be described as administratively confirmed CLABSI.

## Manual Review Frame

- Deterministic balanced sample: 60 episodes, generated with seed 2028.
- The sample spans both discordance directions, structured phenotypes, and MIMIC eras where cases are available.
- Structured data alone cannot adjudicate NHSN CLABSI. A future review should examine clinical notes and source attribution while preserving the existing proxy and ICD labels as separate evidence streams.

## Sources and Rationale

- CDC NHSN January 2026 BSI/CLABSI manual: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf
- MIMIC diagnoses_icd documentation: https://mimic.mit.edu/docs/IV/modules/hosp/diagnoses_icd.html
- MIMIC procedureevents documentation: https://mimic.mit.edu/docs/IV/modules/icu/procedureevents.html

## Key Output Files

- `v0_5_run27_1_grain_reconciled_agreement.csv`
- `v0_5_run27_1_admission_level_icd_agreement.csv`
- `v0_5_run27_1_icd_episode_attribution_ambiguity.csv`
- `v0_5_run27_1_discordance_phenotype_summary.csv`
- `v0_5_run27_1_discordance_by_era.csv`
- `v0_5_run27_1_proxy_positive_icd_negative_review_queue.csv`
- `v0_5_run27_1_icd_positive_proxy_negative_review_queue.csv`
- `v0_5_run27_1_balanced_manual_review_sample.csv`
- `v0_5_run27_1_lockbox_discordance_review_queue.csv`
- `plots/v0_5_run27_1_discordance_phenotypes.png`
- `plots/v0_5_run27_1_discordance_by_era.png`
