# Run 16 - v0.5 Catheter Episode Redesign

## Purpose

Run 16 starts the v0.5 redesign. It does not train a model. It rebuilds the cohort denominator around catheter-exposure episodes rather than selecting the eventual longest catheter per ICU stay.

This directly addresses the largest v0.4 methodological concern: longest-line selection used future information, discarded other catheter exposures, and could associate cultures with the wrong line exposure.

## What Changed

- Kept all CVC procedureevents records for the target central-line item IDs.
- Reconstructed continuous CVC exposure periods within each ICU stay.
- Merged overlapping catheter intervals and intervals separated by a prespecified small gap.
- Used a 4-hour maximum gap to define continuous central-line exposure.
- Required at least 48 observed exposure hours for label eligibility.
- Associated positive blood cultures with each eligible exposure period.
- Built two labels:
  - `cvc_bsi_broad_proxy`: any positive blood culture after 48 hours of observed CVC exposure.
  - `cvc_bsi_strict_proxy`: clear pathogen after 48 hours, or at least two common-commensal cultures at distinct charttimes.
- Preserved the label name as a proxy, not adjudicated CLABSI.

## Important Limitations

This run still does not adjudicate full NHSN CLABSI.

Missing or incomplete elements:

- Secondary-source infection exclusion.
- MBI-LCBI classification/exclusion.
- Full symptom rules for common commensals.
- Manual adjudication of whether a blood culture is primary LCBI.

Recommended outcome wording after Run 16:

`strict CVC-associated BSI proxy`

## Main Results

- Raw CVC procedure records: 26,285
- Raw CVC records with >=48h duration: 13,740
- ICU stays with any CVC procedure record: 21,541
- Continuous CVC exposure periods: 22,812
- Eligible exposure periods with >=48 observed hours: 11,602
- Eligible stays: 11,060
- Stays with multiple exposure periods: 1,162
- Broad proxy positive episodes: 528 (4.55%)
- Strict proxy positive episodes: 371 (3.20%)
- Broad positives downgraded by strict organism rule: 157
- Early positive culture episodes: 282

## Temporal Lockbox Candidate

The audit suggests using the latest `anchor_year_group` as a candidate temporal lockbox:

- Development candidate: 2008-2019 anchor-year groups.
- Temporal lockbox candidate: 2020-2022 anchor-year group.

The 2020-2022 group contains:

- 810 eligible episodes
- 783 eligible stays
- 691 patients
- 33 strict proxy positives
- Strict proxy prevalence: 4.07%

This should be frozen before v0.5 modeling begins.

## Outputs

Data folder:

`data/v0_5`

Key data files:

- `v0_5_cvc_procedure_events.csv`
- `v0_5_catheter_exposure_periods.csv`
- `v0_5_episode_culture_detail.csv`
- `v0_5_episode_label_audit.csv`
- `v0_5_qualifying_organism_counts.csv`
- `v0_5_temporal_lockbox_candidate_audit.csv`
- `v0_5_censoring_audit.csv`
- `v0_5_duration_audit.csv`
- `v0_5_label_reason_counts.csv`
- `v0_5_episode_per_stay_audit.csv`

Run 16 output folder:

`Outputs/Run 16 (v0.5 Catheter Episode Redesign)`

The output folder contains copies of the most important audit tables plus the output manifest.

## Next Step

Build daily episode-level landmarks using the Run 16 exposure-period denominator.

The next script should:

- Generate daily landmark rows for eligible exposure periods.
- Use one seven-day target.
- Stop at infection, discharge, death, ICU outtime, or observed line removal.
- Explicitly mark competing/censoring events.
- Preserve the 2020-2022 anchor-year group as an untouched temporal lockbox.

