# Run 22 - v0.5 Secondary-Source Label Audit

## Purpose

Run 22 improves the outcome hierarchy before further model tuning. The goal is to move closer to NHSN CLABSI logic without pretending that MIMIC-IV contains fully adjudicated infection-prevention surveillance labels.

The run starts from the v0.5 strict CVC-associated BSI proxy and screens strict-positive blood culture episodes for evidence that the bloodstream infection may be secondary to another source.

## Label Hierarchy

Run 22 keeps the existing labels and adds source-screened refinements:

1. `cvc_bsi_broad_proxy`
   - Sensitive CVC-associated BSI proxy.
   - Positive blood culture temporally associated with CVC exposure.

2. `cvc_bsi_strict_proxy`
   - Current primary v0.5 label.
   - Eligible CVC exposure episode, positive blood culture after eligible line exposure, and organism logic for recognized pathogens or repeated common commensals.

3. `cvc_bsi_strict_primary_likely_proxy`
   - High-specificity subset of strict positives.
   - No nearby non-blood source culture evidence and no source-related ICD evidence.

4. `cvc_bsi_strict_primary_or_uncertain_proxy`
   - Pragmatic source-screened label.
   - Keeps strict positives unless there is concordant non-blood source culture evidence.
   - This is likely the best next modeling label.

5. `cvc_bsi_strict_secondary_possible_proxy`
   - Strict positives with concordant non-blood source culture evidence near the blood culture.
   - These are plausible secondary BSIs and should not be treated as clean CLABSI-proxy positives.

## Secondary-Source Evidence Logic

Evidence sources:

- MIMIC-IV `microbiologyevents`
- MIMIC-IV `diagnoses_icd` plus `d_icd_diagnoses`

Culture screen:

- Look within +/- 3 days of strict positive blood culture.
- Include non-blood source-candidate cultures:
  - urinary
  - respiratory
  - wound / skin / soft tissue
  - abdominal / GI / biliary
  - CSF / CNS
  - other sterile site
- Exclude blood specimens.
- Deduplicate susceptibility-level microbiology rows.
- Mark stronger secondary evidence only when the non-blood source organism is concordant with the blood culture organism.

ICD screen:

- Flag source-related diagnosis evidence for urinary, respiratory, skin/soft tissue, abdominal/biliary, surgical/procedure-related, and deep-focus infections.
- ICD-only evidence is treated as uncertain rather than a hard downgrade.

## Main Results

Episode-level counts:

| Label / category | Episodes |
|---|---:|
| Total CVC exposure episodes | 22,812 |
| Eligible 48h observed exposure episodes | 11,602 |
| Broad CVC-associated BSI proxy | 528 |
| Strict CVC-associated BSI proxy | 371 |
| Strict primary-likely proxy | 26 |
| Strict primary-or-uncertain proxy | 291 |
| Strict secondary-possible concordant proxy | 80 |
| Strict positives with any nearby non-blood source culture | 256 |
| Strict positives with concordant non-blood source culture | 80 |
| Strict positives with source-related ICD evidence | 309 |

Landmark-level 7-day future-positive rows:

| Target | Positive rows |
|---|---:|
| Original strict proxy | 1,576 |
| Primary-likely only | 115 |
| Primary-or-uncertain | 1,285 |
| Secondary-possible concordant | 291 |

## Interpretation

The first-pass idea of excluding any nearby source culture or ICD evidence was too aggressive. Many ICU patients have respiratory, urine, wound, or abdominal cultures near bacteremia, but that does not automatically mean the blood culture is secondary to that site.

The concordance-aware screen is more defensible:

- Concordant non-blood source culture is treated as stronger secondary-source evidence.
- Nonconcordant source cultures and ICD-only evidence are treated as uncertain.
- The `primary_or_uncertain` label preserves enough events for modeling while removing the clearest possible secondary BSIs.

The `primary_likely` label is useful as a high-specificity sensitivity analysis, but it is too sparse to serve as the primary model target by itself.

## Practical Conclusion

Do not replace the v0.5 strict label with `primary_likely` as the main target. Instead, the next modeling comparison should use:

- Original strict proxy as the current baseline label.
- Source-screened `primary_or_uncertain` as the improved primary candidate.
- `primary_likely` as a high-specificity sensitivity label.
- Secondary-possible concordant cases as an exclusion or competing category.

## Outputs

Stored in `data/v0_5`:

- `v0_5_run22_source_screened_episode_labels.csv`
- `v0_5_run22_source_screened_daily_landmarks.csv`

Stored in `Outputs/Run 22 (v0.5 Secondary Source Label Audit)`:

- `v0_5_run22_source_screen_label_audit.csv`
- `v0_5_run22_source_screen_class_counts.csv`
- `v0_5_run22_secondary_source_culture_detail.csv`
- `v0_5_run22_source_icd_detail.csv`
- `v0_5_run22_secondary_source_bucket_counts.csv`
- `v0_5_run22_strict_organism_by_source_class.csv`
- `v0_5_run22_manifest.csv`

## Next Step

Run 23 should be a label-sensitivity modeling run, not another feature extraction run:

- train the current best enriched model using original strict target,
- train the same model using `primary_or_uncertain`,
- optionally test `primary_likely` only as a sensitivity analysis,
- compare prevalence, PR-AUC lift, top-review-list PPV, episode capture, and feature stability.

