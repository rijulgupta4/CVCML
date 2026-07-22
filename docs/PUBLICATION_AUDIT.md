# Public Release And Data Agreement Audit

**Audit date:** 2026-07-21  
**Decision:** publish code and publication outputs; withhold data-like derivatives and trained models.

## Governing Principles

1. PhysioNet's MIMIC-IV guidance says derived datasets and models should be treated as sensitive and, if shared, placed on PhysioNet under the same agreement as the source data.
2. eICU-CRD and ARMD-MGB are available only to credentialed users who complete training and sign their data use agreements.
3. PhysioNet prohibits sharing credentialed data with third parties, including sending it through APIs or online platforms, unless the use complies with the agreement and verified handling requirements.
4. Public software repositories are appropriate for analysis code, not as substitutes for controlled distribution of clinical-data derivatives.

Official sources:

- [MIMIC-IV v3.1 guidance](https://physionet.org/content/mimiciv/3.1/)
- [eICU-CRD v2.0 access policy](https://physionet.org/content/eicu-crd/2.0/)
- [ARMD-MGB v1.0.0 access policy](https://physionet.org/content/armd-mgb/1.0.0/)
- [PhysioNet guidance on LLMs and online services](https://physionet.org/news/post/llm-responsible-use/)
- [PhysioNet publishing and license options](https://physionet.org/about/publish/)

## Release Matrix

| Artifact | Decision | Reason |
|---|---|---|
| Original analysis code | Publish | Reusable software; no clinical rows after path and secret audit. |
| Historical run notes | Publish | Methodological provenance; no row-level source data. |
| README, model card, run index | Publish | Repository-authored documentation and aggregate interpretation. |
| Publication-style aggregate figures | Publish | Report figures with no identifiers or reusable records. |
| Comprehensive project report | Publish after final author review | Appropriate as a narrative publication output, but the editable draft and current PDF remain outside this first code release. |
| Aggregate result CSVs | Withhold from GitHub | Machine-readable MIMIC-derived tables can function as derived datasets. |
| Raw source tables or notes | Never publish | Credentialed clinical data. |
| Cohorts, features, episode/landmark rows | Never publish on GitHub | Sensitive source-derived records. |
| Predictions and review queues | Never publish on GitHub | Row-level derivatives and selected cases. |
| Culture/accession-level ARMD outputs | Never publish on GitHub | Credentialed source derivatives. |
| Error-analysis and adjudication samples | Never publish on GitHub | Selected clinical episodes and outcome context. |
| Trained models/calibrators | Withhold; use PhysioNet review if shared | MIMIC-derived model artifacts are treated as sensitive. |
| Manuscript draft and supplement | Hold for now | Author, affiliation, ethics, journal, and preprint decisions remain unresolved. |

## Report-Based Publication Judgment

The public project should foreground four conclusions:

1. **The final model is modest, not clinically ready.** ROC-AUC 0.612, PR-AUC 0.065, and Brier Skill Score near zero do not support autonomous decision-making.
2. **Leakage auditing is a primary contribution.** Outcome-dependent reference times, longest-line selection, and outcome-adjacent culture information materially changed apparent performance.
3. **The endpoint remains a proxy.** Low agreement with ICD-coded central-line infection demonstrates disagreement between operational definitions, not proof that either is ground truth.
4. **The plausible role is bounded review prioritization.** Top-k PPV and recall remain uncertain and workload tradeoffs require prospective evaluation.

The repository should not advertise unqualified “CLABSI prediction,” claim external validation, imply a deployable alarm, or compare headline metrics directly with studies using different cohorts, outcomes, horizons, and censoring rules.

## Checks Applied

- no source dataset files or clinical notes;
- no patient, admission, stay, episode, landmark, culture, or accession rows;
- no predictions, review queues, or adjudication samples;
- no fitted model binaries;
- no personal absolute paths, credentials, keys, or tokens;
- no machine-readable derived result tables;
- public claims trace to the aggregate report or model card.

This is a conservative release decision, not legal advice. Current data use agreements and PhysioNet guidance remain controlling.
