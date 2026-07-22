# Run Index

## Historical Development: Runs 1-15

Runs 1-6 established static baselines, optimized XGBoost, audited dwell-time and culture leakage, refined organism logic, and characterized calibration and alarm burden. Runs 7-15 introduced landmark prediction, vitals, therapy and care-process context, alert policies, split clinical use cases, and candidate characterization.

These runs are retained as provenance. Their cohorts and reference-time logic were superseded by v0.5 and should not be used as the primary evidence.

## v0.5 Redesign: Runs 16-25

| Run | Folder prefix | Purpose |
|---:|---|---|
| 16 | `15_` | Reconstruct all central-line exposure episodes; remove longest-line selection. |
| 17 | `16_` | Create daily landmarks and a seven-day future target. |
| 18 | `17_` | Constrained logistic/XGBoost development comparison. |
| 19 | `18_` | Calibration and review-policy development. |
| 20 | `19_` | Add dynamic vitals and therapy context. |
| 21 | `20_` | Compare 48 h, 72 h, and 168 h functional frames. |
| 22 | `21_` | Add partial secondary-source screening. |
| 23 | `22_` | Compare broad, primary-or-uncertain, and high-specificity labels. |
| 24 | `23_` | Prespecify operating policies without lockbox access. |
| 25 | `24_` | Open the frozen 2020-2022 temporal lockbox once. |

## Characterization And Reporting: Runs 26-34

| Run | Folder prefix | Purpose |
|---:|---|---|
| 26 | `25_` | Locked error analysis; no refitting. |
| 27 | `26_` | ICD agreement and proxy-label validation. |
| 27.1 | `26a_` | Admission-grain discordance phenotyping and review queues. |
| 28 | `27_` | Model card and manuscript-style evidence package. |
| 29 | `28_` | Outcome-validity audit identifying `early_positive_culture` leakage. |
| 30 | `29_` | Leakage-safe characterization with clustered bootstrap CIs. |
| 31 | `30_` | eICU and ARMD external-validation feasibility assessment. |
| 32 | `31_` | External organism/source-screen transportability analysis. |
| 33 | `32_` | Publication evidence consolidation. |
| 34 | `34_` | Same-frame logistic comparator and final document revision. |

The `33_v0_5_comprehensive_report` and `33_v0_5_manuscript_draft` folders contain document-generation scripts created alongside the numbered analytical runs.
