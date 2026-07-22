# Run 15 - v0.4I Final Candidate Characterization

## Purpose

Run 15 does not train a new model. It freezes the best current candidates from the static and dynamic tracks and characterizes them by clinical role.

The goal is to answer a different question than earlier modeling runs:

- Which model is best for baseline risk stratification?
- Which model is best for dynamic 7-day surveillance or infection-prevention review?
- Which model is best interpreted as a workflow/care-process signal?
- What alert burden would each selected candidate create at its chosen operating point?
- Which feature families drive each candidate?

## Inputs

- Static strict organism model:
  - `Outputs/Run 5 (v0.3a Strict Organism Sensitivity)`
  - `Outputs/Run 6 (Static Model Characterization)`
- Dynamic split-use candidates:
  - `Outputs/Run 14 (v0.4H Split Dynamic Use Cases)`

## Frozen Candidates

1. `Static baseline`
   - Source: Run 5/6, `v0.3a strict full`
   - Role: baseline CLABSI risk stratification near catheter reference time.

2. `168h dynamic surveillance`
   - Source: Run 14, `v0.4H 168h baseline cleaned therapy physiology`
   - Role: 7-day infection-prevention surveillance and review-list prioritization.

3. `72h dynamic workflow`
   - Source: Run 14, `v0.4H 72h full care process workflow`
   - Role: near-term workflow-aware monitoring and care-process signal analysis.

## Outputs

Output folder:

`Outputs/Run 15 (v0.4I Final Candidate Characterization)`

Generated files:

- `v0_4i_candidate_model_summary.csv`
- `v0_4i_clinical_role_summary.csv`
- `v0_4i_alert_burden_comparison.csv`
- `v0_4i_top_risk_review_comparison.csv`
- `v0_4i_calibration_summary.csv`
- `v0_4i_feature_family_summary.csv`
- `v0_4i_top_features_by_candidate.csv`
- `v0_4i_final_candidate_characterization_output_manifest.csv`

Generated plots:

- `plots/v0_4i_candidate_model_performance.png`
- `plots/v0_4i_candidate_alert_burden.png`
- `plots/v0_4i_feature_family_heatmap.png`

## Interpretation Notes

Static remains the strongest overall discriminator and the clearest baseline risk model.

The 168h dynamic model is weaker than static in PR-AUC and PPV, but it is the most coherent dynamic surveillance candidate because it answers a different clinical question: who should be reviewed over the coming week?

The 72h dynamic model has high recall but a high false-alert burden. Its value is not as a direct nurse alert yet; it is better treated as evidence that workflow, therapy, fluid, caregiver, line-care, lab, and vital-sign context carries signal in short-horizon CLABSI prediction.

Static top-risk review can be computed only from the saved Run 6 top-100 file. Full static top-5% and top-10% review yield would require saving or regenerating full test-set static predictions.

