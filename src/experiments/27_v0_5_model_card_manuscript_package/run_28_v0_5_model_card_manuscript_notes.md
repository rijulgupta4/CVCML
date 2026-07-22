# Run 28 - v0.5 Model Card and Manuscript Package

Run 28 is an evidence-consolidation run. It did not refit, recalibrate, or tune the frozen model and did not reuse the temporal lockbox for development.

Primary locked result: ROC-AUC 0.623, PR-AUC 0.110 (1.85x prevalence), Brier Skill Score 0.013, calibration slope 1.056, and E:O 0.888 in 2,262 rows from 270 episodes with 26 positive episodes.

Primary operating results use the target-aligned Run 25 estimand. Higher episode-capture values from Run 26 are clearly marked exploratory because they use a broader post-hoc episode estimand.

The package consistently names the outcome a strict CVC-associated BSI proxy and treats ICD agreement as convergent validity, not ground truth.

