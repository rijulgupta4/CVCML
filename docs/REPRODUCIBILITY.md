# Reproducibility Guide

The repository preserves the experiment history, but it is not a data distribution. Reproduction requires independent credentialed access to each source dataset and local execution under its data use agreement.

## Environment

Python dependencies are listed in `requirements.txt`. The reporting scripts also use `python-docx` and Pillow; PDF rendering used LibreOffice in the original environment.

Archived scripts contain generic `C:\path\to\...` placeholders. Configure project and dataset roots locally before running. Do not commit personal paths.

## Recommended v0.5 Order

1. Run 16 reconstructs all recorded catheter exposure episodes.
2. Run 17 creates daily landmark rows and seven-day outcomes.
3. Runs 18-24 develop models, enrich features, refine labels, and prespecify review policies without opening the temporal lockbox.
4. Run 25 performs the frozen temporal evaluation.
5. Runs 26-30 characterize errors, administrative-code agreement, leakage, calibration, uncertainty, and review burden without selecting a new model on the lockbox.
6. Runs 31-32 evaluate external-validation feasibility and label-component transportability.
7. Runs 33-34 consolidate evidence and add the same-frame logistic comparator.

## Boundaries

- Folder numbering is zero-offset relative to human-facing runs: folder `15_v0_5_catheter_episode_redesign` produced Run 16.
- Early scripts are retained for provenance but are methodologically superseded.
- The 2020-2022 period was protected during development. Repeated post-lockbox characterization means it is not a pristine future validation set for new tuning.
- Exact reproduction can depend on source-data and package versions.

Before execution, ensure outputs point outside this repository. Afterward, inspect `git status` and verify that no clinical data, derived rows, predictions, or model binaries are staged.
