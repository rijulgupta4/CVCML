# Code Guide

## Organization

Each folder represents one analytical run. Folder names preserve the development sequence. Human-facing run numbers are listed in `RUN_INDEX.md` because the folder prefixes are zero-offset after the initial pipeline.

Scripts use descriptive filenames for three recurring stages:

- `Data Extraction`: source-table filtering and cohort inputs;
- `Feature Engineering`: episode, landmark, and predictor construction;
- `Modeling`: fitting, calibration, evaluation, or characterization;
- `Reporting`: aggregation and document generation.

## Reading Order

Start with the folder README. It states the purpose, methodological status, files, and data boundary. Read scripts in numerical filename order when a folder contains more than one stage.

Early scripts often use `# %%` sections because development occurred in Spyder. Later scripts use module docstrings and small named functions. Comments are reserved for timing rules, leakage controls, censoring, label logic, and non-obvious implementation constraints. Routine assignments are left uncommented.

## Historical Code

Runs 1-15 are retained for provenance. They contain assumptions later found to be unsuitable for prospective interpretation. Their presence documents the audit trail; it does not make them recommended pipelines.

The primary sequence begins with Run 16 catheter-episode reconstruction. Run 25 is the frozen temporal evaluation. Runs 26-34 characterize and report results without establishing a new untouched test set.

## Local Configuration

Generic `C:\path\to\...` constants replace personal paths. Configure those constants or adapt them to environment variables before execution. Outputs should remain outside the public repository.
