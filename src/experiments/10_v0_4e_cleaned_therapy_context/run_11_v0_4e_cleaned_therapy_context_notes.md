# Run 11 - v0.4E Cleaned Therapy Context

## Purpose

Run 10 showed that therapy/context features carried some dynamic signal, but the antibiotic extraction mixed systemic treatment with local/topical/non-treatment medication records. Run 11 cleans that signal and tests whether the apparent improvement survives after removing likely proxy noise.

The central question is whether systemic antibiotic and vasopressor context adds clinically meaningful early-warning signal, or whether the model is mostly learning clinician suspicion and documentation artifacts.

## What Changed

- Reuses the v0.4D therapy extracts to avoid another slow raw MIMIC scan.
- Filters antibiotic exposure to systemic routes and excludes local, topical, ophthalmic, otic, lock, enema, oral vancomycin liquid, desensitization, and challenge records.
- Separates active antibiotic exposure from new antibiotic starts.
- Keeps vasopressor context as a marker of acute instability.
- Adds explicit ablation models for no-site, no-antibiotic, antibiotic-only, new-start-only, active-existing-antibiotic, vasopressor-only, labs, vitals, and full-value feature sets.

## Run Order

1. `Data Extraction 01 v0.4E Cleaned Therapy.py`
2. `Feature Engineering 03 v0.4E Cleaned Therapy Context.py`
3. `Modeling 08 v0.4E Cleaned Therapy Context.py`

## Expected Outputs

- `data/v0_4e/v0_4e_cleaned_therapy_extraction_audit.csv`
- `data/v0_4e/v0_4e_cleaned_therapy_counts.csv`
- `data/v0_4e/v0_4e_cleaned_therapy_feature_audit.csv`
- `Outputs/Run 11 (v0.4E Cleaned Therapy Context)/v0_4e_cleaned_therapy_context_model_comparison.csv`
- `Outputs/Run 11 (v0.4E Cleaned Therapy Context)/v0_4e_cleaned_therapy_context_best_by_frame_horizon.csv`
- `Outputs/Run 11 (v0.4E Cleaned Therapy Context)/v0_4e_cleaned_therapy_context_threshold_table.csv`
- `Outputs/Run 11 (v0.4E Cleaned Therapy Context)/v0_4e_cleaned_therapy_context_stay_level_summary.csv`
- `Outputs/Run 11 (v0.4E Cleaned Therapy Context)/plots/`

## How To Interpret

- If cleaned therapy performs close to Run 10, therapy context likely captures real clinical trajectory.
- If cleaned therapy drops sharply, Run 10 was partly driven by medication-record artifacts.
- If new-start features outperform active-exposure features, treatment escalation may be a meaningful dynamic trigger.
- If active-existing-antibiotic features outperform new-start features, the model may mainly be detecting already-recognized infection risk.
- If no-antibiotic models remain comparable, the next dynamic improvement should focus on vitals, labs, timing, and patient-state trajectories rather than therapy features.

