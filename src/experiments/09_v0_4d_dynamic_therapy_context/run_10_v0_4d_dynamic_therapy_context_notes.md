# Run 10 - v0.4D Dynamic Therapy Context

## Purpose

Run 10 tests whether dynamic CLABSI prediction improves when the landmark model includes clinical-response context, not just measured physiology.

Run 8 showed that vitals add signal. Run 9 showed that the signal survives after removing lab/vital measurement-intensity proxies, but performance is still weak and alert burden remains high. Run 10 asks whether antibiotics and vasopressors capture clinically meaningful deterioration or treatment escalation before strict CLABSI.

## New Data Sources

- `hosp/prescriptions.csv.gz`
  - Antibiotic exposure before each landmark.
  - Broad-spectrum, anti-MRSA, antipseudomonal, carbapenem, and anaerobe-coverage flags.

- `icu/inputevents.csv.gz`
  - Vasopressor/vasoactive medication exposure before each landmark.
  - Norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, dobutamine, and milrinone.

## Run Order

1. `Data Extraction 01 v0.4D Therapy Context.py`
2. `Feature Engineering 03 v0.4D Dynamic Therapy Context.py`
3. `Modeling 07 v0.4D Dynamic Therapy Context.py`

## Design Logic

- Keep raw MIMIC table loading in extraction.
- Keep landmark aggregation in feature engineering.
- Reuse the v0.4B landmark dynamic matrix so the v0.4D run isolates the effect of therapy/context features.
- Continue evaluating 48h, 72h, and 168h horizons.
- Continue comparing standard labels against gray-zone-excluded labels.
- Continue using patient-level splits.

## What We Hope To Learn

- Whether therapy-context features improve short-horizon prediction more than vitals/labs alone.
- Whether antibiotic and vasopressor exposure act as clinically interpretable acuity markers.
- Whether dynamic performance becomes more competitive with the strict-label static benchmark.
- Whether the top SHAP drivers shift from documentation/context variables toward treatment and physiologic response variables.

## Outputs

Expected output folder:

`Outputs/Run 10 (v0.4D Dynamic Therapy Context)`

Expected key files:

- `v0_4d_therapy_context_model_comparison.csv`
- `v0_4d_therapy_context_best_by_frame_horizon.csv`
- `v0_4d_therapy_context_threshold_table.csv`
- `v0_4d_therapy_context_stay_level_summary.csv`
- `v0_4d_therapy_context_calibration_deciles.csv`
- `v0_4d_therapy_context_feature_audit.csv`
- `plots/v0_4d_standard_best_pr_curves.png`
- `plots/v0_4d_gray_zone_excluded_best_pr_curves.png`
- `plots/v0_4d_therapy_context_best_model_shap.png`

