# Run 6 - Static Model Characterization

Purpose:
- Close the static-model chapter before moving to landmark or dynamic modeling.
- Characterize whether the Run 5 strict-label model is useful as a risk-prioritization score, not just whether it has acceptable ROC-AUC/PR-AUC.

Clinical logic:
- Calibration asks whether predicted risk strata correspond to observed strict CLABSI rates.
- Alert-burden tables translate thresholds into nurse/workflow-facing quantities: alerts per 100 catheterized ICU stays, false alerts per true positive, captured cases, and missed cases.
- Top-risk tables support sanity checking of whether high-risk predictions look clinically plausible.
- Subgroup tables test whether performance is concentrated in documentation-heavy or physiologic subgroups.
- Feature stability and bootstrap CIs help distinguish robust signal from small-sample noise.

Run order:
1. Ensure Run 5 has completed.
2. Run `SRC\05_static_characterization\Modeling 04 v0.3a Static Characterization.py`.

Expected outputs:
- `Outputs\Run 6 (Static Model Characterization)\run6_static_characterization_summary.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_calibration_fixed_risk_bins.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_calibration_deciles.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_alarm_burden_threshold_table.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_subgroup_performance.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_top_100_predicted_risk_stays.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_feature_stability_summary.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_feature_stability_run41_vs_run5.csv`
- `Outputs\Run 6 (Static Model Characterization)\run6_bootstrap_metric_ci.csv`
- `Outputs\Run 6 (Static Model Characterization)\plots\run6_calibration_deciles.png`
- `Outputs\Run 6 (Static Model Characterization)\plots\run6_alert_burden_by_threshold.png`
- `Outputs\Run 6 (Static Model Characterization)\plots\run6_threshold_tradeoff_metrics.png`

Interpretation focus:
- Does observed strict CLABSI rate rise across predicted risk bins?
- Which threshold has a realistic alert burden?
- Is the no-site conservative model close enough to present alongside the full model?
- Are key features stable between Run 4.1 and Run 5?
- Are uncertainty intervals wide enough to temper claims about small PR-AUC differences?

