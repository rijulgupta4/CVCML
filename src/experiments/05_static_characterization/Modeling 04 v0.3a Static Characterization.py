# %% Imports and paths

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    print("Missing characterization dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn matplotlib joblib xgboost")
    sys.exit(1)


PROJECT_PATH = r"C:\path\to\CVCML"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
RUN5_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 5 (v0.3a Strict Organism Sensitivity)")
RUN41_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 4.1 (v0.2 Corrected Static No Audit Leakage)")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 6 (Static Model Characterization)")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_features_v0_3a.csv")
PRIMARY_MODEL_FILE = os.path.join(RUN5_PATH, "models", "v0_3a_strict_full.joblib")
RUN5_COMPARISON_FILE = os.path.join(RUN5_PATH, "v0_3a_strict_organism_model_comparison.csv")
RUN5_IMPORTANCE_FILE = os.path.join(RUN5_PATH, "v0_3a_strict_full_feature_importance.csv")
RUN41_IMPORTANCE_FILE = os.path.join(RUN41_PATH, "v0.2_full_corrected_feature_importance.csv")

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 1000


# %% Helper functions

def make_subject_level_splits(df, subject_col, target_col):
    subject_labels = (
        df.groupby(subject_col)[target_col]
        .max()
        .reset_index()
        .rename(columns={target_col: "subject_positive"})
    )

    rng = np.random.default_rng(RANDOM_STATE)
    train_subjects = []
    val_subjects = []
    test_subjects = []

    for label, group in subject_labels.groupby("subject_positive"):
        subjects = group[subject_col].to_numpy().copy()
        rng.shuffle(subjects)
        n_total = len(subjects)
        n_train = int(round(n_total * 0.60))
        n_val = int(round(n_total * 0.20))

        train_subjects.extend(subjects[:n_train])
        val_subjects.extend(subjects[n_train:n_train + n_val])
        test_subjects.extend(subjects[n_train + n_val:])

    train_df = df[df[subject_col].isin(train_subjects)].copy()
    val_df = df[df[subject_col].isin(val_subjects)].copy()
    test_df = df[df[subject_col].isin(test_subjects)].copy()

    return train_df, val_df, test_df


def threshold_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    n = len(y_true)

    return {
        "threshold": float(threshold),
        "n_stays": int(n),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision_ppv": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "alerts": int(pred.sum()),
        "alerts_per_100_stays": pred.sum() / n * 100,
        "strict_clabsi_captured_per_100_stays": tp / n * 100,
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
        "alerts_per_true_positive": pred.sum() / tp if tp else np.inf,
        "missed_cases": int(fn),
    }


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_ap(y_true, y_prob):
    if np.sum(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, y_prob)


def derive_cvc_type(row):
    cvc_cols = [col for col in row.index if col.startswith("cvc_type_")]
    active = [col.replace("cvc_type_", "") for col in cvc_cols if row[col] == 1]
    return active[0] if active else "Unknown"


def derive_gender(row):
    if "gender_F" in row.index and row["gender_F"] == 1:
        return "F"
    if "gender_M" in row.index and row["gender_M"] == 1:
        return "M"
    return "Unknown"


def add_readable_groups(df):
    df = df.copy()
    df["cvc_type_readable"] = df.apply(derive_cvc_type, axis=1)
    df["gender_readable"] = df.apply(derive_gender, axis=1)
    df["site_known_group"] = np.where(df["site_known"] == 1, "site_known", "site_unknown")
    df["lactate_measured_group"] = np.where(df["lactate_measured"] == 1, "lactate_measured", "lactate_not_measured")
    df["age_band"] = pd.cut(
        df["anchor_age"],
        bins=[-np.inf, 49, 64, 79, np.inf],
        labels=["<50", "50-64", "65-79", "80+"],
    ).astype(str)
    df["dwell_at_ref_band"] = pd.cut(
        df["dwell_at_ref_hours"],
        bins=[-np.inf, 72, 120, 240, np.inf],
        labels=["<=72h", "72-120h", "120-240h", ">240h"],
    ).astype(str)
    return df


def subgroup_rows(df, y_col, prob_col, threshold):
    rows = []
    subgroup_specs = [
        ("site_known", "site_known_group"),
        ("lactate_measured", "lactate_measured_group"),
        ("age_band", "age_band"),
        ("dwell_at_ref_band", "dwell_at_ref_band"),
        ("cvc_type", "cvc_type_readable"),
        ("gender", "gender_readable"),
    ]

    for subgroup, col in subgroup_specs:
        for level, group in df.groupby(col, dropna=False):
            y_true = group[y_col].astype(int).to_numpy()
            y_prob = group[prob_col].to_numpy()
            metrics = threshold_metrics(y_true, y_prob, threshold)
            metrics.update({
                "subgroup": subgroup,
                "level": str(level),
                "positive": int(y_true.sum()),
                "negative": int((y_true == 0).sum()),
                "prevalence": float(np.mean(y_true)),
                "roc_auc": safe_auc(y_true, y_prob),
                "pr_auc": safe_ap(y_true, y_prob),
            })
            rows.append(metrics)
    return pd.DataFrame(rows)


def make_calibration_tables(eval_df):
    fixed_bins = [0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
    eval_df = eval_df.copy()
    eval_df["fixed_risk_bin"] = pd.cut(
        eval_df["predicted_risk"],
        bins=fixed_bins,
        include_lowest=True,
        right=False,
    )

    fixed_calibration = (
        eval_df
        .groupby("fixed_risk_bin", observed=False)
        .agg(
            n_stays=("clabsi", "size"),
            positives=("clabsi", "sum"),
            mean_predicted_risk=("predicted_risk", "mean"),
            median_predicted_risk=("predicted_risk", "median"),
            observed_strict_clabsi_rate=("clabsi", "mean"),
        )
        .reset_index()
    )
    fixed_calibration["risk_bin"] = fixed_calibration["fixed_risk_bin"].astype(str)
    fixed_calibration = fixed_calibration.drop(columns=["fixed_risk_bin"])

    eval_df["risk_decile"] = pd.qcut(
        eval_df["predicted_risk"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    decile_calibration = (
        eval_df
        .groupby("risk_decile", observed=False)
        .agg(
            n_stays=("clabsi", "size"),
            positives=("clabsi", "sum"),
            mean_predicted_risk=("predicted_risk", "mean"),
            median_predicted_risk=("predicted_risk", "median"),
            observed_strict_clabsi_rate=("clabsi", "mean"),
        )
        .reset_index()
    )
    decile_calibration["risk_decile"] = decile_calibration["risk_decile"].astype(int) + 1
    return fixed_calibration, decile_calibration


def plot_calibration(decile_calibration):
    plt.figure(figsize=(6, 5))
    plt.plot(
        decile_calibration["mean_predicted_risk"],
        decile_calibration["observed_strict_clabsi_rate"],
        marker="o",
        linewidth=2,
    )
    max_axis = max(
        decile_calibration["mean_predicted_risk"].max(),
        decile_calibration["observed_strict_clabsi_rate"].max(),
    ) * 1.15
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed strict CLABSI rate")
    plt.title("Run 6 Calibration by Risk Decile")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "run6_calibration_deciles.png"), dpi=300)
    plt.close()


def plot_threshold_tradeoff(workflow_table):
    plot_df = workflow_table.copy()
    plt.figure(figsize=(7, 5))
    plt.plot(plot_df["threshold"], plot_df["recall_sensitivity"], marker="o", label="Recall")
    plt.plot(plot_df["threshold"], plot_df["specificity"], marker="o", label="Specificity")
    plt.plot(plot_df["threshold"], plot_df["precision_ppv"], marker="o", label="PPV")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Run 6 Threshold Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "run6_threshold_tradeoff_metrics.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(plot_df["threshold"], plot_df["alerts_per_100_stays"], marker="o", label="Alerts per 100 stays")
    plt.plot(plot_df["threshold"], plot_df["false_alerts_per_true_positive"], marker="o", label="False alerts per TP")
    plt.xlabel("Threshold")
    plt.ylabel("Workflow burden")
    plt.title("Run 6 Alert Burden by Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "run6_alert_burden_by_threshold.png"), dpi=300)
    plt.close()


def bootstrap_metric_ci(y_true, y_prob, n_iterations=BOOTSTRAP_ITERATIONS):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    n = len(y_true)
    valid = 0

    for idx in range(n_iterations):
        sample_idx = rng.integers(0, n, size=n)
        y_sample = y_true[sample_idx]
        p_sample = y_prob[sample_idx]
        if len(np.unique(y_sample)) < 2:
            continue
        valid += 1
        rows.append({
            "bootstrap_iteration": idx,
            "roc_auc": roc_auc_score(y_sample, p_sample),
            "pr_auc": average_precision_score(y_sample, p_sample),
            "brier_score": brier_score_loss(y_sample, p_sample),
        })

    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["roc_auc", "pr_auc", "brier_score"]:
        ci_rows.append({
            "metric": metric,
            "n_valid_bootstrap_samples": valid,
            "mean": boot[metric].mean(),
            "median": boot[metric].median(),
            "ci_lower_2_5": boot[metric].quantile(0.025),
            "ci_upper_97_5": boot[metric].quantile(0.975),
        })
    return boot, pd.DataFrame(ci_rows)


def feature_stability_tables():
    run5 = pd.read_csv(RUN5_IMPORTANCE_FILE).rename(columns={"importance": "run5_importance"})
    run41 = pd.read_csv(RUN41_IMPORTANCE_FILE).rename(columns={"importance": "run41_importance"})

    run5["run5_rank"] = run5["run5_importance"].rank(ascending=False, method="min")
    run41["run41_rank"] = run41["run41_importance"].rank(ascending=False, method="min")

    merged = run5.merge(run41, on="feature", how="outer")
    merged["run5_importance"] = merged["run5_importance"].fillna(0)
    merged["run41_importance"] = merged["run41_importance"].fillna(0)
    merged["run5_rank"] = merged["run5_rank"].fillna(len(run5) + 1)
    merged["run41_rank"] = merged["run41_rank"].fillna(len(run41) + 1)
    merged["rank_shift_run5_minus_run41"] = merged["run5_rank"] - merged["run41_rank"]
    merged = merged.sort_values("run5_rank")

    top10_run5 = set(run5.nsmallest(10, "run5_rank")["feature"])
    top10_run41 = set(run41.nsmallest(10, "run41_rank")["feature"])
    top20_run5 = set(run5.nsmallest(20, "run5_rank")["feature"])
    top20_run41 = set(run41.nsmallest(20, "run41_rank")["feature"])

    rank_corr = merged[["run5_rank", "run41_rank"]].corr(method="spearman").iloc[0, 1]
    summary = pd.DataFrame([{
        "run5_top10_overlap_with_run41": len(top10_run5 & top10_run41),
        "run5_top20_overlap_with_run41": len(top20_run5 & top20_run41),
        "spearman_rank_correlation": rank_corr,
        "run5_top10_features": "; ".join(run5.nsmallest(10, "run5_rank")["feature"]),
        "run41_top10_features": "; ".join(run41.nsmallest(10, "run41_rank")["feature"]),
    }])
    return merged, summary


# %% Load data and primary model

print("Loading v0.3a feature matrix...")
df = pd.read_csv(FEATURE_FILE)

print("Loading saved Run 5 full model...")
bundle = joblib.load(PRIMARY_MODEL_FILE)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
auto_threshold = float(bundle["threshold"])

target_col = "clabsi"
train_df, val_df, test_df = make_subject_level_splits(df, "subject_id", target_col)
y_test = test_df[target_col].astype(int).to_numpy()
test_prob = model.predict_proba(test_df[feature_cols])[:, 1]

eval_df = test_df.copy()
eval_df["predicted_risk"] = test_prob
eval_df["alert_at_auto_threshold"] = (eval_df["predicted_risk"] >= auto_threshold).astype(int)
eval_df = add_readable_groups(eval_df)

print(f"Test set: {len(eval_df):,} stays | {int(eval_df['clabsi'].sum()):,} positives ({eval_df['clabsi'].mean() * 100:.1f}%)")
print(f"Auto threshold from Run 5: {auto_threshold:.4f}")


# %% Overall characterization summary

overall_metrics = threshold_metrics(y_test, test_prob, auto_threshold)
overall_metrics.update({
    "model": "v0.3a strict full",
    "roc_auc": roc_auc_score(y_test, test_prob),
    "pr_auc": average_precision_score(y_test, test_prob),
    "brier_score": brier_score_loss(y_test, test_prob),
    "base_rate": float(np.mean(y_test)),
})
overall_summary = pd.DataFrame([overall_metrics])
overall_file = os.path.join(OUTPUT_PATH, "run6_static_characterization_summary.csv")
overall_summary.to_csv(overall_file, index=False)


# %% Calibration

fixed_calibration, decile_calibration = make_calibration_tables(eval_df)
fixed_calibration_file = os.path.join(OUTPUT_PATH, "run6_calibration_fixed_risk_bins.csv")
decile_calibration_file = os.path.join(OUTPUT_PATH, "run6_calibration_deciles.csv")
fixed_calibration.to_csv(fixed_calibration_file, index=False)
decile_calibration.to_csv(decile_calibration_file, index=False)
plot_calibration(decile_calibration)


# %% Alarm burden / workflow threshold table

thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
thresholds = np.unique(np.append(thresholds, auto_threshold))
thresholds = np.sort(thresholds)
workflow_table = pd.DataFrame([threshold_metrics(y_test, test_prob, threshold) for threshold in thresholds])
workflow_file = os.path.join(OUTPUT_PATH, "run6_alarm_burden_threshold_table.csv")
workflow_table.to_csv(workflow_file, index=False)
plot_threshold_tradeoff(workflow_table)


# %% Subgroup performance

subgroups = subgroup_rows(eval_df, "clabsi", "predicted_risk", auto_threshold)
subgroup_file = os.path.join(OUTPUT_PATH, "run6_subgroup_performance.csv")
subgroups.to_csv(subgroup_file, index=False)


# %% Top-risk case review table

top_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "predicted_risk",
    "alert_at_auto_threshold",
    "clabsi",
    "clabsi_pragmatic_v0_2",
    "pragmatic_downgraded_to_negative",
    "strict_positive_orgs",
    "strict_label_reason",
    "starttime",
    "endtime",
    "ref_time",
    "culture_time",
    "cvc_type_readable",
    "gender_readable",
    "anchor_age",
    "site_known",
    "dwell_at_ref_hours",
    "lactate_measured",
    "lactate_mean",
    "lactate_last",
    "lactate_trend",
    "wbc_mean",
    "wbc_last",
    "wbc_trend",
    "platelets_mean",
    "platelets_last",
    "platelets_trend",
    "creatinine_mean",
    "creatinine_last",
    "creatinine_trend",
]
top_cols = [col for col in top_cols if col in eval_df.columns]
top_risk = eval_df.sort_values("predicted_risk", ascending=False).head(100)[top_cols]
top_risk_file = os.path.join(OUTPUT_PATH, "run6_top_100_predicted_risk_stays.csv")
top_risk.to_csv(top_risk_file, index=False)


# %% Feature stability against Run 4.1

feature_stability, feature_stability_summary = feature_stability_tables()
feature_stability_file = os.path.join(OUTPUT_PATH, "run6_feature_stability_run41_vs_run5.csv")
feature_stability_summary_file = os.path.join(OUTPUT_PATH, "run6_feature_stability_summary.csv")
feature_stability.to_csv(feature_stability_file, index=False)
feature_stability_summary.to_csv(feature_stability_summary_file, index=False)


# %% Bootstrap uncertainty

print(f"Bootstrapping test-set metrics ({BOOTSTRAP_ITERATIONS:,} iterations)...")
bootstrap_samples, bootstrap_ci = bootstrap_metric_ci(y_test, test_prob)
bootstrap_ci_file = os.path.join(OUTPUT_PATH, "run6_bootstrap_metric_ci.csv")
bootstrap_samples_file = os.path.join(OUTPUT_PATH, "run6_bootstrap_metric_samples.csv")
bootstrap_ci.to_csv(bootstrap_ci_file, index=False)
bootstrap_samples.to_csv(bootstrap_samples_file, index=False)


# %% Conservative-model comparison reference

run5_comparison = pd.read_csv(RUN5_COMPARISON_FILE)
conservative = run5_comparison[
    run5_comparison["model"].isin([
        "v0.3a strict full",
        "v0.3a strict no site_known",
        "v0.3a strict no lactate",
        "v0.3a strict routine labs only",
    ])
].copy()
conservative_file = os.path.join(OUTPUT_PATH, "run6_conservative_static_candidate_comparison.csv")
conservative.to_csv(conservative_file, index=False)


# %% Save manifest and print summary

manifest_rows = [
    ("Overall characterization", overall_file),
    ("Calibration fixed risk bins", fixed_calibration_file),
    ("Calibration deciles", decile_calibration_file),
    ("Alarm burden threshold table", workflow_file),
    ("Subgroup performance", subgroup_file),
    ("Top 100 predicted-risk stays", top_risk_file),
    ("Feature stability detail", feature_stability_file),
    ("Feature stability summary", feature_stability_summary_file),
    ("Bootstrap metric CIs", bootstrap_ci_file),
    ("Bootstrap metric samples", bootstrap_samples_file),
    ("Conservative model comparison", conservative_file),
    ("Calibration plot", os.path.join(PLOT_PATH, "run6_calibration_deciles.png")),
    ("Threshold tradeoff plot", os.path.join(PLOT_PATH, "run6_threshold_tradeoff_metrics.png")),
    ("Alert burden plot", os.path.join(PLOT_PATH, "run6_alert_burden_by_threshold.png")),
]
manifest = pd.DataFrame(manifest_rows, columns=["output", "path"])
manifest_file = os.path.join(OUTPUT_PATH, "run6_output_manifest.csv")
manifest.to_csv(manifest_file, index=False)

print("")
print("Run 6 static characterization summary:")
print(overall_summary[[
    "model",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "base_rate",
    "threshold",
    "recall_sensitivity",
    "specificity",
    "precision_ppv",
    "alerts",
    "alerts_per_100_stays",
    "false_alerts_per_true_positive",
]].round(4).to_string(index=False))

print("")
print("Saved outputs:")
for label, path in manifest_rows:
    print(f"  {label}: {path}")

print("")
print("Modeling 04 v0.3a Static Characterization complete.")

