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
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing dynamic modeling dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost matplotlib joblib")
    sys.exit(1)


PROJECT_PATH = r"C:\path\to\CVCML"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4b")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 8 (v0.4B Dynamic Vitals Horizon Sensitivity)")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4b.csv")
RANDOM_STATE = 42
MIN_RECALL_TARGET = 0.80
XGB_N_JOBS = 2
HORIZONS = [24, 48, 72, 168]


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

    for _, group in subject_labels.groupby("subject_positive"):
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


def select_threshold(y_true, y_prob, min_recall=MIN_RECALL_TARGET):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    candidates = []

    for idx, threshold in enumerate(thresholds):
        if recall[idx] >= min_recall:
            candidates.append((precision[idx], recall[idx], threshold))

    if candidates:
        best_precision, best_recall, best_threshold = max(candidates, key=lambda row: row[0])
        return float(best_threshold), float(best_recall), float(best_precision)

    f2_scores = (5 * precision * recall) / ((4 * precision) + recall + 1e-12)
    best_idx = int(np.nanargmax(f2_scores[:-1]))
    return float(thresholds[best_idx]), float(recall[best_idx]), float(precision[best_idx])


def threshold_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "n": int(n),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision_ppv": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "alerts": int(pred.sum()),
        "alerts_per_100_assessments": pred.sum() / n * 100,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
    }


def make_xgboost(y_train):
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / max(positive, 1)

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )


def clean_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def safe_roc_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def evaluate_model(name, model, feature_cols, val_df, y_val, test_df, y_test, horizon):
    val_prob = model.predict_proba(val_df[feature_cols])[:, 1]
    test_prob = model.predict_proba(test_df[feature_cols])[:, 1]
    threshold, val_recall, val_precision = select_threshold(y_val, val_prob)

    metrics = threshold_metrics(y_test, test_prob, threshold)
    metrics.update({
        "horizon_hours": horizon,
        "model": name,
        "n_features": len(feature_cols),
        "val_recall_at_threshold": val_recall,
        "val_precision_at_threshold": val_precision,
        "roc_auc": safe_roc_auc(y_test, test_prob),
        "pr_auc": average_precision_score(y_test, test_prob),
        "brier_score": brier_score_loss(y_test, test_prob),
        "base_rate": float(np.mean(y_test)),
    })
    return metrics, {"y_test": y_test, "test_prob": test_prob, "threshold": threshold}


def save_feature_importance(model, feature_cols, model_name, horizon):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_file = os.path.join(
        OUTPUT_PATH,
        f"h{horizon}_{clean_name(model_name)}_feature_importance.csv",
    )
    importance.to_csv(out_file, index=False)
    return importance


def make_calibration_deciles(y_true, y_prob, horizon, model_name):
    df = pd.DataFrame({"y_true": y_true, "predicted_risk": y_prob})
    df["risk_decile"] = pd.qcut(df["predicted_risk"], q=10, labels=False, duplicates="drop")
    out = (
        df.groupby("risk_decile", observed=False)
        .agg(
            n=("y_true", "size"),
            positives=("y_true", "sum"),
            mean_predicted_risk=("predicted_risk", "mean"),
            observed_event_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    out["risk_decile"] = out["risk_decile"].astype(int) + 1
    out["horizon_hours"] = horizon
    out["model"] = model_name
    return out


def plot_best_pr_curves(best_curve_results):
    plt.figure(figsize=(7, 6))
    for horizon, data in best_curve_results.items():
        precision, recall, _ = precision_recall_curve(data["y_test"], data["test_prob"])
        ap = average_precision_score(data["y_test"], data["test_prob"])
        base_rate = float(np.mean(data["y_test"]))
        plt.plot(
            recall,
            precision,
            label=f"{horizon}h {data['model']} (AP={ap:.3f}, base={base_rate:.3f})",
        )
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title("v0.4B Dynamic Vitals Best PR Curves by Horizon")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_best_pr_curves.png"), dpi=300)
    plt.close()


def plot_horizon_summary(best_by_horizon):
    plt.figure(figsize=(7, 5))
    plt.plot(best_by_horizon["horizon_hours"], best_by_horizon["pr_auc"], marker="o", label="PR-AUC")
    plt.plot(best_by_horizon["horizon_hours"], best_by_horizon["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Prediction horizon (hours)")
    plt.ylabel("Metric")
    plt.title("v0.4B Best Dynamic Performance by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_performance_by_horizon.png"), dpi=300)
    plt.close()


# %% Load landmark feature matrix

print("Loading v0.4B dynamic vitals feature matrix...")
df = pd.read_csv(FEATURE_FILE)

target_cols = [f"future_clabsi_{h}h" for h in HORIZONS]

print(f"Dataset shape: {df.shape}")
print(f"Unique stays: {df['stay_id'].nunique():,}")
print(f"Unique patients: {df['subject_id'].nunique():,}")
for target_col in target_cols:
    print(f"{target_col}: {df[target_col].sum():,} positives ({df[target_col].mean() * 100:.2f}%)")


# %% Define feature sets

meta_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "culture_time",
    "pragmatic_culture_time",
    "strict_culture_time",
    "strict_positive_orgs",
    "strict_label_reason",
    "landmark_time",
    "time_to_event_hours",
    "earliest_clabsi_time",
    "clabsi",
    "clabsi_pragmatic_v0_2",
    "clabsi_strict_organism",
    "strict_qualifying_culture_rows",
    "strict_clear_pathogen_rows",
    "strict_commensal_rows",
    "strict_distinct_commensal_times",
    "early_positive_culture",
    "pragmatic_downgraded_to_negative",
    "dwell_hours",
    "post_ref_dwell_hours",
]
meta_cols += [c for c in df.columns if c.startswith("prediction_window_end_")]
meta_cols += target_cols
meta_cols = [col for col in meta_cols if col in df.columns]

numeric_cols = [
    col for col in df.columns
    if col not in meta_cols and pd.api.types.is_numeric_dtype(df[col])
]

lab_prefixes = ("creatinine_", "hemoglobin_", "lactate_", "platelets_", "wbc_")
routine_lab_prefixes = ("creatinine_", "hemoglobin_", "platelets_", "wbc_")
vital_prefixes = (
    "heart_rate_",
    "respiratory_rate_",
    "spo2_",
    "temperature_c_",
    "sbp_",
    "dbp_",
    "map_",
    "fever_",
    "hypothermia_",
    "tachycardia_",
    "hypotension_",
    "tachypnea_",
)

context_cols = [
    col for col in ["anchor_age", "site_known", "landmark_hour", "dwell_at_landmark_hours"]
    if col in numeric_cols
]
lab_cols = [col for col in numeric_cols if col.startswith(lab_prefixes)]
routine_lab_cols = [col for col in numeric_cols if col.startswith(routine_lab_prefixes)]
vital_cols = [col for col in numeric_cols if col.startswith(vital_prefixes)]
non_lactate_cols = [
    col for col in numeric_cols
    if not col.startswith("lactate_")
]
static_only_cols = [
    col for col in numeric_cols
    if not col.startswith(lab_prefixes) and not col.startswith(vital_prefixes)
]

feature_sets = {
    "v0.4B dynamic full": numeric_cols,
    "v0.4B dynamic no site_known": [col for col in numeric_cols if col != "site_known"],
    "v0.4B dynamic no lactate": non_lactate_cols,
    "v0.4B dynamic vitals only": sorted(set(vital_cols + context_cols)),
    "v0.4B dynamic routine labs + vitals": sorted(set(routine_lab_cols + vital_cols + context_cols)),
    "v0.4B dynamic static only": static_only_cols,
}

print("")
print("Feature sets:")
for name, cols in feature_sets.items():
    print(f"  {name:<40} {len(cols)} features")


# %% Train/evaluate each horizon and feature-set variant

summary_rows = []
curve_results = {}
best_curve_results = {}
calibration_rows = []
threshold_rows = []
landmark_rows = []
stay_rows = []

for horizon in HORIZONS:
    target_col = f"future_clabsi_{horizon}h"
    print("")
    print("=" * 80)
    print(f"Evaluating horizon: {horizon}h")

    train_df, val_df, test_df = make_subject_level_splits(df, "subject_id", target_col)
    y_train = train_df[target_col].astype(int)
    y_val = val_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    print(
        f"  Train positives: {y_train.sum():,}/{len(y_train):,} "
        f"({y_train.mean() * 100:.2f}%)"
    )
    print(
        f"  Val positives:   {y_val.sum():,}/{len(y_val):,} "
        f"({y_val.mean() * 100:.2f}%)"
    )
    print(
        f"  Test positives:  {y_test.sum():,}/{len(y_test):,} "
        f"({y_test.mean() * 100:.2f}%)"
    )

    if y_train.sum() < 5 or y_val.sum() < 2 or y_test.sum() < 2:
        print("  Skipping horizon because too few positives are available for stable modeling.")
        continue

    horizon_curve_results = {}

    for model_name, feature_cols in feature_sets.items():
        print("")
        print(f"Training {model_name}...")
        model = make_xgboost(y_train)
        model.fit(train_df[feature_cols], y_train)

        metrics, curve_data = evaluate_model(
            model_name,
            model,
            feature_cols,
            val_df,
            y_val,
            test_df,
            y_test,
            horizon,
        )
        summary_rows.append(metrics)
        curve_results[(horizon, model_name)] = curve_data
        horizon_curve_results[model_name] = curve_data

        model_file = os.path.join(MODEL_PATH, f"h{horizon}_{clean_name(model_name)}.joblib")
        joblib.dump({
            "model": model,
            "feature_cols": feature_cols,
            "threshold": metrics["threshold"],
            "metrics": metrics,
            "target_col": target_col,
        }, model_file)

        importance = save_feature_importance(model, feature_cols, model_name, horizon)
        print(f"  PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        print("  Top 6 features:")
        print(importance.head(6).to_string(index=False))

    horizon_summary = pd.DataFrame([row for row in summary_rows if row["horizon_hours"] == horizon])
    best_row = horizon_summary.sort_values("pr_auc", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_curve = horizon_curve_results[best_name]
    best_curve["model"] = best_name
    best_curve_results[horizon] = best_curve

    best_bundle = joblib.load(os.path.join(MODEL_PATH, f"h{horizon}_{clean_name(best_name)}.joblib"))
    best_model = best_bundle["model"]
    best_cols = best_bundle["feature_cols"]
    best_threshold = float(best_bundle["threshold"])
    best_test_prob = best_model.predict_proba(test_df[best_cols])[:, 1]

    thresholds = np.round(np.arange(0.02, 0.52, 0.02), 2)
    thresholds = np.unique(np.append(thresholds, best_threshold))
    for threshold in np.sort(thresholds):
        row = threshold_metrics(y_test, best_test_prob, threshold)
        row.update({"horizon_hours": horizon, "model": best_name})
        threshold_rows.append(row)

    test_eval = test_df.copy()
    test_eval["predicted_risk"] = best_test_prob
    test_eval["alert"] = (test_eval["predicted_risk"] >= best_threshold).astype(int)

    for landmark_hour, group in test_eval.groupby("landmark_hour"):
        y_group = group[target_col].astype(int)
        row = threshold_metrics(y_group, group["predicted_risk"], best_threshold)
        row.update({
            "horizon_hours": horizon,
            "model": best_name,
            "landmark_hour": landmark_hour,
            "future_positive_rows": int(y_group.sum()),
            "base_rate": float(y_group.mean()),
            "roc_auc": safe_roc_auc(y_group, group["predicted_risk"]),
            "pr_auc": average_precision_score(y_group, group["predicted_risk"]) if y_group.sum() > 0 else np.nan,
        })
        landmark_rows.append(row)

    stay_eval = (
        test_eval
        .groupby("stay_id")
        .agg(
            subject_id=("subject_id", "first"),
            max_predicted_risk=("predicted_risk", "max"),
            any_future_clabsi=(target_col, "max"),
            any_alert=("alert", "max"),
            n_landmark_rows=("landmark_hour", "size"),
            first_landmark=("landmark_hour", "min"),
            last_landmark=("landmark_hour", "max"),
        )
        .reset_index()
    )
    stay_metrics = threshold_metrics(
        stay_eval["any_future_clabsi"].astype(int),
        stay_eval["max_predicted_risk"],
        best_threshold,
    )
    stay_metrics.update({
        "horizon_hours": horizon,
        "model": best_name,
        "roc_auc": safe_roc_auc(stay_eval["any_future_clabsi"], stay_eval["max_predicted_risk"]),
        "pr_auc": average_precision_score(stay_eval["any_future_clabsi"], stay_eval["max_predicted_risk"]),
        "base_rate": float(stay_eval["any_future_clabsi"].mean()),
    })
    stay_rows.append(stay_metrics)

    calibration_rows.append(
        make_calibration_deciles(y_test, best_test_prob, horizon, best_name)
    )

    top_cols = [
        "subject_id", "hadm_id", "stay_id", "landmark_hour", "predicted_risk", "alert",
        target_col, "time_to_event_hours", "strict_positive_orgs", "anchor_age",
        "site_known", "dwell_at_landmark_hours", "temperature_c_max_24h",
        "heart_rate_max_24h", "respiratory_rate_max_24h", "spo2_min_24h",
        "wbc_trend_48h", "platelets_last_48h", "creatinine_last_48h",
    ]
    top_cols = [col for col in top_cols if col in test_eval.columns]
    top_file = os.path.join(OUTPUT_PATH, f"v0_4b_h{horizon}_top_100_dynamic_rows.csv")
    test_eval.sort_values("predicted_risk", ascending=False).head(100)[top_cols].to_csv(
        top_file,
        index=False,
    )


# %% Save outputs and plots

summary = pd.DataFrame(summary_rows).sort_values(["horizon_hours", "pr_auc"], ascending=[True, False])
summary_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_model_comparison.csv")
summary.to_csv(summary_file, index=False)

best_by_horizon = (
    summary
    .sort_values(["horizon_hours", "pr_auc"], ascending=[True, False])
    .groupby("horizon_hours", as_index=False)
    .head(1)
    .reset_index(drop=True)
)
best_by_horizon_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_best_by_horizon.csv")
best_by_horizon.to_csv(best_by_horizon_file, index=False)

threshold_table = pd.DataFrame(threshold_rows)
threshold_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_threshold_table.csv")
threshold_table.to_csv(threshold_file, index=False)

landmark_perf = pd.DataFrame(landmark_rows).sort_values(["horizon_hours", "landmark_hour"])
landmark_perf_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_performance_by_landmark.csv")
landmark_perf.to_csv(landmark_perf_file, index=False)

stay_summary = pd.DataFrame(stay_rows)
stay_summary_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_stay_level_summary.csv")
stay_summary.to_csv(stay_summary_file, index=False)

calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
calibration_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_calibration_deciles.csv")
calibration.to_csv(calibration_file, index=False)

plot_best_pr_curves(best_curve_results)
plot_horizon_summary(best_by_horizon)


# %% Optional SHAP for overall best model

try:
    import shap

    overall_best = summary.sort_values("pr_auc", ascending=False).iloc[0]
    horizon = int(overall_best["horizon_hours"])
    model_name = overall_best["model"]
    target_col = f"future_clabsi_{horizon}h"
    print("")
    print(f"Generating SHAP summary for overall best model: {model_name}, {horizon}h...")

    _, _, test_df = make_subject_level_splits(df, "subject_id", target_col)
    bundle = joblib.load(os.path.join(MODEL_PATH, f"h{horizon}_{clean_name(model_name)}.joblib"))
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    shap_sample = test_df[feature_cols].sample(n=min(1000, len(test_df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.tight_layout()
    shap_file = os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_best_model_shap.png")
    plt.savefig(shap_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot: {shap_file}")
except ImportError:
    print("")
    print("SHAP not installed. Skipping SHAP plot.")


# %% Manifest and console summary

manifest_rows = [
    ("Model comparison", summary_file),
    ("Best by horizon", best_by_horizon_file),
    ("Threshold table", threshold_file),
    ("Performance by landmark", landmark_perf_file),
    ("Stay-level summary", stay_summary_file),
    ("Calibration deciles", calibration_file),
    ("Best PR curves by horizon", os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_best_pr_curves.png")),
    ("Performance by horizon plot", os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_performance_by_horizon.png")),
    ("Best model SHAP", os.path.join(PLOT_PATH, "v0_4b_dynamic_vitals_best_model_shap.png")),
]
manifest_file = os.path.join(OUTPUT_PATH, "v0_4b_dynamic_vitals_output_manifest.csv")
pd.DataFrame(manifest_rows, columns=["output", "path"]).to_csv(manifest_file, index=False)

print("")
print("v0.4B dynamic vitals best model by horizon:")
display_cols = [
    "horizon_hours", "model", "roc_auc", "pr_auc", "brier_score", "base_rate",
    "threshold", "recall_sensitivity", "specificity", "precision_ppv",
    "alerts_per_100_assessments", "false_alerts_per_true_positive",
]
print(best_by_horizon[display_cols].round(4).to_string(index=False))

print("")
print("Saved outputs:")
for label, path in manifest_rows:
    print(f"  {label}: {path}")

print("")
print("Modeling 05 v0.4B Landmark Dynamic Vitals complete.")

