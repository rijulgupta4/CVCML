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
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 7 (v0.4 Landmark Dynamic Model)")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4.csv")
RANDOM_STATE = 42
MIN_RECALL_TARGET = 0.80
XGB_N_JOBS = 2


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
    scale_pos_weight = negative / positive

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
    )


def plot_curves(curve_results):
    plt.figure(figsize=(7, 6))
    for name, data in curve_results.items():
        fpr, tpr, _ = roc_curve(data["y_test"], data["test_prob"])
        auc = roc_auc_score(data["y_test"], data["test_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("v0.4 Landmark Dynamic ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_roc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, data in curve_results.items():
        precision, recall, _ = precision_recall_curve(data["y_test"], data["test_prob"])
        ap = average_precision_score(data["y_test"], data["test_prob"])
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    base_rate = next(iter(curve_results.values()))["y_test"].mean()
    plt.axhline(base_rate, linestyle="--", color="gray", linewidth=1, label=f"Base rate={base_rate:.3f}")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title("v0.4 Landmark Dynamic Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_pr_curves.png"), dpi=300)
    plt.close()


def save_feature_importance(model, feature_cols, model_name):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_file = os.path.join(OUTPUT_PATH, f"{clean_name(model_name)}_feature_importance.csv")
    importance.to_csv(out_file, index=False)
    return importance


def make_calibration_deciles(y_true, y_prob):
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
    return out


def plot_landmark_performance(landmark_perf):
    plt.figure(figsize=(7, 5))
    plt.plot(landmark_perf["landmark_hour"], landmark_perf["pr_auc"], marker="o", label="PR-AUC")
    plt.plot(landmark_perf["landmark_hour"], landmark_perf["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Landmark hour")
    plt.ylabel("Metric")
    plt.title("v0.4 Dynamic Performance by Landmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_4_dynamic_performance_by_landmark.png"), dpi=300)
    plt.close()


def evaluate_model(name, model, feature_cols, val_df, y_val, test_df, y_test):
    val_prob = model.predict_proba(val_df[feature_cols])[:, 1]
    test_prob = model.predict_proba(test_df[feature_cols])[:, 1]
    threshold, val_recall, val_precision = select_threshold(y_val, val_prob)

    metrics = threshold_metrics(y_test, test_prob, threshold)
    metrics.update({
        "model": name,
        "val_recall_at_threshold": val_recall,
        "val_precision_at_threshold": val_precision,
        "roc_auc": roc_auc_score(y_test, test_prob),
        "pr_auc": average_precision_score(y_test, test_prob),
        "brier_score": brier_score_loss(y_test, test_prob),
        "base_rate": float(np.mean(y_test)),
    })
    return metrics, {"y_test": y_test, "test_prob": test_prob, "threshold": threshold}


# %% Load landmark feature matrix

print("Loading v0.4 landmark dynamic feature matrix...")
df = pd.read_csv(FEATURE_FILE)

target_col = "future_clabsi"
print(f"Dataset shape: {df.shape}")
print(f"Future-positive rows: {df[target_col].sum():,} ({df[target_col].mean() * 100:.2f}%)")
print(f"Unique stays: {df['stay_id'].nunique():,}")
print(f"Unique patients: {df['subject_id'].nunique():,}")

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
    "lookback_start",
    "prediction_window_end",
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
meta_cols = [col for col in meta_cols if col in df.columns]
base_feature_cols = [col for col in df.columns if col not in meta_cols + [target_col]]

lab_prefixes = ("creatinine_", "hemoglobin_", "lactate_", "platelets_", "wbc_")
routine_lab_prefixes = ("creatinine_", "hemoglobin_", "platelets_", "wbc_")

routine_cols = [col for col in base_feature_cols if col.startswith(routine_lab_prefixes)]
non_lactate_cols = [col for col in base_feature_cols if not col.startswith("lactate_") and col != "lactate_measured"]
static_dynamic_cols = [
    col for col in base_feature_cols
    if not col.startswith(lab_prefixes)
]

feature_sets = {
    "v0.4 dynamic full": base_feature_cols,
    "v0.4 dynamic no site_known": [col for col in base_feature_cols if col != "site_known"],
    "v0.4 dynamic no lactate": non_lactate_cols,
    "v0.4 dynamic routine labs only": sorted(set(routine_cols + ["anchor_age", "site_known", "landmark_hour", "dwell_at_landmark_hours"])),
    "v0.4 dynamic static only": static_dynamic_cols,
}

print("")
print("Feature sets:")
for name, cols in feature_sets.items():
    print(f"  {name:<34} {len(cols)} features")


# %% Patient-level split

train_df, val_df, test_df = make_subject_level_splits(df, "subject_id", target_col)
y_train = train_df[target_col].astype(int)
y_val = val_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

print("")
print("Patient-level split:")
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    y_split = split_df[target_col]
    print(
        f"  {split_name:<5}: {len(split_df):,} rows | "
        f"{split_df['stay_id'].nunique():,} stays | "
        f"{split_df['subject_id'].nunique():,} patients | "
        f"{y_split.sum():,} future-positive rows ({y_split.mean() * 100:.2f}%)"
    )


# %% Train and evaluate dynamic variants

summary_rows = []
curve_results = {}

for model_name, feature_cols in feature_sets.items():
    print("")
    print(f"Training {model_name}...")
    model = make_xgboost(y_train)
    model.fit(train_df[feature_cols], y_train)

    metrics, curve_data = evaluate_model(model_name, model, feature_cols, val_df, y_val, test_df, y_test)
    summary_rows.append(metrics)
    curve_results[model_name] = curve_data

    model_file = os.path.join(MODEL_PATH, f"{clean_name(model_name)}.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "threshold": metrics["threshold"],
        "metrics": metrics,
    }, model_file)

    importance = save_feature_importance(model, feature_cols, model_name)
    print(f"  Row PR-AUC: {metrics['pr_auc']:.4f} | Row ROC-AUC: {metrics['roc_auc']:.4f}")
    print("  Top 8 features:")
    print(importance.head(8).to_string(index=False))


# %% Save model comparison and plots

summary = pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False)
summary_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_model_comparison.csv")
summary.to_csv(summary_file, index=False)

plot_curves(curve_results)

best_model_name = summary.iloc[0]["model"]
best_bundle = joblib.load(os.path.join(MODEL_PATH, f"{clean_name(best_model_name)}.joblib"))
best_model = best_bundle["model"]
best_cols = best_bundle["feature_cols"]
best_threshold = float(best_bundle["threshold"])
best_test_prob = best_model.predict_proba(test_df[best_cols])[:, 1]

thresholds = np.round(np.arange(0.02, 0.52, 0.02), 2)
thresholds = np.unique(np.append(thresholds, best_threshold))
thresholds = np.sort(thresholds)
threshold_table = pd.DataFrame([threshold_metrics(y_test, best_test_prob, threshold) for threshold in thresholds])
threshold_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_threshold_table.csv")
threshold_table.to_csv(threshold_file, index=False)


# %% Landmark-specific and stay-level summaries

landmark_rows = []
test_eval = test_df.copy()
test_eval["predicted_risk"] = best_test_prob
test_eval["alert"] = (test_eval["predicted_risk"] >= best_threshold).astype(int)

for landmark_hour, group in test_eval.groupby("landmark_hour"):
    y_group = group[target_col].astype(int)
    if y_group.nunique() < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_group, group["predicted_risk"])
    pr_auc = average_precision_score(y_group, group["predicted_risk"]) if y_group.sum() > 0 else np.nan
    row = threshold_metrics(y_group, group["predicted_risk"], best_threshold)
    row.update({
        "landmark_hour": landmark_hour,
        "future_positive_rows": int(y_group.sum()),
        "base_rate": float(y_group.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    })
    landmark_rows.append(row)

landmark_perf = pd.DataFrame(landmark_rows).sort_values("landmark_hour")
landmark_perf_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_performance_by_landmark.csv")
landmark_perf.to_csv(landmark_perf_file, index=False)
plot_landmark_performance(landmark_perf)

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
    "model": best_model_name,
    "roc_auc": roc_auc_score(stay_eval["any_future_clabsi"], stay_eval["max_predicted_risk"]),
    "pr_auc": average_precision_score(stay_eval["any_future_clabsi"], stay_eval["max_predicted_risk"]),
    "base_rate": float(stay_eval["any_future_clabsi"].mean()),
})
stay_summary = pd.DataFrame([stay_metrics])
stay_summary_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_stay_level_summary.csv")
stay_summary.to_csv(stay_summary_file, index=False)

top_dynamic_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_top_100_rows.csv")
top_cols = [
    "subject_id", "hadm_id", "stay_id", "landmark_hour", "predicted_risk", "alert", target_col,
    "time_to_event_hours", "strict_positive_orgs", "cvc_type_Dialysis Catheter",
    "cvc_type_PICC Line", "anchor_age", "site_known", "dwell_at_landmark_hours",
    "lactate_measured", "lactate_mean", "lactate_last", "wbc_trend", "platelets_trend",
    "creatinine_trend",
]
top_cols = [col for col in top_cols if col in test_eval.columns]
test_eval.sort_values("predicted_risk", ascending=False).head(100)[top_cols].to_csv(top_dynamic_file, index=False)


# %% Calibration deciles for best dynamic model

calibration = make_calibration_deciles(y_test, best_test_prob)
calibration_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_calibration_deciles.csv")
calibration.to_csv(calibration_file, index=False)

plt.figure(figsize=(6, 5))
plt.plot(calibration["mean_predicted_risk"], calibration["observed_event_rate"], marker="o", linewidth=2)
max_axis = max(calibration["mean_predicted_risk"].max(), calibration["observed_event_rate"].max()) * 1.15
plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="gray", linewidth=1)
plt.xlabel("Mean predicted risk")
plt.ylabel("Observed future CLABSI row rate")
plt.title("v0.4 Dynamic Calibration by Risk Decile")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_calibration_deciles.png"), dpi=300)
plt.close()


# %% Optional SHAP for best model

try:
    import shap

    print("")
    print(f"Generating SHAP summary for {best_model_name}...")
    shap_sample = test_df[best_cols].sample(n=min(1000, len(test_df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.tight_layout()
    shap_file = os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_best_model_shap.png")
    plt.savefig(shap_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot: {shap_file}")
except ImportError:
    print("")
    print("SHAP not installed. Skipping SHAP plot.")


# %% Manifest and console summary

manifest_rows = [
    ("Model comparison", summary_file),
    ("Threshold table", threshold_file),
    ("Performance by landmark", landmark_perf_file),
    ("Stay-level summary", stay_summary_file),
    ("Top 100 dynamic rows", top_dynamic_file),
    ("Calibration deciles", calibration_file),
    ("PR curves", os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_pr_curves.png")),
    ("ROC curves", os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_roc_curves.png")),
    ("Performance by landmark plot", os.path.join(PLOT_PATH, "v0_4_dynamic_performance_by_landmark.png")),
    ("Calibration plot", os.path.join(PLOT_PATH, "v0_4_landmark_dynamic_calibration_deciles.png")),
]
manifest_file = os.path.join(OUTPUT_PATH, "v0_4_landmark_dynamic_output_manifest.csv")
pd.DataFrame(manifest_rows, columns=["output", "path"]).to_csv(manifest_file, index=False)

print("")
print("v0.4 landmark dynamic model comparison:")
display_cols = [
    "model", "roc_auc", "pr_auc", "brier_score", "base_rate", "threshold",
    "recall_sensitivity", "specificity", "precision_ppv", "alerts",
    "alerts_per_100_assessments", "false_alerts_per_true_positive",
]
print(summary[display_cols].round(4).to_string(index=False))

print("")
print("Best model by row-level PR-AUC:", best_model_name)
print("Stay-level summary for best model:")
print(stay_summary[["roc_auc", "pr_auc", "base_rate", "recall_sensitivity", "specificity", "precision_ppv", "alerts"]].round(4).to_string(index=False))

print("")
print("Saved outputs:")
for label, path in manifest_rows:
    print(f"  {label}: {path}")

print("")
print("Modeling 05 v0.4 Landmark Dynamic complete.")

