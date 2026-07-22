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
    print("Missing modeling dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost matplotlib joblib")
    sys.exit(1)


PROJECT_PATH = r"C:\path\to\CVCML"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_3a")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 5 (v0.3a Strict Organism Sensitivity)")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_features_v0_3a.csv")
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


def threshold_table(y_true, y_prob, thresholds):
    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        rows.append({
            "threshold": threshold,
            "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "precision_ppv": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "alerts": int(pred.sum()),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        })
    return pd.DataFrame(rows)


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


def evaluate_model(name, model, X_val, y_val, X_test, y_test):
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    threshold, val_recall, val_precision = select_threshold(y_val, val_prob)

    test_pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    metrics = {
        "model": name,
        "threshold": threshold,
        "val_recall_at_threshold": val_recall,
        "val_precision_at_threshold": val_precision,
        "roc_auc": roc_auc_score(y_test, test_prob),
        "pr_auc": average_precision_score(y_test, test_prob),
        "brier_score": brier_score_loss(y_test, test_prob),
        "recall_sensitivity": recall_score(y_test, test_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "precision_ppv": precision_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "alerts": int(test_pred.sum()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, {
        "y_test": y_test,
        "test_prob": test_prob,
        "test_pred": test_pred,
    }


def clean_model_name(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(".", "_")
    )


def save_feature_importance(model, feature_cols, model_name):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_file = os.path.join(OUTPUT_PATH, f"{clean_model_name(model_name)}_feature_importance.csv")
    importance.to_csv(out_file, index=False)
    return importance


def plot_curves(curve_results):
    plt.figure(figsize=(7, 6))
    for name, data in curve_results.items():
        fpr, tpr, _ = roc_curve(data["y_test"], data["test_prob"])
        auc = roc_auc_score(data["y_test"], data["test_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("v0.3a Strict Organism ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_3a_strict_organism_roc_curves.png"), dpi=300)
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
    plt.title("v0.3a Strict Organism Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "v0_3a_strict_organism_pr_curves.png"), dpi=300)
    plt.close()


def make_feature_set(base_cols, remove_cols=None):
    remove_cols = set(remove_cols or [])
    return [col for col in base_cols if col not in remove_cols]


# %% Load v0.3a feature matrix

print("Loading v0.3a strict-organism feature matrix...")
df = pd.read_csv(FEATURE_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Strict CLABSI positive: {df['clabsi'].sum():,} ({df['clabsi'].mean() * 100:.1f}%)")
print(f"Strict CLABSI negative: {(df['clabsi'] == 0).sum():,}")

meta_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "culture_time",
    "pragmatic_culture_time",
    "strict_culture_time",
    "ref_time",
    "window_start",
    "earliest_clabsi_time",
    "post_ref_dwell_hours",
    "dwell_hours",
    "early_positive_culture",
    "clabsi_pragmatic_v0_2",
    "clabsi_strict_organism",
    "strict_positive_orgs",
    "strict_label_reason",
    "strict_qualifying_culture_rows",
    "strict_clear_pathogen_rows",
    "strict_commensal_rows",
    "strict_distinct_commensal_times",
    "pragmatic_downgraded_to_negative",
]
meta_cols = [col for col in meta_cols if col in df.columns]

target_col = "clabsi"
base_feature_cols = [col for col in df.columns if col not in meta_cols + [target_col]]

routine_lab_cols = [
    col for col in base_feature_cols
    if col.startswith(("creatinine_", "hemoglobin_", "platelets_", "wbc_"))
]
non_lactate_cols = [
    col for col in base_feature_cols
    if not col.startswith("lactate_") and col != "lactate_measured"
]

print("")
print(f"Feature columns after audit exclusion: {len(base_feature_cols)}")
print(f"Includes dwell_at_ref_hours: {'dwell_at_ref_hours' in base_feature_cols}")
print(f"Includes site_known:         {'site_known' in base_feature_cols}")
print(f"Includes audit leakage cols: {bool(set(meta_cols) & set(base_feature_cols))}")


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
        f"  {split_name:<5}: {len(split_df):,} stays | "
        f"{split_df['subject_id'].nunique():,} patients | "
        f"{y_split.sum():,} positive ({y_split.mean() * 100:.1f}%)"
    )


# %% Define model variants

feature_sets = {
    "v0.3a strict full": make_feature_set(base_feature_cols),
    "v0.3a strict no site_known": make_feature_set(base_feature_cols, remove_cols=["site_known"]),
    "v0.3a strict no lactate": make_feature_set(
        base_feature_cols,
        remove_cols=set(base_feature_cols) - set(non_lactate_cols),
    ),
    "v0.3a strict routine labs only": make_feature_set(
        base_feature_cols,
        remove_cols=set(base_feature_cols) - set(routine_lab_cols + ["dwell_at_ref_hours", "site_known", "anchor_age"]),
    ),
    "v0.3a strict no dwell no site": make_feature_set(
        base_feature_cols,
        remove_cols=["dwell_at_ref_hours", "site_known"],
    ),
}

print("")
print("Model variants:")
for name, cols in feature_sets.items():
    print(f"  {name:<34} {len(cols)} features")


# %% Train/evaluate variants

summary_rows = []
curve_results = {}

for model_name, feature_cols in feature_sets.items():
    print("")
    print(f"Training {model_name}...")
    model = make_xgboost(y_train)
    model.fit(train_df[feature_cols], y_train)

    metrics, curve_data = evaluate_model(
        model_name,
        model,
        val_df[feature_cols],
        y_val,
        test_df[feature_cols],
        y_test,
    )
    summary_rows.append(metrics)
    curve_results[model_name] = curve_data

    model_file = os.path.join(MODEL_PATH, f"{clean_model_name(model_name)}.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "threshold": metrics["threshold"],
        "metrics": metrics,
    }, model_file)

    importance = save_feature_importance(model, feature_cols, model_name)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    print("  Top 8 features:")
    print(importance.head(8).to_string(index=False))


# %% Save outputs

summary = pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False)
summary_file = os.path.join(OUTPUT_PATH, "v0_3a_strict_organism_model_comparison.csv")
summary.to_csv(summary_file, index=False)

best_model_name = summary.iloc[0]["model"]
best_curve = curve_results[best_model_name]
thresholds = np.round(np.arange(0.05, 0.55, 0.05), 2)
thresholds = np.unique(np.append(thresholds, summary.iloc[0]["threshold"]))
thresholds = np.sort(thresholds)
threshold_out = threshold_table(best_curve["y_test"], best_curve["test_prob"], thresholds)
threshold_file = os.path.join(OUTPUT_PATH, "v0_3a_strict_organism_threshold_table.csv")
threshold_out.to_csv(threshold_file, index=False)

plot_curves(curve_results)

print("")
print("v0.3a strict-organism model comparison:")
display_cols = [
    "model",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "threshold",
    "recall_sensitivity",
    "specificity",
    "precision_ppv",
    "f1",
    "alerts",
    "false_negative",
    "false_positive",
]
print(summary[display_cols].round(4).to_string(index=False))

print("")
print(f"Best model by PR-AUC: {best_model_name}")
print("")
print("Saved outputs:")
print(f"  Model comparison: {summary_file}")
print(f"  Threshold table:  {threshold_file}")
print(f"  Models:           {MODEL_PATH}")
print(f"  Plots:            {PLOT_PATH}")


# %% Optional SHAP for best model

try:
    import shap

    print("")
    print(f"Generating SHAP summary for {best_model_name}...")
    best_bundle = joblib.load(os.path.join(MODEL_PATH, f"{clean_model_name(best_model_name)}.joblib"))
    best_model = best_bundle["model"]
    best_cols = best_bundle["feature_cols"]

    shap_sample = test_df[best_cols].sample(n=min(1000, len(test_df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.tight_layout()
    shap_file = os.path.join(PLOT_PATH, "v0_3a_strict_organism_best_model_shap.png")
    plt.savefig(shap_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot: {shap_file}")
except ImportError:
    print("")
    print("SHAP not installed. Skipping SHAP plot.")

print("")
print("Modeling 03 v0.3a Strict Sensitivity complete.")

