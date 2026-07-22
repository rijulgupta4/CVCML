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
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing modeling dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost matplotlib joblib")
    print("")
    print("Optional for explainability:")
    print("  pip install shap")
    sys.exit(1)


PROJECT_PATH = r"C:\path\to\CVCML"
DATA_PATH = os.path.join(PROJECT_PATH, "data")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_features.csv")
RANDOM_STATE = 42


# %% Helper functions

def select_threshold_for_clinical_alert(y_true, y_prob, min_recall=0.80):
    """Pick the highest-precision threshold that reaches the target recall."""
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


def train_xgboost(name, X_train, y_train):
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    model = XGBClassifier(
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
        n_jobs=-1,
    )

    print("")
    print(f"Training {name}...")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_split, y_split, threshold):
    X_train, X_val, X_test = X_split
    y_train, y_val, y_test = y_split

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0

    metrics = {
        "model": name,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "positive_rate_train": y_train.mean(),
        "positive_rate_val": y_val.mean(),
        "positive_rate_test": y_test.mean(),
        "roc_auc": roc_auc_score(y_test, test_prob),
        "pr_auc": average_precision_score(y_test, test_prob),
        "threshold": threshold,
        "recall_sensitivity": recall_score(y_test, test_pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    curve_data = {
        "test_prob": test_prob,
        "test_pred": test_pred,
        "y_test": y_test,
    }
    return metrics, curve_data


def plot_curves(results_by_model):
    plt.figure(figsize=(7, 6))
    for name, data in results_by_model.items():
        fpr, tpr, _ = roc_curve(data["y_test"], data["test_prob"])
        auc = roc_auc_score(data["y_test"], data["test_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CLABSI Prediction ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "roc_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, data in results_by_model.items():
        precision, recall, _ = precision_recall_curve(data["y_test"], data["test_prob"])
        ap = average_precision_score(data["y_test"], data["test_prob"])
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    base_rate = next(iter(results_by_model.values()))["y_test"].mean()
    plt.axhline(base_rate, linestyle="--", color="gray", linewidth=1, label=f"Base rate={base_rate:.3f}")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title("CLABSI Prediction Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "precision_recall_curves.png"), dpi=300)
    plt.close()


def save_feature_importance(model, feature_names, model_name):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_file = os.path.join(OUTPUT_PATH, f"{model_name.lower().replace(' ', '_')}_feature_importance.csv")
    importance.to_csv(out_file, index=False)
    return importance


# %% Load model-ready dataset

print("Loading feature matrix...")
df = pd.read_csv(FEATURE_FILE)

print(f"Dataset shape: {df.shape}")
print(f"CLABSI positive: {df['clabsi'].sum():,} ({df['clabsi'].mean() * 100:.1f}%)")
print(f"CLABSI negative: {(df['clabsi'] == 0).sum():,} ({(1 - df['clabsi'].mean()) * 100:.1f}%)")

meta_cols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "culture_time",
    "ref_time",
    "window_start",
]
meta_cols = [col for col in meta_cols if col in df.columns]

target_col = "clabsi"
all_feature_cols = [col for col in df.columns if col not in meta_cols + [target_col]]

lab_prefixes = ("creatinine_", "hemoglobin_", "lactate_", "platelets_", "wbc_")
lab_cols = [col for col in all_feature_cols if col.startswith(lab_prefixes)]
lab_cols += [col for col in all_feature_cols if col == "lactate_measured"]

static_feature_cols = [col for col in all_feature_cols if col not in lab_cols]
improved_feature_cols = all_feature_cols

print("")
print("Feature sets:")
print(f"  Baseline static features: {len(static_feature_cols)}")
print(f"  Improved static + labs:   {len(improved_feature_cols)}")


# %% Split once so both models are compared on identical patients

train_df, temp_df = train_test_split(
    df,
    test_size=0.40,
    stratify=df[target_col],
    random_state=RANDOM_STATE,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df[target_col],
    random_state=RANDOM_STATE,
)

y_train = train_df[target_col].astype(int)
y_val = val_df[target_col].astype(int)
y_test = test_df[target_col].astype(int)

print("")
print("Split sizes:")
print(f"  Train: {len(train_df):,} | positives: {y_train.sum():,} ({y_train.mean() * 100:.1f}%)")
print(f"  Val:   {len(val_df):,} | positives: {y_val.sum():,} ({y_val.mean() * 100:.1f}%)")
print(f"  Test:  {len(test_df):,} | positives: {y_test.sum():,} ({y_test.mean() * 100:.1f}%)")


# %% Train and evaluate models

model_specs = {
    "Baseline XGBoost": static_feature_cols,
    "Improved XGBoost": improved_feature_cols,
}

summary_rows = []
curve_results = {}

for model_name, feature_cols in model_specs.items():
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    model = train_xgboost(model_name, X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_recall, val_precision = select_threshold_for_clinical_alert(
        y_val,
        val_prob,
        min_recall=0.80,
    )
    print(f"  Selected threshold: {threshold:.4f}")
    print(f"  Validation recall at threshold: {val_recall:.3f}")
    print(f"  Validation precision at threshold: {val_precision:.3f}")

    metrics, curve_data = evaluate_model(
        model_name,
        model,
        (X_train, X_val, X_test),
        (y_train, y_val, y_test),
        threshold,
    )
    summary_rows.append(metrics)
    curve_results[model_name] = curve_data

    model_file = os.path.join(MODEL_PATH, f"{model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "metrics": metrics,
    }, model_file)

    importance = save_feature_importance(model, feature_cols, model_name)
    print(f"  Saved model: {model_file}")
    print("  Top 10 features:")
    print(importance.head(10).to_string(index=False))


# %% Save comparison tables and plots

summary = pd.DataFrame(summary_rows)
summary_file = os.path.join(OUTPUT_PATH, "model_comparison.csv")
summary.to_csv(summary_file, index=False)

plot_curves(curve_results)

print("")
print("Model comparison:")
display_cols = [
    "model",
    "roc_auc",
    "pr_auc",
    "threshold",
    "recall_sensitivity",
    "specificity",
    "precision_ppv",
    "f1",
    "false_positive",
    "false_negative",
]
print(summary[display_cols].round(4).to_string(index=False))

print("")
print("Saved outputs:")
print(f"  Metrics: {summary_file}")
print(f"  Models:  {MODEL_PATH}")
print(f"  Plots:   {PLOT_PATH}")


# %% Optional SHAP scaffold

try:
    import shap

    print("")
    print("Generating SHAP summary for Improved XGBoost...")
    improved_bundle = joblib.load(os.path.join(MODEL_PATH, "improved_xgboost.joblib"))
    improved_model = improved_bundle["model"]
    improved_cols = improved_bundle["feature_cols"]

    shap_sample = test_df[improved_cols].sample(
        n=min(1000, len(test_df)),
        random_state=RANDOM_STATE,
    )
    explainer = shap.TreeExplainer(improved_model)
    shap_values = explainer.shap_values(shap_sample)

    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.tight_layout()
    shap_file = os.path.join(PLOT_PATH, "improved_xgboost_shap_summary.png")
    plt.savefig(shap_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot: {shap_file}")
except ImportError:
    print("")
    print("SHAP not installed. Skipping explainability plot for now.")
    print("Install later with: pip install shap")


print("")
print("Modeling 03.py complete.")
print("Next phase: build an hourly/dynamic feature table from chartevents + labevents.")

