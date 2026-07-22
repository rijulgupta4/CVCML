# %% Imports and paths

import sys
from pathlib import Path

try:
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing v0.5 Run 19 dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost joblib matplotlib")
    sys.exit(1)


PROJECT_PATH = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 19 (v0.5 Calibration Review Policy)"
PLOT_PATH = OUTPUT_PATH / "plots"
MODEL_PATH = OUTPUT_PATH / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_PATH / "v0_5_run18_development_features.csv"
TARGET_COL = "future_strict_cvc_bsi_proxy_7d"
RANDOM_STATE = 42
GENERATE_PLOTS = False

TRAIN_CORE_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013"]
CALIBRATION_YEAR_GROUPS = ["2014 - 2016"]
VALIDATION_YEAR_GROUPS = ["2017 - 2019"]
LOCKBOX_YEAR_GROUP = "2020 - 2022"

LAB_PREFIXES = ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"]


# %% Helpers

def clip_prob(prob):
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)


def logit(prob):
    prob = clip_prob(prob)
    return np.log(prob / (1 - prob))


def sigmoid(score):
    score = np.clip(np.asarray(score, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-score))


def safe_roc_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def calibration_intercept_slope(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    x = np.clip(logit(y_prob), -20, 20)
    intercept = float(logit([y_true.mean()])[0])
    slope = 1.0

    for _ in range(100):
        eta = intercept + slope * x
        p = sigmoid(eta)
        diff = p - y_true
        weight = np.clip(p * (1 - p), 1e-8, None)

        grad0 = float(diff.sum())
        grad1 = float((diff * x).sum())
        h00 = float(weight.sum()) + 1e-6
        h01 = float((weight * x).sum())
        h11 = float((weight * x * x).sum()) + 1e-6
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break

        step0 = (h11 * grad0 - h01 * grad1) / det
        step1 = (-h01 * grad0 + h00 * grad1) / det
        intercept -= step0
        slope -= step1

        if max(abs(step0), abs(step1)) < 1e-6:
            break

    return float(intercept), float(slope)


def brier_skill_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    prevalence = y_true.mean()
    model_brier = brier_score_loss(y_true, y_prob)
    reference_prob = np.repeat(prevalence, len(y_true))
    reference_brier = brier_score_loss(y_true, reference_prob)
    if reference_brier == 0:
        return np.nan, model_brier, reference_brier
    return 1 - (model_brier / reference_brier), model_brier, reference_brier


def expected_observed_ratio(y_true, y_prob):
    observed = np.asarray(y_true).sum()
    expected = np.asarray(y_prob).sum()
    return expected / observed if observed else np.nan


def evaluate_predictions(df, split_name, score_name, probability_col):
    y_true = df[TARGET_COL].astype(int).to_numpy()
    y_prob = df[probability_col].to_numpy()
    prevalence = float(np.mean(y_true))
    bss, brier, brier_reference = brier_skill_score(y_true, y_prob)
    cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_prob)

    return {
        "split": split_name,
        "score_name": score_name,
        "rows": int(len(df)),
        "positive_rows": int(np.sum(y_true)),
        "prevalence": prevalence,
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "pr_auc_lift_over_prevalence": average_precision_score(y_true, y_prob) / prevalence if prevalence else np.nan,
        "brier_score": brier,
        "brier_reference_prevalence": brier_reference,
        "brier_skill_score": bss,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "expected_observed_ratio": expected_observed_ratio(y_true, y_prob),
    }


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])


def make_topk_table(df, score_name, probability_col):
    rows = []
    ranked = df.sort_values(probability_col, ascending=False).reset_index(drop=True)
    total_positive_rows = int(ranked[TARGET_COL].sum())
    total_positive_episodes = int(ranked.loc[ranked[TARGET_COL].eq(1), "episode_id"].nunique())

    for pct in [1, 2, 5, 10]:
        n_reviewed = max(1, int(np.ceil(len(ranked) * pct / 100)))
        flagged = ranked.head(n_reviewed)
        rows.append(review_summary(flagged, ranked, score_name, f"top_{pct}_percent_rows"))

    for n_reviewed in [25, 50, 100, 250, 500]:
        n_reviewed = min(n_reviewed, len(ranked))
        flagged = ranked.head(n_reviewed)
        rows.append(review_summary(flagged, ranked, score_name, f"top_{n_reviewed}_rows"))

    topk = pd.DataFrame(rows)
    topk["total_positive_rows"] = total_positive_rows
    topk["total_positive_episodes"] = total_positive_episodes
    return topk


def review_summary(flagged, all_rows, score_name, policy_name):
    tp_rows = int(flagged[TARGET_COL].sum())
    rows_reviewed = int(len(flagged))
    fp_rows = rows_reviewed - tp_rows
    positive_rows = int(all_rows[TARGET_COL].sum())
    positive_episodes = set(all_rows.loc[all_rows[TARGET_COL].eq(1), "episode_id"])
    captured_positive_episodes = set(flagged.loc[flagged[TARGET_COL].eq(1), "episode_id"])

    return {
        "score_name": score_name,
        "policy": policy_name,
        "rows_reviewed": rows_reviewed,
        "reviewed_row_fraction": rows_reviewed / len(all_rows) if len(all_rows) else np.nan,
        "true_positive_rows": tp_rows,
        "false_positive_rows": fp_rows,
        "precision_ppv": tp_rows / rows_reviewed if rows_reviewed else np.nan,
        "row_recall_sensitivity": tp_rows / positive_rows if positive_rows else np.nan,
        "positive_episodes_captured": len(captured_positive_episodes),
        "episode_recall_sensitivity": len(captured_positive_episodes) / len(positive_episodes) if positive_episodes else np.nan,
        "false_alerts_per_true_positive": fp_rows / tp_rows if tp_rows else np.nan,
        "alerts_per_100_rows": 100 * rows_reviewed / len(all_rows) if len(all_rows) else np.nan,
    }


def make_threshold_table(df, score_name, probability_col):
    rows = []
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        flagged = df[df[probability_col] >= threshold]
        row = review_summary(flagged, df, score_name, f"threshold_{threshold:.3f}")
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def make_first_alert_table(df, score_name, probability_col):
    rows = []
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        alerts = (
            df[df[probability_col] >= threshold]
            .sort_values(["episode_id", "landmark_hour", probability_col], ascending=[True, True, False])
            .groupby("episode_id")
            .head(1)
        )
        row = review_summary(alerts, df, score_name, f"first_alert_threshold_{threshold:.3f}")
        row["threshold"] = threshold
        row["episodes_alerted"] = int(alerts["episode_id"].nunique())
        row["episode_alert_rate"] = alerts["episode_id"].nunique() / df["episode_id"].nunique() if df["episode_id"].nunique() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_calibration_deciles(df, score_name, probability_col):
    out = df[["landmark_id", TARGET_COL, probability_col]].copy()
    out["score_name"] = score_name
    try:
        out["risk_decile"] = pd.qcut(out[probability_col], q=10, labels=False, duplicates="drop")
    except ValueError:
        out["risk_decile"] = pd.cut(out[probability_col], bins=10, labels=False)
    out["risk_decile"] = out["risk_decile"].astype("Int64")

    return (
        out
        .groupby(["score_name", "risk_decile"], observed=True, dropna=True)
        .agg(
            rows=("landmark_id", "count"),
            mean_predicted_risk=(probability_col, "mean"),
            observed_event_rate=(TARGET_COL, "mean"),
            positive_rows=(TARGET_COL, "sum"),
        )
        .reset_index()
    )


def apply_platt(platt_model, raw_prob):
    raw_logit = logit(raw_prob)
    coef = float(platt_model.coef_[0][0])
    intercept = float(platt_model.intercept_[0])
    return sigmoid(raw_logit * coef + intercept)


# %% Load feature matrix and split development eras

print("Loading Run 18 feature matrix...", flush=True)
features = pd.read_csv(FEATURE_FILE, parse_dates=["landmark_time"])
features[TARGET_COL] = features[TARGET_COL].astype(int)

model_frame = features[
    features["run18_primary_model_frame"].eq(1)
    & features["split_role"].eq("development")
].copy()

train_df = model_frame[model_frame["anchor_year_group"].isin(TRAIN_CORE_YEAR_GROUPS)].copy()
calib_df = model_frame[model_frame["anchor_year_group"].isin(CALIBRATION_YEAR_GROUPS)].copy()
val_df = model_frame[model_frame["anchor_year_group"].isin(VALIDATION_YEAR_GROUPS)].copy()
lockbox_df = features[features["anchor_year_group"].eq(LOCKBOX_YEAR_GROUP)].copy()

split_audit = pd.DataFrame([
    {
        "split": "train_core",
        "anchor_year_groups": ", ".join(TRAIN_CORE_YEAR_GROUPS),
        "rows": len(train_df),
        "positive_rows": int(train_df[TARGET_COL].sum()),
        "prevalence": float(train_df[TARGET_COL].mean()),
        "episodes": int(train_df["episode_id"].nunique()),
        "patients": int(train_df["subject_id"].nunique()),
    },
    {
        "split": "calibration",
        "anchor_year_groups": ", ".join(CALIBRATION_YEAR_GROUPS),
        "rows": len(calib_df),
        "positive_rows": int(calib_df[TARGET_COL].sum()),
        "prevalence": float(calib_df[TARGET_COL].mean()),
        "episodes": int(calib_df["episode_id"].nunique()),
        "patients": int(calib_df["subject_id"].nunique()),
    },
    {
        "split": "validation",
        "anchor_year_groups": ", ".join(VALIDATION_YEAR_GROUPS),
        "rows": len(val_df),
        "positive_rows": int(val_df[TARGET_COL].sum()),
        "prevalence": float(val_df[TARGET_COL].mean()),
        "episodes": int(val_df["episode_id"].nunique()),
        "patients": int(val_df["subject_id"].nunique()),
    },
    {
        "split": "temporal_lockbox_audit_only",
        "anchor_year_groups": LOCKBOX_YEAR_GROUP,
        "rows": len(lockbox_df),
        "positive_rows": int(lockbox_df[TARGET_COL].sum()),
        "prevalence": float(lockbox_df[TARGET_COL].mean()) if len(lockbox_df) else np.nan,
        "episodes": int(lockbox_df["episode_id"].nunique()),
        "patients": int(lockbox_df["subject_id"].nunique()),
    },
])

print(split_audit.to_string(index=False), flush=True)


# %% Fit best Run 18 model family on early development rows

static_numeric = [
    "landmark_hour",
    "landmark_day",
    "anchor_age",
    "early_positive_culture",
]
static_categorical = [
    "gender",
    "admission_type",
    "insurance",
    "race",
    "first_careunit",
]
lab_numeric = sorted({
    c for c in features.columns
    if any(c.startswith(prefix) for prefix in LAB_PREFIXES)
})
numeric_cols = static_numeric + lab_numeric
categorical_cols = static_categorical
feature_cols = numeric_cols + categorical_cols

positives = int(train_df[TARGET_COL].sum())
negatives = int(len(train_df) - positives)
scale_pos_weight = negatives / positives if positives else 1.0

print("", flush=True)
print("Fitting XGBoost static_context_labs_48h on train_core years...", flush=True)
pipeline = Pipeline([
    ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=2,
        tree_method="hist",
    )),
])
pipeline.fit(train_df[feature_cols], train_df[TARGET_COL].astype(int))

joblib.dump(pipeline, MODEL_PATH / "run19_xgboost_static_context_labs_48h_train_core.joblib")

calib_raw = pipeline.predict_proba(calib_df[feature_cols])[:, 1]
val_raw = pipeline.predict_proba(val_df[feature_cols])[:, 1]


# %% Fit calibration maps on calibration-era rows only

print("Fitting calibration maps on 2014-2016 only...", flush=True)
platt = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
platt.fit(logit(calib_raw).reshape(-1, 1), calib_df[TARGET_COL].astype(int))

isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
isotonic.fit(calib_raw, calib_df[TARGET_COL].astype(int))

print("Saving base model and calibrators...", flush=True)
joblib.dump(platt, MODEL_PATH / "run19_platt_calibrator_2014_2016.joblib")
joblib.dump(isotonic, MODEL_PATH / "run19_isotonic_calibrator_2014_2016.joblib")
print("Building calibration and validation prediction frames...", flush=True)

calib_scored = calib_df[[
    "landmark_id",
    "episode_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "anchor_year_group",
    "landmark_hour",
    "landmark_day",
    "landmark_time",
    TARGET_COL,
]].copy()
val_scored = val_df[calib_scored.columns].copy()

calib_scored["raw_probability"] = calib_raw
print("Applying Platt calibrator to calibration split...", flush=True)
calib_scored["platt_probability"] = apply_platt(platt, calib_raw)
print("Applying isotonic calibrator to calibration split...", flush=True)
calib_scored["isotonic_probability"] = isotonic.predict(calib_raw)

val_scored["raw_probability"] = val_raw
print("Applying Platt calibrator to validation split...", flush=True)
val_scored["platt_probability"] = apply_platt(platt, val_raw)
print("Applying isotonic calibrator to validation split...", flush=True)
val_scored["isotonic_probability"] = isotonic.predict(val_raw)

score_specs = [
    ("raw_xgboost", "raw_probability"),
    ("platt_calibrated_xgboost", "platt_probability"),
    ("isotonic_calibrated_xgboost", "isotonic_probability"),
]


# %% Evaluate calibration and review policies

metric_rows = []
topk_frames = []
threshold_frames = []
first_alert_frames = []
decile_frames = []

for score_name, probability_col in score_specs:
    print(f"Evaluating metrics: {score_name}", flush=True)
    metric_rows.append(evaluate_predictions(calib_scored, "calibration", score_name, probability_col))
    metric_rows.append(evaluate_predictions(val_scored, "validation", score_name, probability_col))
    print(f"Building top-k table: {score_name}", flush=True)
    topk_frames.append(make_topk_table(val_scored, score_name, probability_col))
    print(f"Building threshold table: {score_name}", flush=True)
    threshold_frames.append(make_threshold_table(val_scored, score_name, probability_col))
    print(f"Building first-alert table: {score_name}", flush=True)
    first_alert_frames.append(make_first_alert_table(val_scored, score_name, probability_col))
    print(f"Building calibration deciles: {score_name}", flush=True)
    decile_frames.append(make_calibration_deciles(val_scored, score_name, probability_col))

print("Combining evaluation tables...", flush=True)
model_comparison = pd.DataFrame(metric_rows)
topk_review = pd.concat(topk_frames, ignore_index=True)
threshold_policy = pd.concat(threshold_frames, ignore_index=True)
first_alert_policy = pd.concat(first_alert_frames, ignore_index=True)
calibration_deciles = pd.concat(decile_frames, ignore_index=True)
print("Evaluation tables combined.", flush=True)


# %% Plots

plot_artifacts = []
if GENERATE_PLOTS:
    print("Creating calibration plot...", flush=True)
    plt.figure(figsize=(8, 6))
    for score_name in calibration_deciles["score_name"].unique():
        subset = calibration_deciles[calibration_deciles["score_name"].eq(score_name)]
        plt.plot(
            subset["mean_predicted_risk"],
            subset["observed_event_rate"],
            marker="o",
            label=score_name,
        )
    limit = max(
        0.10,
        float(calibration_deciles["mean_predicted_risk"].max()),
        float(calibration_deciles["observed_event_rate"].max()),
    )
    plt.plot([0, limit], [0, limit], "--", color="gray", label="perfect calibration")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed strict proxy event rate")
    plt.title("Run 19 Validation Calibration by Risk Decile")
    plt.legend()
    plt.tight_layout()
    calibration_plot = PLOT_PATH / "v0_5_run19_validation_calibration_deciles.png"
    plt.savefig(calibration_plot, dpi=200)
    plt.close()
    plot_artifacts.append({"artifact": "calibration_decile_plot", "path": str(calibration_plot)})

    print("Creating top-k PPV plot...", flush=True)
    plt.figure(figsize=(8, 6))
    subset = topk_review[topk_review["policy"].isin(["top_1_percent_rows", "top_2_percent_rows", "top_5_percent_rows", "top_10_percent_rows"])]
    for score_name in subset["score_name"].unique():
        score_subset = subset[subset["score_name"].eq(score_name)]
        plt.plot(
            score_subset["reviewed_row_fraction"] * 100,
            score_subset["precision_ppv"],
            marker="o",
            label=score_name,
        )
    plt.xlabel("Rows reviewed (%)")
    plt.ylabel("PPV among reviewed rows")
    plt.title("Run 19 Validation Top-Risk Review PPV")
    plt.legend()
    plt.tight_layout()
    topk_plot = PLOT_PATH / "v0_5_run19_validation_topk_ppv.png"
    plt.savefig(topk_plot, dpi=200)
    plt.close()
    plot_artifacts.append({"artifact": "topk_ppv_plot", "path": str(topk_plot)})
else:
    print("Plot generation disabled in this environment; CSV outputs will be saved.", flush=True)


# %% Save outputs

model_comparison_file = OUTPUT_PATH / "v0_5_run19_calibration_model_comparison.csv"
topk_file = OUTPUT_PATH / "v0_5_run19_validation_topk_review.csv"
threshold_file = OUTPUT_PATH / "v0_5_run19_validation_threshold_policy.csv"
first_alert_file = OUTPUT_PATH / "v0_5_run19_validation_first_alert_policy.csv"
deciles_file = OUTPUT_PATH / "v0_5_run19_validation_calibration_deciles.csv"
split_audit_file = OUTPUT_PATH / "v0_5_run19_development_split_audit.csv"
predictions_file = OUTPUT_PATH / "v0_5_run19_validation_predictions.csv"

print("Saving Run 19 CSV outputs...", flush=True)
model_comparison.to_csv(model_comparison_file, index=False)
topk_review.to_csv(topk_file, index=False)
threshold_policy.to_csv(threshold_file, index=False)
first_alert_policy.to_csv(first_alert_file, index=False)
calibration_deciles.to_csv(deciles_file, index=False)
split_audit.to_csv(split_audit_file, index=False)
val_scored.to_csv(predictions_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "model_comparison", "path": str(model_comparison_file)},
    {"artifact": "topk_review", "path": str(topk_file)},
    {"artifact": "threshold_policy", "path": str(threshold_file)},
    {"artifact": "first_alert_policy", "path": str(first_alert_file)},
    {"artifact": "calibration_deciles", "path": str(deciles_file)},
    {"artifact": "development_split_audit", "path": str(split_audit_file)},
    {"artifact": "validation_predictions", "path": str(predictions_file)},
] + plot_artifacts)
manifest_file = OUTPUT_PATH / "v0_5_run19_manifest.csv"
manifest.to_csv(manifest_file, index=False)
print("Run 19 CSV outputs saved.", flush=True)


# %% Console summary

print("", flush=True)
print("Run 19 validation performance:", flush=True)
print(
    model_comparison[
        model_comparison["split"].eq("validation")
    ][[
        "score_name",
        "rows",
        "positive_rows",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "pr_auc_lift_over_prevalence",
        "brier_skill_score",
        "calibration_intercept",
        "calibration_slope",
        "expected_observed_ratio",
    ]].round(4).to_string(index=False),
    flush=True,
)
print("", flush=True)
print("Run 19 validation top-risk review:", flush=True)
print(
    topk_review[
        topk_review["policy"].isin(["top_1_percent_rows", "top_5_percent_rows", "top_100_rows"])
    ][[
        "score_name",
        "policy",
        "rows_reviewed",
        "true_positive_rows",
        "precision_ppv",
        "row_recall_sensitivity",
        "episode_recall_sensitivity",
        "false_alerts_per_true_positive",
    ]].round(4).to_string(index=False),
    flush=True,
)
print("", flush=True)
print("Temporal lockbox remains held out; no lockbox predictions were generated.", flush=True)
print(f"Saved Run 19 outputs to: {OUTPUT_PATH}", flush=True)
print("Modeling 14 v0.5 Calibration Review Policy complete.", flush=True)

