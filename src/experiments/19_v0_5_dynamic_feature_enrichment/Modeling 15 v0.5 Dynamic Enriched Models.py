# %% Imports and paths

import sys
from pathlib import Path

try:
    import joblib
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
    print("Missing Run 20 modeling dependency:", exc)
    print("Install required packages: pip install pandas numpy scikit-learn xgboost joblib")
    sys.exit(1)


PROJECT_PATH = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 20 (v0.5 Dynamic Feature Enrichment)"
MODEL_PATH = OUTPUT_PATH / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_PATH / "v0_5_run20_dynamic_enriched_features.csv"
TARGET_COL = "future_strict_cvc_bsi_proxy_7d"
RANDOM_STATE = 42

TRAIN_CORE_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013"]
CALIBRATION_YEAR_GROUPS = ["2014 - 2016"]
VALIDATION_YEAR_GROUPS = ["2017 - 2019"]
LOCKBOX_YEAR_GROUP = "2020 - 2022"

STATIC_NUMERIC = ["landmark_hour", "landmark_day", "anchor_age", "early_positive_culture"]
STATIC_CATEGORICAL = ["gender", "admission_type", "insurance", "race", "first_careunit"]
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


def apply_platt(platt_model, raw_prob):
    raw_logit = logit(raw_prob)
    coef = float(platt_model.coef_[0][0])
    intercept = float(platt_model.intercept_[0])
    return sigmoid(raw_logit * coef + intercept)


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])


def evaluate_predictions(df, split_name, score_name, probability_col):
    y_true = df[TARGET_COL].astype(int).to_numpy()
    y_prob = df[probability_col].to_numpy()
    prevalence = float(np.mean(y_true))
    pr_auc = average_precision_score(y_true, y_prob)
    bss, brier, brier_reference = brier_skill_score(y_true, y_prob)
    cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "split": split_name,
        "score_name": score_name,
        "rows": int(len(df)),
        "positive_rows": int(np.sum(y_true)),
        "prevalence": prevalence,
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": pr_auc / prevalence if prevalence else np.nan,
        "brier_score": brier,
        "brier_reference_prevalence": brier_reference,
        "brier_skill_score": bss,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "expected_observed_ratio": expected_observed_ratio(y_true, y_prob),
    }


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
    }


def make_topk_table(df, score_name, probability_col):
    rows = []
    ranked = df.sort_values(probability_col, ascending=False).reset_index(drop=True)
    for pct in [1, 2, 5, 10]:
        n_reviewed = max(1, int(np.ceil(len(ranked) * pct / 100)))
        rows.append(review_summary(ranked.head(n_reviewed), ranked, score_name, f"top_{pct}_percent_rows"))
    for n_reviewed in [25, 50, 100, 250, 500]:
        rows.append(review_summary(ranked.head(min(n_reviewed, len(ranked))), ranked, score_name, f"top_{n_reviewed}_rows"))
    return pd.DataFrame(rows)


def make_first_alert_table(df, score_name, probability_col):
    rows = []
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
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


def make_threshold_table(df, score_name, probability_col):
    rows = []
    for threshold in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
        flagged = df[df[probability_col] >= threshold]
        row = review_summary(flagged, df, score_name, f"threshold_{threshold:.3f}")
        row["threshold"] = threshold
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


def fit_xgboost(train_df, numeric_cols, categorical_cols):
    positives = int(train_df[TARGET_COL].sum())
    negatives = int(len(train_df) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
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
    pipeline.fit(train_df[numeric_cols + categorical_cols], train_df[TARGET_COL].astype(int))
    return pipeline


def feature_importance_frame(pipeline, score_name):
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        if len(feature_names) == len(importance):
            return (
                pd.DataFrame({"score_name": score_name, "feature": feature_names, "importance": importance})
                .sort_values(["score_name", "importance"], ascending=[True, False])
            )
    except Exception:
        pass
    return pd.DataFrame(columns=["score_name", "feature", "importance"])


# %% Load features and define split

print("Loading Run 20 enriched feature matrix...", flush=True)
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
    {"split": "train_core", "anchor_year_groups": ", ".join(TRAIN_CORE_YEAR_GROUPS), "rows": len(train_df), "positive_rows": int(train_df[TARGET_COL].sum()), "prevalence": float(train_df[TARGET_COL].mean()), "episodes": int(train_df["episode_id"].nunique()), "patients": int(train_df["subject_id"].nunique())},
    {"split": "calibration", "anchor_year_groups": ", ".join(CALIBRATION_YEAR_GROUPS), "rows": len(calib_df), "positive_rows": int(calib_df[TARGET_COL].sum()), "prevalence": float(calib_df[TARGET_COL].mean()), "episodes": int(calib_df["episode_id"].nunique()), "patients": int(calib_df["subject_id"].nunique())},
    {"split": "validation", "anchor_year_groups": ", ".join(VALIDATION_YEAR_GROUPS), "rows": len(val_df), "positive_rows": int(val_df[TARGET_COL].sum()), "prevalence": float(val_df[TARGET_COL].mean()), "episodes": int(val_df["episode_id"].nunique()), "patients": int(val_df["subject_id"].nunique())},
    {"split": "temporal_lockbox_audit_only", "anchor_year_groups": LOCKBOX_YEAR_GROUP, "rows": len(lockbox_df), "positive_rows": int(lockbox_df[TARGET_COL].sum()), "prevalence": float(lockbox_df[TARGET_COL].mean()) if len(lockbox_df) else np.nan, "episodes": int(lockbox_df["episode_id"].nunique()), "patients": int(lockbox_df["subject_id"].nunique())},
])
print(split_audit.to_string(index=False), flush=True)


# %% Feature sets

lab_cols = sorted({c for c in features.columns if any(c.startswith(prefix) for prefix in LAB_PREFIXES)})
vital_cols = sorted({c for c in features.columns if c.startswith("vital_")})
therapy_cols = sorted({c for c in features.columns if c.startswith("abx_") or c.startswith("vaso_")})

feature_sets = [
    {"name": "static_labs", "numeric_cols": STATIC_NUMERIC + lab_cols, "categorical_cols": STATIC_CATEGORICAL},
    {"name": "static_labs_vitals", "numeric_cols": STATIC_NUMERIC + lab_cols + vital_cols, "categorical_cols": STATIC_CATEGORICAL},
    {"name": "static_labs_vitals_therapy", "numeric_cols": STATIC_NUMERIC + lab_cols + vital_cols + therapy_cols, "categorical_cols": STATIC_CATEGORICAL},
]

feature_audit = []
for feature_set in feature_sets:
    for col in feature_set["numeric_cols"] + feature_set["categorical_cols"]:
        feature_audit.append({
            "feature_set": feature_set["name"],
            "feature": col,
            "role": "numeric" if col in feature_set["numeric_cols"] else "categorical",
            "missing_rate_train": float(train_df[col].isna().mean()) if col in train_df.columns else np.nan,
            "missing_rate_validation": float(val_df[col].isna().mean()) if col in val_df.columns else np.nan,
        })
feature_audit = pd.DataFrame(feature_audit)


# %% Fit, calibrate, evaluate

metric_rows = []
prediction_frames = []
topk_frames = []
threshold_frames = []
first_alert_frames = []
decile_frames = []
importance_frames = []

for feature_set in feature_sets:
    name = feature_set["name"]
    numeric_cols = feature_set["numeric_cols"]
    categorical_cols = feature_set["categorical_cols"]
    feature_cols = numeric_cols + categorical_cols

    print("", flush=True)
    print(f"Fitting XGBoost: {name}", flush=True)
    pipeline = fit_xgboost(train_df, numeric_cols, categorical_cols)
    joblib.dump(pipeline, MODEL_PATH / f"run20_xgboost_{name}.joblib")

    calib_raw = pipeline.predict_proba(calib_df[feature_cols])[:, 1]
    val_raw = pipeline.predict_proba(val_df[feature_cols])[:, 1]

    platt = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
    platt.fit(logit(calib_raw).reshape(-1, 1), calib_df[TARGET_COL].astype(int))
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(calib_raw, calib_df[TARGET_COL].astype(int))
    joblib.dump(platt, MODEL_PATH / f"run20_platt_{name}.joblib")
    joblib.dump(isotonic, MODEL_PATH / f"run20_isotonic_{name}.joblib")

    keep_cols = ["landmark_id", "episode_id", "subject_id", "hadm_id", "stay_id", "anchor_year_group", "landmark_hour", "landmark_day", "landmark_time", TARGET_COL]
    calib_scored = calib_df[keep_cols].copy()
    val_scored = val_df[keep_cols].copy()
    calib_scored["feature_set"] = name
    val_scored["feature_set"] = name
    calib_scored["raw_probability"] = calib_raw
    val_scored["raw_probability"] = val_raw
    calib_scored["platt_probability"] = apply_platt(platt, calib_raw)
    val_scored["platt_probability"] = apply_platt(platt, val_raw)
    calib_scored["isotonic_probability"] = isotonic.predict(calib_raw)
    val_scored["isotonic_probability"] = isotonic.predict(val_raw)

    score_specs = [
        (f"{name}_raw", "raw_probability"),
        (f"{name}_platt", "platt_probability"),
        (f"{name}_isotonic", "isotonic_probability"),
    ]

    for score_name, probability_col in score_specs:
        metric_rows.append(evaluate_predictions(calib_scored, "calibration", score_name, probability_col))
        metric_rows.append(evaluate_predictions(val_scored, "validation", score_name, probability_col))
        topk_frames.append(make_topk_table(val_scored, score_name, probability_col))
        threshold_frames.append(make_threshold_table(val_scored, score_name, probability_col))
        first_alert_frames.append(make_first_alert_table(val_scored, score_name, probability_col))
        decile_frames.append(make_calibration_deciles(val_scored, score_name, probability_col))

    prediction_frames.append(val_scored)
    importance_frames.append(feature_importance_frame(pipeline, name))


# %% Save outputs

model_comparison = pd.DataFrame(metric_rows)
predictions = pd.concat(prediction_frames, ignore_index=True)
topk_review = pd.concat(topk_frames, ignore_index=True)
threshold_policy = pd.concat(threshold_frames, ignore_index=True)
first_alert_policy = pd.concat(first_alert_frames, ignore_index=True)
calibration_deciles = pd.concat(decile_frames, ignore_index=True)
feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()

model_comparison_file = OUTPUT_PATH / "v0_5_run20_dynamic_model_comparison.csv"
topk_file = OUTPUT_PATH / "v0_5_run20_validation_topk_review.csv"
threshold_file = OUTPUT_PATH / "v0_5_run20_validation_threshold_policy.csv"
first_alert_file = OUTPUT_PATH / "v0_5_run20_validation_first_alert_policy.csv"
deciles_file = OUTPUT_PATH / "v0_5_run20_validation_calibration_deciles.csv"
split_audit_file = OUTPUT_PATH / "v0_5_run20_development_split_audit.csv"
feature_audit_file = OUTPUT_PATH / "v0_5_run20_model_feature_audit.csv"
importance_file = OUTPUT_PATH / "v0_5_run20_xgboost_feature_importance.csv"
predictions_file = OUTPUT_PATH / "v0_5_run20_validation_predictions.csv"
manifest_file = OUTPUT_PATH / "v0_5_run20_manifest.csv"

model_comparison.to_csv(model_comparison_file, index=False)
topk_review.to_csv(topk_file, index=False)
threshold_policy.to_csv(threshold_file, index=False)
first_alert_policy.to_csv(first_alert_file, index=False)
calibration_deciles.to_csv(deciles_file, index=False)
split_audit.to_csv(split_audit_file, index=False)
feature_audit.to_csv(feature_audit_file, index=False)
feature_importance.to_csv(importance_file, index=False)
predictions.to_csv(predictions_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "feature_matrix", "path": str(FEATURE_FILE)},
    {"artifact": "model_comparison", "path": str(model_comparison_file)},
    {"artifact": "topk_review", "path": str(topk_file)},
    {"artifact": "threshold_policy", "path": str(threshold_file)},
    {"artifact": "first_alert_policy", "path": str(first_alert_file)},
    {"artifact": "calibration_deciles", "path": str(deciles_file)},
    {"artifact": "development_split_audit", "path": str(split_audit_file)},
    {"artifact": "feature_audit", "path": str(feature_audit_file)},
    {"artifact": "feature_importance", "path": str(importance_file)},
    {"artifact": "validation_predictions", "path": str(predictions_file)},
])
manifest.to_csv(manifest_file, index=False)


# %% Console summary

print("", flush=True)
print("Run 20 validation performance:", flush=True)
print(
    model_comparison[model_comparison["split"].eq("validation")][[
        "score_name",
        "rows",
        "positive_rows",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "pr_auc_lift_over_prevalence",
        "brier_skill_score",
        "expected_observed_ratio",
    ]].round(4).to_string(index=False),
    flush=True,
)
print("", flush=True)
print("Run 20 validation top-risk review:", flush=True)
print(
    topk_review[topk_review["policy"].isin(["top_1_percent_rows", "top_5_percent_rows", "top_100_rows"])][[
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
print(f"Saved Run 20 outputs to: {OUTPUT_PATH}", flush=True)
print("Modeling 15 v0.5 Dynamic Enriched Models complete.", flush=True)

