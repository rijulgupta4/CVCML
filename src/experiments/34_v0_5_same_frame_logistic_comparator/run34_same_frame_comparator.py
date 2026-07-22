"""Run 34: post-hoc same-frame simple-model comparator for manuscript revision.

This analysis does not reopen model selection. It fits one prespecified,
L2-regularized logistic regression on the same leakage-safe features and
calendar partitions used by the frozen Run 29 XGBoost candidate. The
2020-2022 period is never loaded or scored.
"""

from __future__ import annotations

from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(r"C:\path\to\CVCML")
DATA = PROJECT / "data" / "v0_5"
RUN29 = PROJECT / "Outputs" / "Run 29 (v0.5 Outcome Validity and Leakage Audit)"
OUTPUT = Path(__file__).resolve().parent / "run34_outputs"
MODELS = OUTPUT / "models"

FEATURE_FILE = DATA / "v0_5_run20_dynamic_enriched_features.csv"
LABEL_FILE = DATA / "v0_5_run22_source_screened_daily_landmarks.csv"
XGB_PREDICTION_FILE = RUN29 / "v0_5_run29_validation_predictions.csv"

TARGET = "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013"]
CALIB_GROUPS = ["2014 - 2016"]
VALID_GROUPS = ["2017 - 2019"]
LOCKBOX_GROUP = "2020 - 2022"
STATIC_NUMERIC = ["landmark_hour", "landmark_day", "anchor_age"]
STATIC_CATEGORICAL = ["gender", "admission_type", "insurance", "race", "first_careunit"]
LAB_PREFIXES = ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"]
RANDOM_STATE = 2034
N_BOOTSTRAP = int(os.environ.get("RUN34_BOOTSTRAP", "2000"))


def clip_prob(probability):
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)


def logit(probability):
    probability = clip_prob(probability)
    return np.log(probability / (1 - probability))


def sigmoid(score):
    score = np.clip(np.asarray(score, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-score))


def apply_platt(model, raw_probability):
    score = logit(raw_probability) * float(model.coef_[0][0]) + float(model.intercept_[0])
    return sigmoid(score)


def predict_logistic_probability(pipeline, frame):
    """Avoid the sparse BLAS crash seen in this project's sklearn build."""
    transformed = pipeline.named_steps["preprocess"].transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    model = pipeline.named_steps["model"]
    coefficients = np.asarray(model.coef_[0], dtype=float).reshape(1, -1)
    scores = (transformed * coefficients).sum(axis=1) + float(model.intercept_[0])
    return sigmoid(scores)


def calibration_intercept_slope(y_true, probability, sample_weight=None):
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    predictor = np.clip(logit(probability), -20, 20).reshape(-1, 1)
    model = LogisticRegression(solver="liblinear", C=1e6, max_iter=2000)
    model.fit(predictor, y_true, sample_weight=sample_weight)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metric_values(y_true, probability, sample_weight=None):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if sample_weight is None:
        prevalence = float(y_true.mean())
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        prevalence = float(np.average(y_true, weights=sample_weight))
    if len(np.unique(y_true)) < 2:
        return {key: np.nan for key in [
            "roc_auc", "pr_auc", "prevalence", "pr_auc_lift", "brier_score",
            "brier_skill_score", "calibration_intercept", "calibration_slope",
            "expected_observed_ratio",
        ]}
    roc_auc = float(roc_auc_score(y_true, probability, sample_weight=sample_weight))
    pr_auc = float(average_precision_score(y_true, probability, sample_weight=sample_weight))
    brier = float(brier_score_loss(y_true, probability, sample_weight=sample_weight))
    reference = float(
        brier_score_loss(
            y_true,
            np.repeat(prevalence, len(y_true)),
            sample_weight=sample_weight,
        )
    )
    intercept, slope = calibration_intercept_slope(y_true, probability, sample_weight)
    if sample_weight is None:
        expected = float(probability.sum())
        observed = int(y_true.sum())
    else:
        expected = float(np.sum(probability * sample_weight))
        observed = float(np.sum(y_true * sample_weight))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence else np.nan,
        "brier_score": brier,
        "brier_skill_score": 1 - brier / reference if reference else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_observed_ratio": expected / observed if observed else np.nan,
    }


def make_preprocessor(numeric_cols, categorical_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_cols),
        ("categorical", categorical, categorical_cols),
    ])


def load_frame():
    features = pd.read_csv(FEATURE_FILE, parse_dates=["landmark_time", "exposure_start"])
    labels = pd.read_csv(LABEL_FILE, usecols=["landmark_id", TARGET])
    features = features.merge(labels, on="landmark_id", how="left", validate="one_to_one")
    features[TARGET] = features[TARGET].fillna(0).astype(int)
    frame = features[
        features["run18_primary_model_frame"].eq(1)
        & features["split_role"].eq("development")
        & ~features["anchor_year_group"].eq(LOCKBOX_GROUP)
    ].copy()
    dynamic_numeric = sorted({
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in LAB_PREFIXES)
        or column.startswith("vital_")
        or column.startswith("abx_")
        or column.startswith("vaso_")
    })
    numeric_cols = STATIC_NUMERIC + dynamic_numeric
    return frame, numeric_cols


def fit_logistic(train, calib, numeric_cols):
    feature_cols = numeric_cols + STATIC_CATEGORICAL
    model = Pipeline([
        ("preprocess", make_preprocessor(numeric_cols, STATIC_CATEGORICAL)),
        (
            "model",
            LogisticRegression(
                C=1.0,
                l1_ratio=0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])
    model.fit(train[feature_cols], train[TARGET].astype(int))
    calib_raw = predict_logistic_probability(model, calib[feature_cols])
    platt = LogisticRegression(solver="liblinear", C=1e6, max_iter=2000)
    platt.fit(logit(calib_raw).reshape(-1, 1), calib[TARGET].astype(int))
    return model, platt, feature_cols


def attach_xgb_predictions(valid):
    predictions = pd.read_csv(XGB_PREDICTION_FILE)
    predictions = predictions[
        predictions["model_variant"].eq("safe_exclude_early_positive")
    ][["landmark_id", "platt_probability"]].rename(
        columns={"platt_probability": "xgboost_probability"}
    )
    scored = valid.merge(predictions, on="landmark_id", how="inner", validate="one_to_one")
    if len(scored) != len(valid):
        raise ValueError(f"XGBoost prediction coverage mismatch: {len(scored):,}/{len(valid):,}")
    return scored


def collapse_episode(scored, probability_col):
    best = (
        scored.sort_values(["episode_id", probability_col, "landmark_hour"], ascending=[True, False, True])
        .groupby("episode_id", sort=False)
        .head(1)
        .copy()
    )
    episode_target = scored.groupby("episode_id")[TARGET].max()
    best["episode_positive"] = best["episode_id"].map(episode_target).astype(int)
    return best


def summarize_metric_frame(scored, model_name, probability_col, analysis_unit, sample_weight=None):
    if analysis_unit == "landmark_row":
        local = scored
        target_col = TARGET
    elif analysis_unit == "episode_maximum":
        local = collapse_episode(scored, probability_col)
        target_col = "episode_positive"
    else:
        raise ValueError(analysis_unit)
    values = metric_values(local[target_col], local[probability_col], sample_weight=sample_weight)
    rows = []
    for metric, estimate in values.items():
        rows.append({
            "model": model_name,
            "analysis_unit": analysis_unit,
            "metric": metric,
            "estimate": estimate,
            "rows_or_episodes": int(len(local)),
            "positive_rows_or_episodes": int(local[target_col].sum()),
        })
    return pd.DataFrame(rows)


def bootstrap_models(scored):
    probability_cols = {"XGBoost": "xgboost_probability", "Logistic regression": "logistic_probability"}
    episode_frames = {
        model_name: collapse_episode(scored, probability_col)
        for model_name, probability_col in probability_cols.items()
    }
    units = {
        "landmark_row": {
            "target": TARGET,
            "frames": {model_name: scored for model_name in probability_cols},
        },
        "episode_maximum": {
            "target": "episode_positive",
            "frames": episode_frames,
        },
    }
    subject_ids = np.asarray(scored["subject_id"].drop_duplicates())
    arrays = {}
    for analysis_unit, specification in units.items():
        arrays[analysis_unit] = {}
        for model_name, probability_col in probability_cols.items():
            frame = specification["frames"][model_name]
            arrays[analysis_unit][model_name] = {
                subject_id: (
                    group[specification["target"]].to_numpy(dtype=int),
                    group[probability_col].to_numpy(dtype=float),
                )
                for subject_id, group in frame.groupby("subject_id", sort=False)
            }
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    for iteration in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        for analysis_unit in ["landmark_row", "episode_maximum"]:
            model_metrics = {}
            for model_name, probability_col in probability_cols.items():
                model_arrays = arrays[analysis_unit][model_name]
                y = np.concatenate([model_arrays[subject_id][0] for subject_id in sampled_ids])
                probability = np.concatenate([model_arrays[subject_id][1] for subject_id in sampled_ids])
                full_values = metric_values(y, probability)
                values = {metric: full_values[metric] for metric in [
                    "roc_auc", "pr_auc", "prevalence", "pr_auc_lift",
                    "brier_score", "brier_skill_score",
                ]}
                model_metrics[model_name] = values
                for metric, estimate in values.items():
                    records.append({
                        "iteration": iteration,
                        "analysis_unit": analysis_unit,
                        "model": model_name,
                        "metric": metric,
                        "estimate": estimate,
                    })
            for metric in model_metrics["XGBoost"]:
                records.append({
                    "iteration": iteration,
                    "analysis_unit": analysis_unit,
                    "model": "Logistic minus XGBoost",
                    "metric": metric,
                    "estimate": model_metrics["Logistic regression"][metric] - model_metrics["XGBoost"][metric],
                })
    return pd.DataFrame(records)


def add_intervals(point_estimates, bootstrap):
    summaries = []
    for keys, group in bootstrap.groupby(["analysis_unit", "model", "metric"]):
        values = group["estimate"].replace([np.inf, -np.inf], np.nan).dropna()
        summaries.append({
            "analysis_unit": keys[0],
            "model": keys[1],
            "metric": keys[2],
            "ci_lower_95": values.quantile(0.025) if len(values) else np.nan,
            "ci_upper_95": values.quantile(0.975) if len(values) else np.nan,
            "valid_bootstrap_replicates": int(len(values)),
        })
    intervals = pd.DataFrame(summaries)
    return point_estimates.merge(intervals, on=["analysis_unit", "model", "metric"], how="left")


def review_policy_points(scored, model_name, probability_col):
    best = collapse_episode(scored, probability_col).sort_values(probability_col, ascending=False)
    total_positive = int(best["episode_positive"].sum())
    rows = []
    for percent in [1, 2, 5, 10, 20]:
        n_review = max(1, int(np.ceil(len(best) * percent / 100)))
        selected = best.head(n_review)
        true_positive = int(selected["episode_positive"].sum())
        false_positive = n_review - true_positive
        rows.append({
            "model": model_name,
            "review_percent": percent,
            "episodes_reviewed": n_review,
            "true_positive_episodes": true_positive,
            "false_positive_episodes": false_positive,
            "ppv": true_positive / n_review,
            "recall": true_positive / total_positive,
            "false_reviews_per_true_positive": false_positive / true_positive if true_positive else np.nan,
        })
    return pd.DataFrame(rows)


def episode_weighted_sensitivity(scored):
    row_count = scored.groupby("episode_id")["landmark_id"].transform("count")
    weights = 1 / row_count
    rows = []
    for model_name, probability_col in {
        "XGBoost": "xgboost_probability",
        "Logistic regression": "logistic_probability",
    }.items():
        values = metric_values(scored[TARGET], scored[probability_col], sample_weight=weights)
        for metric, estimate in values.items():
            rows.append({"model": model_name, "metric": metric, "estimate": estimate})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    frame, numeric_cols = load_frame()
    train = frame[frame["anchor_year_group"].isin(TRAIN_GROUPS)].copy()
    calib = frame[frame["anchor_year_group"].isin(CALIB_GROUPS)].copy()
    valid = frame[frame["anchor_year_group"].isin(VALID_GROUPS)].copy()

    model, platt, feature_cols = fit_logistic(train, calib, numeric_cols)
    valid_raw = predict_logistic_probability(model, valid[feature_cols])
    valid["logistic_probability"] = apply_platt(platt, valid_raw)
    scored = attach_xgb_predictions(valid)

    point_frames = []
    for model_name, probability_col in {
        "XGBoost": "xgboost_probability",
        "Logistic regression": "logistic_probability",
    }.items():
        for unit in ["landmark_row", "episode_maximum"]:
            point_frames.append(summarize_metric_frame(scored, model_name, probability_col, unit))
    points = pd.concat(point_frames, ignore_index=True)
    bootstrap = bootstrap_models(scored)
    comparison = add_intervals(points, bootstrap)

    delta_points = []
    for unit in ["landmark_row", "episode_maximum"]:
        local = points[points["analysis_unit"].eq(unit)].pivot(index="metric", columns="model", values="estimate")
        for metric, row in local.iterrows():
            delta_points.append({
                "model": "Logistic minus XGBoost",
                "analysis_unit": unit,
                "metric": metric,
                "estimate": row["Logistic regression"] - row["XGBoost"],
                "rows_or_episodes": np.nan,
                "positive_rows_or_episodes": np.nan,
            })
    comparison = pd.concat([
        comparison,
        add_intervals(pd.DataFrame(delta_points), bootstrap),
    ], ignore_index=True)

    review = pd.concat([
        review_policy_points(scored, "XGBoost", "xgboost_probability"),
        review_policy_points(scored, "Logistic regression", "logistic_probability"),
    ], ignore_index=True)
    weighted = episode_weighted_sensitivity(scored)

    scored[[
        "landmark_id", "episode_id", "subject_id", "anchor_year_group", "landmark_hour",
        "landmark_time", TARGET, "xgboost_probability", "logistic_probability",
    ]].to_csv(OUTPUT / "run34_same_frame_predictions.csv", index=False)
    comparison.to_csv(OUTPUT / "run34_same_frame_model_comparison.csv", index=False)
    review.to_csv(OUTPUT / "run34_same_frame_review_policy.csv", index=False)
    weighted.to_csv(OUTPUT / "run34_episode_weighted_sensitivity.csv", index=False)
    joblib.dump(model, MODELS / "run34_logistic_same_frame.joblib")
    joblib.dump(platt, MODELS / "run34_logistic_platt_calibrator.joblib")

    metadata = {
        "purpose": "Post-hoc same-frame simple comparator; not model reselection",
        "train_groups": TRAIN_GROUPS,
        "calibration_groups": CALIB_GROUPS,
        "evaluation_groups": VALID_GROUPS,
        "excluded_group": LOCKBOX_GROUP,
        "numeric_features": len(numeric_cols),
        "categorical_features": STATIC_CATEGORICAL,
        "train_rows": len(train),
        "calibration_rows": len(calib),
        "evaluation_rows": len(scored),
        "evaluation_episodes": int(scored["episode_id"].nunique()),
        "evaluation_positive_episodes": int(scored.loc[scored[TARGET].eq(1), "episode_id"].nunique()),
        "bootstrap_replicates": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
    }
    (OUTPUT / "run34_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison[comparison["metric"].isin(["roc_auc", "pr_auc", "brier_skill_score"])]
          .sort_values(["analysis_unit", "metric", "model"]).to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()

