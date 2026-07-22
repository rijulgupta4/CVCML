# %% Imports and paths

import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing Run 23 modeling dependency:", exc)
    print("Install required packages: pip install pandas numpy scikit-learn xgboost joblib pillow")
    sys.exit(1)


PROJECT_PATH = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 23 (v0.5 Label Sensitivity Modeling)"
PLOT_PATH = OUTPUT_PATH / "plots"
MODEL_PATH = OUTPUT_PATH / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = DATA_PATH / "v0_5_run20_dynamic_enriched_features.csv"
LABEL_FILE = DATA_PATH / "v0_5_run22_source_screened_daily_landmarks.csv"
RANDOM_STATE = 42

TRAIN_CORE_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013"]
CALIBRATION_YEAR_GROUPS = ["2014 - 2016"]
VALIDATION_YEAR_GROUPS = ["2017 - 2019"]
LOCKBOX_YEAR_GROUP = "2020 - 2022"

STATIC_NUMERIC = ["landmark_hour", "landmark_day", "anchor_age", "early_positive_culture"]
STATIC_CATEGORICAL = ["gender", "admission_type", "insurance", "race", "first_careunit"]
LAB_PREFIXES = ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"]

TARGET_SPECS = [
    {
        "label": "original_strict",
        "target_col": "future_strict_cvc_bsi_proxy_7d",
        "role": "current_baseline",
        "description": "Original v0.5 strict CVC-associated BSI proxy",
    },
    {
        "label": "primary_or_uncertain",
        "target_col": "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d",
        "role": "candidate_primary",
        "description": "Strict positives excluding concordant secondary-source culture evidence",
    },
    {
        "label": "primary_likely",
        "target_col": "future_strict_primary_likely_cvc_bsi_proxy_7d",
        "role": "high_specificity_sensitivity",
        "description": "High-specificity source-screened subset; too sparse for main target",
    },
]


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


def evaluate_predictions(df, split_name, target_label, target_role, target_col, calibration, probability_col):
    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df[probability_col].to_numpy()
    prevalence = float(np.mean(y_true))
    pr_auc = average_precision_score(y_true, y_prob)
    bss, brier, brier_reference = brier_skill_score(y_true, y_prob)
    cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "split": split_name,
        "target_label": target_label,
        "target_role": target_role,
        "target_col": target_col,
        "feature_set": "static_labs_vitals_therapy",
        "calibration": calibration,
        "score_name": f"static_labs_vitals_therapy_{calibration}_{target_label}",
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


def review_summary(flagged, all_rows, target_label, target_role, target_col, calibration, policy_name):
    tp_rows = int(flagged[target_col].sum())
    rows_reviewed = int(len(flagged))
    fp_rows = rows_reviewed - tp_rows
    positive_rows = int(all_rows[target_col].sum())
    positive_episodes = set(all_rows.loc[all_rows[target_col].eq(1), "episode_id"])
    captured_positive_episodes = set(flagged.loc[flagged[target_col].eq(1), "episode_id"])
    return {
        "target_label": target_label,
        "target_role": target_role,
        "calibration": calibration,
        "score_name": f"static_labs_vitals_therapy_{calibration}_{target_label}",
        "policy": policy_name,
        "rows_reviewed": rows_reviewed,
        "reviewed_row_fraction": rows_reviewed / len(all_rows) if len(all_rows) else np.nan,
        "true_positive_rows": tp_rows,
        "false_positive_rows": fp_rows,
        "precision_ppv": tp_rows / rows_reviewed if rows_reviewed else np.nan,
        "row_recall_sensitivity": tp_rows / positive_rows if positive_rows else np.nan,
        "positive_episodes_captured": len(captured_positive_episodes),
        "episode_recall_sensitivity": len(captured_positive_episodes) / len(positive_episodes) if positive_episodes else np.nan,
        "false_reviews_per_true_positive": fp_rows / tp_rows if tp_rows else np.nan,
    }


def make_topk_row_table(df, target_label, target_role, target_col, calibration, probability_col):
    rows = []
    ranked = df.sort_values(probability_col, ascending=False).reset_index(drop=True)
    for pct in [1, 2, 5, 10]:
        n_reviewed = max(1, int(np.ceil(len(ranked) * pct / 100)))
        rows.append(review_summary(
            ranked.head(n_reviewed), ranked, target_label, target_role, target_col, calibration, f"top_{pct}_percent_rows"
        ))
    for n_reviewed in [25, 50, 100, 250, 500]:
        rows.append(review_summary(
            ranked.head(min(n_reviewed, len(ranked))), ranked, target_label, target_role, target_col, calibration, f"top_{n_reviewed}_rows"
        ))
    return pd.DataFrame(rows)


def make_topk_episode_table(df, target_label, target_role, target_col, calibration, probability_col):
    episode_best = (
        df.sort_values(["episode_id", probability_col, "landmark_hour"], ascending=[True, False, True])
        .groupby("episode_id")
        .head(1)
        .copy()
    )
    episode_best["episode_positive"] = episode_best["episode_id"].map(
        df.groupby("episode_id")[target_col].max()
    ).fillna(0).astype(int)
    rows = []
    ranked = episode_best.sort_values(probability_col, ascending=False).reset_index(drop=True)
    total_positive_episodes = int(ranked["episode_positive"].sum())
    for pct in [1, 2, 5, 10]:
        n_reviewed = max(1, int(np.ceil(len(ranked) * pct / 100)))
        flagged = ranked.head(n_reviewed)
        tp = int(flagged["episode_positive"].sum())
        fp = len(flagged) - tp
        rows.append({
            "target_label": target_label,
            "target_role": target_role,
            "calibration": calibration,
            "score_name": f"static_labs_vitals_therapy_{calibration}_{target_label}",
            "policy": f"top_{pct}_percent_episodes_by_max_risk",
            "episodes_reviewed": int(len(flagged)),
            "reviewed_episode_fraction": len(flagged) / len(ranked) if len(ranked) else np.nan,
            "true_positive_episodes": tp,
            "false_positive_episodes": fp,
            "episode_precision_ppv": tp / len(flagged) if len(flagged) else np.nan,
            "episode_recall_sensitivity": tp / total_positive_episodes if total_positive_episodes else np.nan,
            "false_episode_reviews_per_true_positive": fp / tp if tp else np.nan,
        })
    return pd.DataFrame(rows)


def make_threshold_table(df, target_label, target_role, target_col, calibration, probability_col):
    rows = []
    for threshold in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
        flagged = df[df[probability_col] >= threshold]
        row = review_summary(flagged, df, target_label, target_role, target_col, calibration, f"threshold_{threshold:.3f}")
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def make_first_alert_table(df, target_label, target_role, target_col, calibration, probability_col):
    rows = []
    for threshold in [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
        alerts = (
            df[df[probability_col] >= threshold]
            .sort_values(["episode_id", "landmark_hour", probability_col], ascending=[True, True, False])
            .groupby("episode_id")
            .head(1)
        )
        row = review_summary(alerts, df, target_label, target_role, target_col, calibration, f"first_alert_threshold_{threshold:.3f}")
        row["threshold"] = threshold
        row["episodes_alerted"] = int(alerts["episode_id"].nunique())
        row["episode_alert_rate"] = alerts["episode_id"].nunique() / df["episode_id"].nunique() if df["episode_id"].nunique() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_calibration_deciles(df, target_label, target_role, target_col, calibration, probability_col):
    out = df[["landmark_id", target_col, probability_col]].copy()
    out["target_label"] = target_label
    out["target_role"] = target_role
    out["calibration"] = calibration
    out["score_name"] = f"static_labs_vitals_therapy_{calibration}_{target_label}"
    try:
        out["risk_decile"] = pd.qcut(out[probability_col], q=10, labels=False, duplicates="drop")
    except ValueError:
        out["risk_decile"] = pd.cut(out[probability_col], bins=10, labels=False)
    out["risk_decile"] = out["risk_decile"].astype("Int64")
    return (
        out
        .groupby(["target_label", "target_role", "calibration", "score_name", "risk_decile"], observed=True, dropna=True)
        .agg(
            rows=("landmark_id", "count"),
            mean_predicted_risk=(probability_col, "mean"),
            observed_event_rate=(target_col, "mean"),
            positive_rows=(target_col, "sum"),
        )
        .reset_index()
    )


def fit_xgboost(train_df, target_col, numeric_cols, categorical_cols):
    positives = int(train_df[target_col].sum())
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
    pipeline.fit(train_df[numeric_cols + categorical_cols], train_df[target_col].astype(int))
    return pipeline


def feature_importance_frame(pipeline, target_label):
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        if len(feature_names) == len(importance):
            return (
                pd.DataFrame({
                    "target_label": target_label,
                    "feature_set": "static_labs_vitals_therapy",
                    "feature": feature_names,
                    "importance": importance,
                })
                .sort_values(["target_label", "importance"], ascending=[True, False])
            )
    except Exception:
        pass
    return pd.DataFrame(columns=["target_label", "feature_set", "feature", "importance"])


# %% Simple Pillow plots

def load_font(size=28, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(40, True)
FONT_AXIS = load_font(27)
FONT_TICK = load_font(23)
COLORS = {
    "original_strict": (31, 119, 180),
    "primary_or_uncertain": (44, 160, 44),
    "primary_likely": (214, 39, 40),
}


def draw_bar_chart(df, value_col, title, y_label, output_file):
    img = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 180, 150, 1450, 770
    draw.text((800, 45), title, font=FONT_TITLE, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    y_max = float(np.ceil(df[value_col].max() * 12) / 10) if df[value_col].max() > 0 else 1.0
    for frac in np.linspace(0, 1, 6):
        y = bottom - frac * (bottom - top)
        val = y_max * frac
        draw.line((left - 10, y, left, y), fill="black", width=2)
        draw.text((left - 18, y), f"{val:.2f}", font=FONT_TICK, fill="black", anchor="rm")
        if frac > 0:
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
    labels = df["target_label"].tolist()
    n = len(labels)
    slot = (right - left) / n
    bar_w = slot * 0.55
    for idx, row in enumerate(df.itertuples()):
        cx = left + slot * (idx + 0.5)
        h = getattr(row, value_col) / y_max * (bottom - top)
        color = COLORS.get(row.target_label, (80, 80, 80))
        draw.rectangle((cx - bar_w / 2, bottom - h, cx + bar_w / 2, bottom), fill=color)
        draw.text((cx, bottom + 20), row.target_label.replace("_", "\n"), font=FONT_TICK, fill="black", anchor="ma")
        draw.text((cx, bottom - h - 12), f"{getattr(row, value_col):.3f}", font=FONT_TICK, fill="black", anchor="ma")
    draw.text((800, 940), "Label definition", font=FONT_AXIS, fill="black", anchor="ma")
    draw.text((left, 105), y_label, font=FONT_AXIS, fill="black", anchor="la")
    img.save(output_file)


# %% Load and merge features/labels

print("Loading Run 20 enriched feature matrix...", flush=True)
features = pd.read_csv(FEATURE_FILE, parse_dates=["landmark_time"])
label_cols = ["landmark_id"] + [spec["target_col"] for spec in TARGET_SPECS if spec["target_col"] != "future_strict_cvc_bsi_proxy_7d"]
labels = pd.read_csv(LABEL_FILE, usecols=label_cols)
features = features.merge(labels, on="landmark_id", how="left", validate="one_to_one")

for spec in TARGET_SPECS:
    features[spec["target_col"]] = features[spec["target_col"]].fillna(0).astype(int)

model_frame = features[
    features["run18_primary_model_frame"].eq(1)
    & features["split_role"].eq("development")
].copy()
lockbox_df = features[features["anchor_year_group"].eq(LOCKBOX_YEAR_GROUP)].copy()

lab_cols = sorted({c for c in features.columns if any(c.startswith(prefix) for prefix in LAB_PREFIXES)})
vital_cols = sorted({c for c in features.columns if c.startswith("vital_")})
therapy_cols = sorted({c for c in features.columns if c.startswith("abx_") or c.startswith("vaso_")})
numeric_cols = STATIC_NUMERIC + lab_cols + vital_cols + therapy_cols
categorical_cols = STATIC_CATEGORICAL
feature_cols = numeric_cols + categorical_cols


# %% Fit/evaluate one enriched model per label

metric_rows = []
topk_row_frames = []
topk_episode_frames = []
threshold_frames = []
first_alert_frames = []
decile_frames = []
prediction_frames = []
importance_frames = []
split_audit_rows = []
target_audit_rows = []

for spec in TARGET_SPECS:
    target_label = spec["label"]
    target_role = spec["role"]
    target_col = spec["target_col"]

    train_df = model_frame[model_frame["anchor_year_group"].isin(TRAIN_CORE_YEAR_GROUPS)].copy()
    calib_df = model_frame[model_frame["anchor_year_group"].isin(CALIBRATION_YEAR_GROUPS)].copy()
    val_df = model_frame[model_frame["anchor_year_group"].isin(VALIDATION_YEAR_GROUPS)].copy()

    print("", flush=True)
    print(f"Target {target_label}: {spec['description']}", flush=True)
    print(
        f"Train positives={int(train_df[target_col].sum())}, "
        f"calibration positives={int(calib_df[target_col].sum())}, "
        f"validation positives={int(val_df[target_col].sum())}",
        flush=True,
    )

    for split_name, split_df, year_groups in [
        ("train_core", train_df, ", ".join(TRAIN_CORE_YEAR_GROUPS)),
        ("calibration", calib_df, ", ".join(CALIBRATION_YEAR_GROUPS)),
        ("validation", val_df, ", ".join(VALIDATION_YEAR_GROUPS)),
    ]:
        split_audit_rows.append({
            "target_label": target_label,
            "target_role": target_role,
            "split": split_name,
            "anchor_year_groups": year_groups,
            "rows": int(len(split_df)),
            "positive_rows": int(split_df[target_col].sum()),
            "prevalence": float(split_df[target_col].mean()) if len(split_df) else np.nan,
            "episodes": int(split_df["episode_id"].nunique()),
            "patients": int(split_df["subject_id"].nunique()),
        })

    split_audit_rows.append({
        "target_label": target_label,
        "target_role": target_role,
        "split": "temporal_lockbox_audit_only",
        "anchor_year_groups": LOCKBOX_YEAR_GROUP,
        "rows": int(len(lockbox_df)),
        "positive_rows": np.nan,
        "prevalence": np.nan,
        "episodes": int(lockbox_df["episode_id"].nunique()),
        "patients": int(lockbox_df["subject_id"].nunique()),
    })

    target_audit_rows.append({
        "target_label": target_label,
        "target_role": target_role,
        "target_col": target_col,
        "description": spec["description"],
        "development_rows": int(len(model_frame)),
        "development_positive_rows": int(model_frame[target_col].sum()),
        "development_prevalence": float(model_frame[target_col].mean()),
        "development_positive_episodes": int(model_frame.loc[model_frame[target_col].eq(1), "episode_id"].nunique()),
    })

    pipeline = fit_xgboost(train_df, target_col, numeric_cols, categorical_cols)
    joblib.dump(pipeline, MODEL_PATH / f"run23_xgboost_static_labs_vitals_therapy_{target_label}.joblib")

    calib_raw = pipeline.predict_proba(calib_df[feature_cols])[:, 1]
    val_raw = pipeline.predict_proba(val_df[feature_cols])[:, 1]

    platt = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
    platt.fit(logit(calib_raw).reshape(-1, 1), calib_df[target_col].astype(int))
    joblib.dump(platt, MODEL_PATH / f"run23_platt_static_labs_vitals_therapy_{target_label}.joblib")

    keep_cols = [
        "landmark_id", "episode_id", "subject_id", "hadm_id", "stay_id",
        "anchor_year_group", "landmark_hour", "landmark_day", "landmark_time",
        target_col,
    ]
    calib_scored = calib_df[keep_cols].copy()
    val_scored = val_df[keep_cols].copy()
    calib_scored["target_label"] = target_label
    val_scored["target_label"] = target_label
    calib_scored["target_role"] = target_role
    val_scored["target_role"] = target_role
    calib_scored["target"] = calib_scored[target_col]
    val_scored["target"] = val_scored[target_col]
    calib_scored["raw_probability"] = calib_raw
    val_scored["raw_probability"] = val_raw
    calib_scored["platt_probability"] = apply_platt(platt, calib_raw)
    val_scored["platt_probability"] = apply_platt(platt, val_raw)

    for calibration, probability_col in [("raw", "raw_probability"), ("platt", "platt_probability")]:
        metric_rows.append(evaluate_predictions(calib_scored, "calibration", target_label, target_role, target_col, calibration, probability_col))
        metric_rows.append(evaluate_predictions(val_scored, "validation", target_label, target_role, target_col, calibration, probability_col))
        topk_row_frames.append(make_topk_row_table(val_scored, target_label, target_role, target_col, calibration, probability_col))
        topk_episode_frames.append(make_topk_episode_table(val_scored, target_label, target_role, target_col, calibration, probability_col))
        threshold_frames.append(make_threshold_table(val_scored, target_label, target_role, target_col, calibration, probability_col))
        first_alert_frames.append(make_first_alert_table(val_scored, target_label, target_role, target_col, calibration, probability_col))
        decile_frames.append(make_calibration_deciles(val_scored, target_label, target_role, target_col, calibration, probability_col))

    prediction_frames.append(val_scored.drop(columns=[target_col]).copy())
    importance_frames.append(feature_importance_frame(pipeline, target_label))


# %% Save outputs

model_comparison = pd.DataFrame(metric_rows)
target_audit = pd.DataFrame(target_audit_rows)
split_audit = pd.DataFrame(split_audit_rows)
topk_row_review = pd.concat(topk_row_frames, ignore_index=True)
topk_episode_review = pd.concat(topk_episode_frames, ignore_index=True)
threshold_policy = pd.concat(threshold_frames, ignore_index=True)
first_alert_policy = pd.concat(first_alert_frames, ignore_index=True)
calibration_deciles = pd.concat(decile_frames, ignore_index=True)
validation_predictions = pd.concat(prediction_frames, ignore_index=True)
feature_importance = pd.concat(importance_frames, ignore_index=True)

model_comparison_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_model_comparison.csv"
target_audit_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_target_audit.csv"
split_audit_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_split_audit.csv"
topk_row_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_topk_row_review.csv"
topk_episode_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_topk_episode_review.csv"
threshold_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_threshold_policy.csv"
first_alert_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_first_alert_policy.csv"
deciles_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_calibration_deciles.csv"
importance_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_feature_importance.csv"
predictions_file = OUTPUT_PATH / "v0_5_run23_label_sensitivity_validation_predictions.csv"
manifest_file = OUTPUT_PATH / "v0_5_run23_manifest.csv"

model_comparison.to_csv(model_comparison_file, index=False)
target_audit.to_csv(target_audit_file, index=False)
split_audit.to_csv(split_audit_file, index=False)
topk_row_review.to_csv(topk_row_file, index=False)
topk_episode_review.to_csv(topk_episode_file, index=False)
threshold_policy.to_csv(threshold_file, index=False)
first_alert_policy.to_csv(first_alert_file, index=False)
calibration_deciles.to_csv(deciles_file, index=False)
feature_importance.to_csv(importance_file, index=False)
validation_predictions.to_csv(predictions_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "feature_matrix", "path": str(FEATURE_FILE)},
    {"artifact": "source_screened_labels", "path": str(LABEL_FILE)},
    {"artifact": "model_comparison", "path": str(model_comparison_file)},
    {"artifact": "target_audit", "path": str(target_audit_file)},
    {"artifact": "split_audit", "path": str(split_audit_file)},
    {"artifact": "topk_row_review", "path": str(topk_row_file)},
    {"artifact": "topk_episode_review", "path": str(topk_episode_file)},
    {"artifact": "threshold_policy", "path": str(threshold_file)},
    {"artifact": "first_alert_policy", "path": str(first_alert_file)},
    {"artifact": "calibration_deciles", "path": str(deciles_file)},
    {"artifact": "feature_importance", "path": str(importance_file)},
    {"artifact": "validation_predictions", "path": str(predictions_file)},
])
manifest.to_csv(manifest_file, index=False)


# %% Plots

validation_metrics = model_comparison[
    model_comparison["split"].eq("validation")
    & model_comparison["calibration"].eq("platt")
].copy()
validation_metrics = validation_metrics.sort_values("target_label")

draw_bar_chart(
    validation_metrics,
    "pr_auc",
    "Run 23 Validation PR-AUC by Label Definition",
    "Validation PR-AUC",
    PLOT_PATH / "v0_5_run23_pr_auc_by_label.png",
)
draw_bar_chart(
    validation_metrics,
    "pr_auc_lift_over_prevalence",
    "Run 23 Validation Lift by Label Definition",
    "PR-AUC lift over prevalence",
    PLOT_PATH / "v0_5_run23_pr_auc_lift_by_label.png",
)

top5 = topk_row_review[
    topk_row_review["policy"].eq("top_5_percent_rows")
    & topk_row_review["calibration"].eq("platt")
].copy().sort_values("target_label")
draw_bar_chart(
    top5.rename(columns={"precision_ppv": "top5_ppv"}),
    "top5_ppv",
    "Run 23 Top 5% Row-Level PPV by Label",
    "Top 5% PPV",
    PLOT_PATH / "v0_5_run23_top5_ppv_by_label.png",
)


# %% Console summary

print("", flush=True)
print("Run 23 target audit:", flush=True)
print(target_audit.to_string(index=False), flush=True)
print("", flush=True)
print("Run 23 validation performance, Platt-calibrated:", flush=True)
print(
    validation_metrics[[
        "target_label",
        "target_role",
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
print("Run 23 top 5% row-level review, Platt-calibrated:", flush=True)
print(
    top5[[
        "target_label",
        "target_role",
        "rows_reviewed",
        "true_positive_rows",
        "precision_ppv",
        "row_recall_sensitivity",
        "episode_recall_sensitivity",
        "false_reviews_per_true_positive",
    ]].round(4).to_string(index=False),
    flush=True,
)
print("", flush=True)
print("Temporal lockbox remains held out; no lockbox predictions were generated.", flush=True)
print(f"Saved Run 23 outputs to: {OUTPUT_PATH}", flush=True)
print("Modeling 17 v0.5 Label Sensitivity complete.", flush=True)

