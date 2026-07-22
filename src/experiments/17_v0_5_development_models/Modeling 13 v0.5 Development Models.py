# %% Imports and paths

import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing v0.5 modeling dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost joblib")
    sys.exit(1)


MIMIC_PATH = Path(r"C:\path\to\mimic-iv")
PROJECT_PATH = Path(r"C:\path\to\CVCML")
HOSP = MIMIC_PATH / "hosp"
DATA_PATH = PROJECT_PATH / "data" / "v0_5"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 18 (v0.5 Development Models)"
MODEL_PATH = OUTPUT_PATH / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

LANDMARK_FILE = DATA_PATH / "v0_5_daily_landmarks.csv"
FEATURE_FILE = DATA_PATH / "v0_5_run18_development_features.csv"
V05_FILTERED_LABS_CACHE_FILE = DATA_PATH / "v0_5_run18_labs_long.pkl"
LAB_COVERAGE_FILE = OUTPUT_PATH / "v0_5_run18_lab_feature_coverage.csv"
TARGET_COL = "future_strict_cvc_bsi_proxy_7d"
RANDOM_STATE = 42
LOOKBACK_HOURS = 48
TRAIN_YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
VAL_YEAR_GROUPS = ["2017 - 2019"]
LOCKBOX_YEAR_GROUP = "2020 - 2022"

LAB_ITEMS = {
    "wbc": [51301],
    "lactate": [50813, 52442],
    "hemoglobin": [51222],
    "platelets": [51265],
    "creatinine": [50912],
}
ALL_LAB_IDS = [itemid for ids in LAB_ITEMS.values() for itemid in ids]
ID_TO_LAB = {itemid: lab for lab, ids in LAB_ITEMS.items() for itemid in ids}


# %% Helpers

def clip_prob(prob):
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)


def stable_sigmoid(score):
    score = np.clip(np.asarray(score, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-score))


def predict_positive_probability(pipeline, X):
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        transformed = pipeline.named_steps["preprocess"].transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        coef = np.asarray(model.coef_[0], dtype=float).reshape(1, -1)
        scores = (transformed * coef).sum(axis=1) + float(model.intercept_[0])
        return stable_sigmoid(scores)
    return pipeline.predict_proba(X)[:, 1]


def safe_roc_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def calibration_intercept_slope(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    prob = clip_prob(y_prob)
    logits = np.log(prob / (1 - prob)).reshape(-1, 1)
    cal_model = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
    cal_model.fit(logits, y_true)
    return float(cal_model.intercept_[0]), float(cal_model.coef_[0][0])


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


def make_topk_table(result_df, model_name, feature_set_name, split_name):
    rows = []
    total_positive = int(result_df[TARGET_COL].sum())
    ranked = result_df.sort_values("predicted_risk", ascending=False).reset_index(drop=True)

    for pct in [1, 2, 5, 10]:
        n_flagged = max(1, int(np.ceil(len(ranked) * pct / 100)))
        flagged = ranked.head(n_flagged)
        tp = int(flagged[TARGET_COL].sum())
        fp = n_flagged - tp
        rows.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "split": split_name,
            "review_policy": f"top_{pct}_percent_rows",
            "rows_reviewed": n_flagged,
            "true_positive_rows": tp,
            "false_positive_rows": fp,
            "precision_ppv": tp / n_flagged if n_flagged else np.nan,
            "recall_sensitivity": tp / total_positive if total_positive else np.nan,
            "false_alerts_per_true_positive": fp / tp if tp else np.nan,
            "total_positive_rows": total_positive,
            "total_rows": len(ranked),
        })

    for n_flagged in [25, 50, 100]:
        n_flagged = min(n_flagged, len(ranked))
        flagged = ranked.head(n_flagged)
        tp = int(flagged[TARGET_COL].sum())
        fp = n_flagged - tp
        rows.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "split": split_name,
            "review_policy": f"top_{n_flagged}_rows",
            "rows_reviewed": n_flagged,
            "true_positive_rows": tp,
            "false_positive_rows": fp,
            "precision_ppv": tp / n_flagged if n_flagged else np.nan,
            "recall_sensitivity": tp / total_positive if total_positive else np.nan,
            "false_alerts_per_true_positive": fp / tp if tp else np.nan,
            "total_positive_rows": total_positive,
            "total_rows": len(ranked),
        })

    return pd.DataFrame(rows)


def make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False, force_dense=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    onehot_kwargs = {"handle_unknown": "ignore", "min_frequency": 20}
    if force_dense:
        onehot_kwargs["sparse_output"] = False
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(**onehot_kwargs)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], sparse_threshold=0.0 if force_dense else 0.3)


def get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([])


def fit_and_evaluate(train_df, val_df, numeric_cols, categorical_cols, model_name, feature_set_name):
    y_train = train_df[TARGET_COL].astype(int)
    y_val = val_df[TARGET_COL].astype(int)

    if model_name == "Logistic Regression":
        preprocessor = make_preprocessor(
            numeric_cols,
            categorical_cols,
            scale_numeric=True,
            force_dense=True,
        )
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            solver="liblinear",
        )
    elif model_name == "XGBoost":
        preprocessor = make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        scale_pos_weight = negatives / positives if positives else 1.0
        estimator = XGBClassifier(
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
        )
    else:
        raise ValueError(model_name)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(train_df[numeric_cols + categorical_cols], y_train)

    val_prob = predict_positive_probability(pipeline, val_df[numeric_cols + categorical_cols])
    train_prob = predict_positive_probability(pipeline, train_df[numeric_cols + categorical_cols])

    rows = []
    prediction_frames = []
    for split_name, split_df, y_true, prob in [
        ("train", train_df, y_train, train_prob),
        ("validation", val_df, y_val, val_prob),
    ]:
        prevalence = float(np.mean(y_true))
        pr_auc = average_precision_score(y_true, prob)
        roc_auc = safe_roc_auc(y_true, prob)
        bss, brier, brier_reference = brier_skill_score(y_true, prob)
        cal_intercept, cal_slope = calibration_intercept_slope(y_true, prob)
        eo_ratio = expected_observed_ratio(y_true, prob)
        rows.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "split": split_name,
            "rows": int(len(split_df)),
            "positive_rows": int(np.sum(y_true)),
            "prevalence": prevalence,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "pr_auc_lift_over_prevalence": pr_auc / prevalence if prevalence else np.nan,
            "brier_score": brier,
            "brier_reference_prevalence": brier_reference,
            "brier_skill_score": bss,
            "calibration_intercept": cal_intercept,
            "calibration_slope": cal_slope,
            "expected_observed_ratio": eo_ratio,
        })

        pred_df = split_df[[
            "landmark_id",
            "episode_id",
            "subject_id",
            "hadm_id",
            "stay_id",
            "anchor_year_group",
            "landmark_hour",
            TARGET_COL,
        ]].copy()
        pred_df["model_name"] = model_name
        pred_df["feature_set"] = feature_set_name
        pred_df["split"] = split_name
        pred_df["predicted_risk"] = prob
        prediction_frames.append(pred_df)

    model_file = MODEL_PATH / f"run18_{feature_set_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(pipeline, model_file)

    feature_importance = pd.DataFrame()
    if model_name == "XGBoost":
        fitted_preprocessor = pipeline.named_steps["preprocess"]
        feature_names = get_feature_names(fitted_preprocessor)
        importances = pipeline.named_steps["model"].feature_importances_
        if len(feature_names) == len(importances):
            feature_importance = (
                pd.DataFrame({
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "feature": feature_names,
                    "importance": importances,
                })
                .sort_values("importance", ascending=False)
            )

    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True), feature_importance


def load_v05_filtered_labs(landmark_windows):
    if V05_FILTERED_LABS_CACHE_FILE.exists():
        print(f"  Loading cached v0.5 filtered labs: {V05_FILTERED_LABS_CACHE_FILE}", flush=True)
        return pd.read_pickle(V05_FILTERED_LABS_CACHE_FILE)

    cohort_subjects = set(landmark_windows["subject_id"].astype("int64"))
    cohort_hadms = set(landmark_windows["hadm_id"].astype("int64"))
    min_time = landmark_windows["lookback_start"].min()
    max_time = landmark_windows["landmark_time"].max()

    retained_chunks = []
    total_rows_seen = 0
    retained_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(
        HOSP / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        chunksize=500000,
        low_memory=False,
        parse_dates=["charttime"],
    ), start=1):
        total_rows_seen += len(chunk)
        chunk = chunk[chunk["hadm_id"].notna()].copy()
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

        filtered = chunk[
            chunk["subject_id"].isin(cohort_subjects)
            & chunk["hadm_id"].isin(cohort_hadms)
            & chunk["itemid"].isin(ALL_LAB_IDS)
            & chunk["valuenum"].notna()
            & (chunk["charttime"] >= min_time)
            & (chunk["charttime"] <= max_time)
        ].copy()
        if len(filtered):
            filtered["lab_name"] = filtered["itemid"].map(ID_TO_LAB)
            retained_chunks.append(filtered[["subject_id", "hadm_id", "charttime", "lab_name", "valuenum"]])
            retained_rows += len(filtered)

        if chunk_idx % 5 == 0:
            print(
                f"    labevents chunks scanned: {chunk_idx:,} | "
                f"rows seen: {total_rows_seen:,} | retained: {retained_rows:,}",
                flush=True,
            )

    if retained_chunks:
        labevents = pd.concat(retained_chunks, ignore_index=True)
    else:
        labevents = pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "lab_name", "valuenum"])

    labevents.to_pickle(V05_FILTERED_LABS_CACHE_FILE)
    print(f"  Cached v0.5 filtered labs: {V05_FILTERED_LABS_CACHE_FILE}", flush=True)
    return labevents


def aggregate_lab_features_by_admission(labevents, landmark_windows):
    if len(labevents) == 0:
        empty = pd.DataFrame({"landmark_id": landmark_windows["landmark_id"].head(0)})
        coverage = {
            "filtered_source_lab_rows": 0,
            "windowed_lab_rows": 0,
            "landmark_rows": int(len(landmark_windows)),
            "landmarks_with_any_labs_48h": 0,
            "admission_groups_with_labs": 0,
        }
        return empty, coverage

    labevents = labevents.sort_values(["subject_id", "hadm_id", "charttime"])
    landmark_windows = landmark_windows.sort_values(["subject_id", "hadm_id", "landmark_time"])
    lab_groups = {
        key: group[["charttime", "lab_name", "valuenum"]].sort_values("charttime")
        for key, group in labevents.groupby(["subject_id", "hadm_id"], sort=False)
    }

    feature_frames = []
    windowed_row_count = 0
    admissions_with_labs = 0
    cross_merge_limit = 1_500_000
    grouped_windows = landmark_windows.groupby(["subject_id", "hadm_id"], sort=False)

    for group_idx, (key, windows) in enumerate(grouped_windows, start=1):
        events = lab_groups.get(key)
        if events is None or len(events) == 0:
            continue
        admissions_with_labs += 1

        if len(events) * len(windows) <= cross_merge_limit:
            merged = events.merge(
                windows[["landmark_id", "landmark_time", "lookback_start"]],
                how="cross",
            )
            merged = merged[
                (merged["charttime"] >= merged["lookback_start"])
                & (merged["charttime"] <= merged["landmark_time"])
            ].copy()
        else:
            window_chunks = []
            for window in windows.itertuples(index=False):
                window_events = events[
                    (events["charttime"] >= window.lookback_start)
                    & (events["charttime"] <= window.landmark_time)
                ].copy()
                if len(window_events):
                    window_events["landmark_id"] = window.landmark_id
                    window_events["landmark_time"] = window.landmark_time
                    window_chunks.append(window_events)
            merged = pd.concat(window_chunks, ignore_index=True) if window_chunks else pd.DataFrame()

        if len(merged):
            merged["hours_since_lab"] = (
                (merged["landmark_time"] - merged["charttime"]).dt.total_seconds() / 3600
            )
            merged = merged.sort_values(["landmark_id", "lab_name", "charttime"])
            grouped_features = (
                merged
                .groupby(["landmark_id", "lab_name"])
                .agg(
                    mean_val=("valuenum", "mean"),
                    last_val=("valuenum", "last"),
                    first_val=("valuenum", "first"),
                    lab_count=("valuenum", "size"),
                    hours_since_last=("hours_since_lab", "min"),
                )
                .reset_index()
            )
            grouped_features["trend"] = grouped_features["last_val"] - grouped_features["first_val"]
            feature_frames.append(grouped_features)
            windowed_row_count += len(merged)

        if group_idx % 1000 == 0:
            print(
                f"    admission groups processed: {group_idx:,} | "
                f"windowed lab rows: {windowed_row_count:,}",
                flush=True,
            )

    if feature_frames:
        lab_features = pd.concat(feature_frames, ignore_index=True)
        lab_pivot = lab_features.pivot_table(
            index="landmark_id",
            columns="lab_name",
            values=["mean_val", "last_val", "trend", "lab_count", "hours_since_last"],
        )
        lab_pivot.columns = [
            f"{lab}_{metric.replace('_val', '')}"
            for metric, lab in lab_pivot.columns
        ]
        lab_pivot = lab_pivot.reset_index()
    else:
        lab_pivot = pd.DataFrame({"landmark_id": landmark_windows["landmark_id"].head(0)})

    coverage = {
        "filtered_source_lab_rows": int(len(labevents)),
        "windowed_lab_rows": int(windowed_row_count),
        "landmark_rows": int(len(landmark_windows)),
        "landmarks_with_any_labs_48h": int(lab_pivot["landmark_id"].nunique()) if "landmark_id" in lab_pivot else 0,
        "admission_groups_with_labs": int(admissions_with_labs),
    }
    return lab_pivot, coverage


# %% Load landmarks and build lab features

print("Loading v0.5 daily landmark frame...")
landmarks = pd.read_csv(
    LANDMARK_FILE,
    parse_dates=[
        "landmark_time",
        "exposure_start",
        "exposure_end_observed",
        "strict_proxy_culture_time",
        "broad_proxy_culture_time",
    ],
)
landmarks[TARGET_COL] = landmarks[TARGET_COL].astype(int)
landmarks["lookback_start"] = landmarks["landmark_time"] - pd.Timedelta(hours=LOOKBACK_HOURS)

print(f"  Landmark rows: {len(landmarks):,}")
print(f"  Development rows: {int(landmarks['split_role'].eq('development').sum()):,}")
print(f"  Temporal lockbox rows held out: {int(landmarks['split_role'].eq('temporal_lockbox').sum()):,}")

primary_frame_mask = (
    landmarks[TARGET_COL].eq(1)
    | landmarks["full_7d_followup_observed"].eq(1)
)
landmarks["run18_primary_model_frame"] = primary_frame_mask.astype(int)

print("")
print("Loading target labevents and aggregating 48h lookback features...")
landmark_windows = landmarks[[
    "landmark_id",
    "subject_id",
    "hadm_id",
    "landmark_time",
    "lookback_start",
]].copy()
landmark_windows = landmark_windows.dropna(subset=["subject_id", "hadm_id"]).copy()
landmark_windows["subject_id"] = landmark_windows["subject_id"].astype("int64")
landmark_windows["hadm_id"] = landmark_windows["hadm_id"].astype("int64")

labevents = load_v05_filtered_labs(landmark_windows)
print(f"  Filtered source lab rows: {len(labevents):,}", flush=True)

lab_pivot, lab_coverage = aggregate_lab_features_by_admission(labevents, landmark_windows)
pd.DataFrame([lab_coverage]).to_csv(LAB_COVERAGE_FILE, index=False)
print(f"  Windowed lab rows: {lab_coverage['windowed_lab_rows']:,}", flush=True)
print(f"  Landmarks with any 48h lab: {lab_coverage['landmarks_with_any_labs_48h']:,}", flush=True)

features = landmarks.merge(lab_pivot, on="landmark_id", how="left")
lab_cols = [c for c in features.columns if any(c.startswith(lab) for lab in LAB_ITEMS.keys())]
for lab in LAB_ITEMS.keys():
    last_col = f"{lab}_last"
    measured_col = f"{lab}_measured_48h"
    features[measured_col] = features[last_col].notna().astype(int) if last_col in features.columns else 0

for col in [c for c in lab_cols if c.endswith("_count")]:
    features[col] = features[col].fillna(0)

features.to_csv(FEATURE_FILE, index=False)

print(f"  Feature matrix saved: {FEATURE_FILE}")
print(f"  Feature matrix shape: {features.shape}")


# %% Define modeling frames and feature sets

frame_audit = pd.DataFrame([{
    "frame": "all_daily_landmarks",
    "rows": int(len(features)),
    "positive_rows": int(features[TARGET_COL].sum()),
    "positive_rate": float(features[TARGET_COL].mean()),
    "development_rows": int(features["split_role"].eq("development").sum()),
    "lockbox_rows_held_out": int(features["split_role"].eq("temporal_lockbox").sum()),
}, {
    "frame": "primary_event_or_full_7d_followup",
    "rows": int(features["run18_primary_model_frame"].sum()),
    "positive_rows": int(features.loc[features["run18_primary_model_frame"].eq(1), TARGET_COL].sum()),
    "positive_rate": float(features.loc[features["run18_primary_model_frame"].eq(1), TARGET_COL].mean()),
    "development_rows": int(features["split_role"].eq("development").mul(features["run18_primary_model_frame"].eq(1)).sum()),
    "lockbox_rows_held_out": int(features["split_role"].eq("temporal_lockbox").mul(features["run18_primary_model_frame"].eq(1)).sum()),
}])

model_frame = features[
    features["run18_primary_model_frame"].eq(1)
    & features["split_role"].eq("development")
].copy()

train_df = model_frame[model_frame["anchor_year_group"].isin(TRAIN_YEAR_GROUPS)].copy()
val_df = model_frame[model_frame["anchor_year_group"].isin(VAL_YEAR_GROUPS)].copy()
lockbox_audit = features[features["split_role"].eq("temporal_lockbox")].groupby("anchor_year_group").agg(
    rows=("landmark_id", "count"),
    primary_frame_rows=("run18_primary_model_frame", "sum"),
    positive_rows=(TARGET_COL, "sum"),
    episodes=("episode_id", "nunique"),
    patients=("subject_id", "nunique"),
).reset_index()

print("")
print("Run 18 modeling frame:")
print(f"  Train rows: {len(train_df):,} | positives: {int(train_df[TARGET_COL].sum()):,} ({train_df[TARGET_COL].mean() * 100:.2f}%)")
print(f"  Val rows:   {len(val_df):,} | positives: {int(val_df[TARGET_COL].sum()):,} ({val_df[TARGET_COL].mean() * 100:.2f}%)")
print(f"  Lockbox rows held out: {int(features['split_role'].eq('temporal_lockbox').sum()):,}")

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

lab_numeric = [
    c for c in features.columns
    if any(c.startswith(lab) for lab in LAB_ITEMS.keys())
]
lab_numeric = sorted(set(lab_numeric))

feature_sets = [
    {
        "feature_set": "static_context",
        "numeric_cols": static_numeric,
        "categorical_cols": static_categorical,
    },
    {
        "feature_set": "static_context_labs_48h",
        "numeric_cols": static_numeric + lab_numeric,
        "categorical_cols": static_categorical,
    },
]

feature_audit_rows = []
for feature_set in feature_sets:
    for col in feature_set["numeric_cols"] + feature_set["categorical_cols"]:
        feature_audit_rows.append({
            "feature_set": feature_set["feature_set"],
            "feature": col,
            "role": "numeric" if col in feature_set["numeric_cols"] else "categorical",
            "missing_rate_train": float(train_df[col].isna().mean()) if col in train_df.columns else np.nan,
            "missing_rate_validation": float(val_df[col].isna().mean()) if col in val_df.columns else np.nan,
        })
feature_audit = pd.DataFrame(feature_audit_rows)


# %% Fit constrained development models

print("")
print("Fitting constrained Run 18 development models...")
comparison_frames = []
prediction_frames = []
importance_frames = []

for feature_set in feature_sets:
    for model_name in ["Logistic Regression", "XGBoost"]:
        print(f"  {model_name} | {feature_set['feature_set']}")
        metrics, predictions, importance = fit_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            numeric_cols=feature_set["numeric_cols"],
            categorical_cols=feature_set["categorical_cols"],
            model_name=model_name,
            feature_set_name=feature_set["feature_set"],
        )
        comparison_frames.append(metrics)
        prediction_frames.append(predictions)
        if len(importance):
            importance_frames.append(importance)

model_comparison = pd.concat(comparison_frames, ignore_index=True)
predictions = pd.concat(prediction_frames, ignore_index=True)
feature_importance = (
    pd.concat(importance_frames, ignore_index=True)
    if importance_frames
    else pd.DataFrame(columns=["model_name", "feature_set", "feature", "importance"])
)

topk_tables = []
for (model_name, feature_set, split), group in predictions.groupby(["model_name", "feature_set", "split"]):
    if split == "validation":
        topk_tables.append(make_topk_table(group, model_name, feature_set, split))
topk_review = pd.concat(topk_tables, ignore_index=True) if topk_tables else pd.DataFrame()


# %% Save outputs

model_comparison_file = OUTPUT_PATH / "v0_5_run18_development_model_comparison.csv"
topk_file = OUTPUT_PATH / "v0_5_run18_topk_review_table.csv"
feature_audit_file = OUTPUT_PATH / "v0_5_run18_feature_audit.csv"
frame_audit_file = OUTPUT_PATH / "v0_5_run18_modeling_frame_audit.csv"
lockbox_audit_file = OUTPUT_PATH / "v0_5_run18_lockbox_holdout_audit.csv"
feature_importance_file = OUTPUT_PATH / "v0_5_run18_xgboost_feature_importance.csv"
predictions_file = OUTPUT_PATH / "v0_5_run18_development_predictions.csv"

model_comparison.to_csv(model_comparison_file, index=False)
topk_review.to_csv(topk_file, index=False)
feature_audit.to_csv(feature_audit_file, index=False)
frame_audit.to_csv(frame_audit_file, index=False)
lockbox_audit.to_csv(lockbox_audit_file, index=False)
feature_importance.to_csv(feature_importance_file, index=False)
predictions.to_csv(predictions_file, index=False)

manifest = pd.DataFrame([
    {"artifact": "feature_matrix", "path": str(FEATURE_FILE)},
    {"artifact": "lab_feature_coverage", "path": str(LAB_COVERAGE_FILE)},
    {"artifact": "model_comparison", "path": str(model_comparison_file)},
    {"artifact": "topk_review_table", "path": str(topk_file)},
    {"artifact": "feature_audit", "path": str(feature_audit_file)},
    {"artifact": "modeling_frame_audit", "path": str(frame_audit_file)},
    {"artifact": "lockbox_holdout_audit", "path": str(lockbox_audit_file)},
    {"artifact": "xgboost_feature_importance", "path": str(feature_importance_file)},
    {"artifact": "development_predictions", "path": str(predictions_file)},
])
manifest_file = OUTPUT_PATH / "v0_5_run18_development_models_manifest.csv"
manifest.to_csv(manifest_file, index=False)


# %% Console summary

print("")
print("Run 18 development model comparison:")
print(
    model_comparison[
        model_comparison["split"].eq("validation")
    ][[
        "model_name",
        "feature_set",
        "rows",
        "positive_rows",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "pr_auc_lift_over_prevalence",
        "brier_skill_score",
        "calibration_slope",
        "expected_observed_ratio",
    ]].round(4).to_string(index=False)
)
print("")
print("Validation top-k review:")
print(
    topk_review[
        topk_review["review_policy"].isin(["top_1_percent_rows", "top_5_percent_rows", "top_100_rows"])
    ][[
        "model_name",
        "feature_set",
        "review_policy",
        "rows_reviewed",
        "true_positive_rows",
        "precision_ppv",
        "recall_sensitivity",
        "false_alerts_per_true_positive",
    ]].round(4).to_string(index=False)
)
print("")
print("Lockbox holdout audit only; no lockbox predictions were generated:")
print(lockbox_audit.to_string(index=False))
print("")
print(f"Saved Run 18 outputs to: {OUTPUT_PATH}")
print("Modeling 13 v0.5 Development Models complete.")

