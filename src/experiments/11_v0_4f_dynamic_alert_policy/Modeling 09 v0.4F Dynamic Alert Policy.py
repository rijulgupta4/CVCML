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
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from xgboost import XGBClassifier
except ImportError as exc:
    print("Missing dynamic modeling dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy scikit-learn xgboost matplotlib joblib")
    sys.exit(1)


PROJECT_PATH = r"C:\path\to\CVCML"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "v0_4e")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "Outputs", "Run 12 (v0.4F Dynamic Alert Policy)")
MODEL_PATH = os.path.join(OUTPUT_PATH, "models")
PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

FEATURE_FILE = os.path.join(DATA_PATH, "clabsi_landmark_features_v0_4e.csv")
RANDOM_STATE = 42
XGB_N_JOBS = 2
HORIZONS = [48, 72, 168]
LABEL_FRAMES = ["standard", "gray_zone_excluded"]
GRAY_ZONE_MAX_HOURS = 168
MIN_RECALL_TARGET = 0.80
ALERT_CAPS_PER_100_ASSESSMENTS = [10, 20, 30, 50]
COOLDOWN_HOURS = [48, 72]
TOP_RISK_PCTS = [1, 2, 5, 10]


# %% Helpers

def clean_name(name):
    return (
        str(name).lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def unique_cols(cols):
    return sorted(dict.fromkeys([col for col in cols if col is not None]))


def safe_roc_auc(y_true, y_prob):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


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

    return (
        df[df[subject_col].isin(train_subjects)].copy(),
        df[df[subject_col].isin(val_subjects)].copy(),
        df[df[subject_col].isin(test_subjects)].copy(),
    )


def make_xgboost(y_train):
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        scale_pos_weight=negative / max(positive, 1),
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )


def make_label_frame(df, horizon, label_frame):
    target_col = f"future_clabsi_{horizon}h"
    out = df.copy()
    out["run12_target"] = out[target_col].astype(int)
    out["run12_label_frame"] = label_frame

    if label_frame == "standard":
        return out
    if label_frame == "gray_zone_excluded":
        time_to_event = pd.to_numeric(out["time_to_event_hours"], errors="coerce")
        gray_zone = (
            out["run12_target"].eq(0)
            & time_to_event.notna()
            & time_to_event.gt(horizon)
            & time_to_event.le(GRAY_ZONE_MAX_HOURS)
        )
        return out.loc[~gray_zone].copy()
    raise ValueError(f"Unknown label frame: {label_frame}")


def threshold_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "n": int(n),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision_ppv": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "alerts": int(pred.sum()),
        "alerts_per_100_assessments": pred.sum() / n * 100 if n else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
    }


def select_recall_threshold(y_true, y_prob, min_recall=MIN_RECALL_TARGET):
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


def select_alert_cap_threshold(y_true, y_prob, cap_per_100):
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    thresholds = np.unique(np.append(thresholds, np.arange(0.01, 0.91, 0.01)))
    rows = []
    for threshold in thresholds:
        row = threshold_metrics(y_true, y_prob, threshold)
        if row["alerts_per_100_assessments"] <= cap_per_100:
            rows.append(row)
    if not rows:
        return float(np.max(y_prob) + 1e-6)
    candidates = pd.DataFrame(rows)
    best = candidates.sort_values(
        ["recall_sensitivity", "precision_ppv", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["threshold"])


def fit_calibrators(y_val, val_prob):
    calibrators = {"raw": None}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, y_val)
    calibrators["isotonic"] = iso

    platt = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    platt.fit(val_prob.reshape(-1, 1), y_val)
    calibrators["platt"] = platt
    return calibrators


def apply_calibrator(calibrator_name, calibrator, prob):
    if calibrator_name == "raw":
        return prob
    if calibrator_name == "isotonic":
        return calibrator.predict(prob)
    if calibrator_name == "platt":
        return calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
    raise ValueError(calibrator_name)


def make_calibration_deciles(y_true, y_prob, label_frame, horizon, model_name, score_version):
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
    out["label_frame"] = label_frame
    out["horizon_hours"] = horizon
    out["model"] = model_name
    out["score_version"] = score_version
    return out


def add_physiology_features(df):
    out = df.copy()
    made = []

    def add_col(name, series):
        out[name] = series.replace([np.inf, -np.inf], np.nan)
        made.append(name)

    for window in [24, 48]:
        temp_max = out.get(f"temperature_c_max_{window}h")
        temp_min = out.get(f"temperature_c_min_{window}h")
        hr_max = out.get(f"heart_rate_max_{window}h")
        rr_max = out.get(f"respiratory_rate_max_{window}h")
        map_min = out.get(f"map_min_{window}h")
        sbp_min = out.get(f"sbp_min_{window}h")
        lactate_max = out.get(f"lactate_max_{window}h")
        lactate_trend = out.get(f"lactate_trend_{window}h")
        platelets_min = out.get(f"platelets_min_{window}h")
        platelets_trend = out.get(f"platelets_trend_{window}h")
        wbc_max = out.get(f"wbc_max_{window}h")
        wbc_min = out.get(f"wbc_min_{window}h")

        if temp_max is not None:
            add_col(f"phys_fever_excess_{window}h", (temp_max - 38.0).clip(lower=0))
        if temp_min is not None:
            add_col(f"phys_hypothermia_excess_{window}h", (36.0 - temp_min).clip(lower=0))
        if temp_max is not None and temp_min is not None:
            add_col(f"phys_temperature_range_{window}h", temp_max - temp_min)
        if hr_max is not None:
            add_col(f"phys_tachycardia_excess_{window}h", (hr_max - 110.0).clip(lower=0))
        if rr_max is not None:
            add_col(f"phys_tachypnea_excess_{window}h", (rr_max - 22.0).clip(lower=0))
        if map_min is not None:
            add_col(f"phys_map_hypotension_depth_{window}h", (65.0 - map_min).clip(lower=0))
        if sbp_min is not None:
            add_col(f"phys_sbp_hypotension_depth_{window}h", (90.0 - sbp_min).clip(lower=0))
        if lactate_max is not None:
            add_col(f"phys_lactate_excess_{window}h", (lactate_max - 2.0).clip(lower=0))
        if lactate_trend is not None:
            add_col(f"phys_lactate_rise_{window}h", lactate_trend.clip(lower=0))
        if platelets_min is not None:
            add_col(f"phys_thrombocytopenia_depth_{window}h", (150.0 - platelets_min).clip(lower=0))
        if platelets_trend is not None:
            add_col(f"phys_platelet_drop_{window}h", (-platelets_trend).clip(lower=0))
        if wbc_max is not None:
            add_col(f"phys_wbc_high_excess_{window}h", (wbc_max - 12.0).clip(lower=0))
        if wbc_min is not None:
            add_col(f"phys_wbc_low_depth_{window}h", (4.0 - wbc_min).clip(lower=0))

    components = [col for col in made if col.startswith("phys_")]
    for window in [24, 48]:
        window_components = [col for col in components if col.endswith(f"_{window}h")]
        if window_components:
            ranks = out[window_components].rank(pct=True)
            add_col(f"phys_composite_instability_rank_{window}h", ranks.mean(axis=1))

    return out, made


def policy_first_alert_per_stay(test_df, target_col, score_col, threshold):
    alerts = (
        test_df[test_df[score_col] >= threshold]
        .sort_values(["stay_id", "landmark_hour"])
        .groupby("stay_id", as_index=False)
        .head(1)
        .copy()
    )
    positive_stays = set(test_df.loc[test_df[target_col].eq(1), "stay_id"].unique())
    alerted_stays = set(alerts["stay_id"].unique())
    caught_stays = set(alerts.loc[alerts[target_col].eq(1), "stay_id"].unique())
    false_alerts = int((alerts[target_col] == 0).sum())
    tp = len(caught_stays)
    fp = false_alerts
    fn = len(positive_stays - caught_stays)
    tn = test_df["stay_id"].nunique() - tp - fp - fn
    n_stays = test_df["stay_id"].nunique()
    return {
        "policy": "first_alert_per_stay",
        "n_units": int(n_stays),
        "alerts": int(len(alerts)),
        "alerts_per_100_stays": len(alerts) / n_stays * 100 if n_stays else np.nan,
        "recall_sensitivity": tp / len(positive_stays) if positive_stays else np.nan,
        "precision_ppv": tp / len(alerts) if len(alerts) else 0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(max(tn, 0)),
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
    }


def policy_max_risk_one_per_stay(test_df, target_col, score_col, threshold):
    max_rows = (
        test_df.sort_values(["stay_id", score_col], ascending=[True, False])
        .groupby("stay_id", as_index=False)
        .head(1)
        .copy()
    )
    alerts = max_rows[max_rows[score_col] >= threshold]
    positive_stays = set(test_df.loc[test_df[target_col].eq(1), "stay_id"].unique())
    caught_stays = set(alerts.loc[alerts[target_col].eq(1), "stay_id"].unique())
    tp = len(caught_stays)
    fp = int((alerts[target_col] == 0).sum())
    fn = len(positive_stays - caught_stays)
    n_stays = test_df["stay_id"].nunique()
    tn = n_stays - tp - fp - fn
    return {
        "policy": "max_risk_one_per_stay",
        "n_units": int(n_stays),
        "alerts": int(len(alerts)),
        "alerts_per_100_stays": len(alerts) / n_stays * 100 if n_stays else np.nan,
        "recall_sensitivity": tp / len(positive_stays) if positive_stays else np.nan,
        "precision_ppv": tp / len(alerts) if len(alerts) else 0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(max(tn, 0)),
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
    }


def policy_cooldown(test_df, target_col, score_col, threshold, cooldown_hours):
    alerts = []
    for _, group in test_df.sort_values(["stay_id", "landmark_hour"]).groupby("stay_id"):
        last_alert_hour = -np.inf
        for _, row in group.iterrows():
            if row[score_col] >= threshold and row["landmark_hour"] >= last_alert_hour + cooldown_hours:
                alerts.append(row)
                last_alert_hour = row["landmark_hour"]
    alerts = pd.DataFrame(alerts)
    positive_stays = set(test_df.loc[test_df[target_col].eq(1), "stay_id"].unique())
    if len(alerts) == 0:
        tp = 0
        fp = 0
        caught_stays = set()
    else:
        caught_stays = set(alerts.loc[alerts[target_col].eq(1), "stay_id"].unique())
        tp = len(caught_stays)
        fp = int((alerts[target_col] == 0).sum())
    fn = len(positive_stays - caught_stays)
    n_stays = test_df["stay_id"].nunique()
    tn = n_stays - tp - fp - fn
    return {
        "policy": f"cooldown_{cooldown_hours}h",
        "n_units": int(n_stays),
        "alerts": int(len(alerts)),
        "alerts_per_100_stays": len(alerts) / n_stays * 100 if n_stays else np.nan,
        "recall_sensitivity": tp / len(positive_stays) if positive_stays else np.nan,
        "precision_ppv": tp / len(alerts) if len(alerts) else 0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(max(tn, 0)),
        "false_alerts_per_true_positive": fp / tp if tp else np.inf,
    }


def evaluate_top_risk(test_df, target_col, score_col, pct, unit):
    if unit == "assessment":
        ranked = test_df.sort_values(score_col, ascending=False).copy()
        n_select = max(1, int(np.ceil(len(ranked) * pct / 100)))
        selected = ranked.head(n_select)
        y_true = ranked[target_col].astype(int)
        positives = int(y_true.sum())
        tp = int(selected[target_col].sum())
        fp = int(len(selected) - tp)
        return {
            "unit": unit,
            "top_percent": pct,
            "n_units": int(len(ranked)),
            "selected": int(len(selected)),
            "base_rate": positives / len(ranked) if len(ranked) else np.nan,
            "recall_sensitivity": tp / positives if positives else np.nan,
            "precision_ppv": tp / len(selected) if len(selected) else 0,
            "false_alerts_per_true_positive": fp / tp if tp else np.inf,
        }
    if unit == "stay":
        stay = (
            test_df.sort_values(["stay_id", score_col], ascending=[True, False])
            .groupby("stay_id", as_index=False)
            .head(1)
            .copy()
        )
        n_select = max(1, int(np.ceil(len(stay) * pct / 100)))
        selected = stay.sort_values(score_col, ascending=False).head(n_select)
        positives = int(stay[target_col].sum())
        tp = int(selected[target_col].sum())
        fp = int(len(selected) - tp)
        return {
            "unit": unit,
            "top_percent": pct,
            "n_units": int(len(stay)),
            "selected": int(len(selected)),
            "base_rate": positives / len(stay) if len(stay) else np.nan,
            "recall_sensitivity": tp / positives if positives else np.nan,
            "precision_ppv": tp / len(selected) if len(selected) else 0,
            "false_alerts_per_true_positive": fp / tp if tp else np.inf,
        }
    raise ValueError(unit)


# %% Load matrix and add derived physiology features

print("Loading v0.4E feature matrix...")
df_raw = pd.read_csv(FEATURE_FILE)
df_raw, physiology_cols = add_physiology_features(df_raw)
print(f"Dataset shape with derived physiology features: {df_raw.shape}")
print(f"Derived physiology features: {len(physiology_cols)}")
print(f"Unique stays: {df_raw['stay_id'].nunique():,}")
print(f"Unique patients: {df_raw['subject_id'].nunique():,}")


# %% Feature sets

target_cols = [f"future_clabsi_{h}h" for h in [24, 48, 72, 168]]
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
meta_cols += [c for c in df_raw.columns if c.startswith("prediction_window_end_")]
meta_cols += target_cols
meta_cols = [col for col in meta_cols if col in df_raw.columns]

numeric_cols = [
    col for col in df_raw.columns
    if col not in meta_cols and pd.api.types.is_numeric_dtype(df_raw[col])
]

therapy_prefixes = ("systemic_antibiotic_", "vasopressor_", "therapy_")
therapy_cols = [col for col in numeric_cols if col.startswith(therapy_prefixes)]
intensity_tokens = ("_count_", "_hours_since_last_", "_measured_")
lab_vital_intensity_cols = [
    col for col in numeric_cols
    if any(token in col for token in intensity_tokens) and not col.startswith(therapy_prefixes)
]
value_cols = [col for col in numeric_cols if col not in lab_vital_intensity_cols]
non_therapy_value_cols = [col for col in value_cols if not col.startswith(therapy_prefixes)]
value_no_site_cols = [col for col in value_cols if col != "site_known"]
non_therapy_no_site_cols = [col for col in non_therapy_value_cols if col != "site_known"]

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
    "phys_",
)
lab_value_cols = [
    col for col in value_cols
    if col.startswith(lab_prefixes) and not any(token in col for token in intensity_tokens)
]
routine_lab_value_cols = [
    col for col in value_cols
    if col.startswith(routine_lab_prefixes) and not any(token in col for token in intensity_tokens)
]
vital_value_cols = [
    col for col in value_cols
    if col.startswith(vital_prefixes) and not any(token in col for token in intensity_tokens)
]
static_context_cols = [
    col for col in non_therapy_value_cols
    if not col.startswith(lab_prefixes) and not col.startswith(vital_prefixes)
]
static_context_no_site_cols = [col for col in static_context_cols if col != "site_known"]

feature_sets = {
    "v0.4F physiology trajectory no therapy": unique_cols(
        lab_value_cols + vital_value_cols + physiology_cols + static_context_cols
    ),
    "v0.4F physiology trajectory no therapy no site": unique_cols(
        lab_value_cols + vital_value_cols + physiology_cols + static_context_no_site_cols
    ),
    "v0.4F routine bedside no lactate no therapy": unique_cols(
        routine_lab_value_cols + vital_value_cols + physiology_cols + static_context_no_site_cols
    ),
    "v0.4F full no therapy": unique_cols(non_therapy_value_cols + physiology_cols),
    "v0.4F full no therapy no site": unique_cols(non_therapy_no_site_cols + physiology_cols),
    "v0.4F full cleaned therapy": unique_cols(value_cols + physiology_cols),
    "v0.4F full cleaned therapy no site": unique_cols(value_no_site_cols + physiology_cols),
    "v0.4F therapy context only": unique_cols(therapy_cols + [
        col for col in ["anchor_age", "site_known", "landmark_hour", "dwell_at_landmark_hours"]
        if col in numeric_cols
    ]),
}

feature_audit = pd.DataFrame([{
    "total_numeric_features": len(numeric_cols),
    "excluded_lab_vital_intensity_features": len(lab_vital_intensity_cols),
    "value_features": len(value_cols),
    "derived_physiology_features": len(physiology_cols),
    "therapy_features": len(therapy_cols),
    "lab_value_features": len(lab_value_cols),
    "routine_lab_value_features": len(routine_lab_value_cols),
    "vital_value_features": len(vital_value_cols),
    "static_context_features": len(static_context_cols),
}])
feature_audit.to_csv(
    os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_feature_audit.csv"),
    index=False,
)

print("")
print("Run 12 feature sets:")
for name, cols in feature_sets.items():
    print(f"  {name:<50} {len(cols)} features")


# %% Train, calibrate, evaluate

model_rows = []
threshold_rows = []
policy_rows = []
calibration_rows = []
top_risk_rows = []
best_curve_results = {frame: {} for frame in LABEL_FRAMES}

for label_frame in LABEL_FRAMES:
    print("")
    print("=" * 88)
    print(f"Label frame: {label_frame}")

    for horizon in HORIZONS:
        print("")
        print(f"Horizon: {horizon}h")
        df = make_label_frame(df_raw, horizon, label_frame)
        target_col = "run12_target"
        train_df, val_df, test_df = make_subject_level_splits(df, "subject_id", target_col)
        y_train = train_df[target_col].astype(int).to_numpy()
        y_val = val_df[target_col].astype(int).to_numpy()
        y_test = test_df[target_col].astype(int).to_numpy()

        print(
            f"  Rows: {len(df):,} | positives: {df[target_col].sum():,} "
            f"({df[target_col].mean() * 100:.2f}%)"
        )
        print(
            f"  Split positives train/val/test: "
            f"{y_train.sum():,}/{y_val.sum():,}/{y_test.sum():,}"
        )

        if y_train.sum() < 5 or y_val.sum() < 2 or y_test.sum() < 2:
            print("  Skipping horizon: too few positives.")
            continue

        for model_name, feature_cols in feature_sets.items():
            print(f"  Training {model_name}...")
            model = make_xgboost(y_train)
            model.fit(train_df[feature_cols], y_train)

            raw_val_prob = model.predict_proba(val_df[feature_cols])[:, 1]
            raw_test_prob = model.predict_proba(test_df[feature_cols])[:, 1]
            calibrators = fit_calibrators(y_val, raw_val_prob)

            calibrator_scores = []
            score_arrays = {}
            for score_version, calibrator in calibrators.items():
                val_prob = apply_calibrator(score_version, calibrator, raw_val_prob)
                test_prob = apply_calibrator(score_version, calibrator, raw_test_prob)
                score_arrays[score_version] = (val_prob, test_prob)
                calibrator_scores.append({
                    "score_version": score_version,
                    "val_brier_score": brier_score_loss(y_val, val_prob),
                    "val_pr_auc": average_precision_score(y_val, val_prob),
                    "val_roc_auc": safe_roc_auc(y_val, val_prob),
                })

            calibrator_scores = pd.DataFrame(calibrator_scores)
            selected_score_version = (
                calibrator_scores
                .sort_values(["val_brier_score", "val_pr_auc"], ascending=[True, False])
                .iloc[0]["score_version"]
            )

            for score_version, (val_prob, test_prob) in score_arrays.items():
                threshold, val_recall, val_precision = select_recall_threshold(y_val, val_prob)
                row = threshold_metrics(y_test, test_prob, threshold)
                row.update({
                    "label_frame": label_frame,
                    "horizon_hours": horizon,
                    "model": model_name,
                    "score_version": score_version,
                    "selected_by_val_brier": score_version == selected_score_version,
                    "n_features": len(feature_cols),
                    "val_recall_at_threshold": val_recall,
                    "val_precision_at_threshold": val_precision,
                    "roc_auc": safe_roc_auc(y_test, test_prob),
                    "pr_auc": average_precision_score(y_test, test_prob),
                    "brier_score": brier_score_loss(y_test, test_prob),
                    "base_rate": float(np.mean(y_test)),
                })
                model_rows.append(row)

                for cap in ALERT_CAPS_PER_100_ASSESSMENTS:
                    cap_threshold = select_alert_cap_threshold(y_val, val_prob, cap)
                    cap_row = threshold_metrics(y_test, test_prob, cap_threshold)
                    cap_row.update({
                        "threshold_strategy": f"val_alert_cap_{cap}_per_100_assessments",
                        "label_frame": label_frame,
                        "horizon_hours": horizon,
                        "model": model_name,
                        "score_version": score_version,
                    })
                    threshold_rows.append(cap_row)

                recall_row = row.copy()
                recall_row["threshold_strategy"] = f"val_min_recall_{MIN_RECALL_TARGET:.2f}"
                threshold_rows.append(recall_row)

                test_policy_df = test_df[[
                    "subject_id", "stay_id", "landmark_hour", target_col
                ]].copy()
                test_policy_df["score"] = test_prob

                for policy_func in [policy_first_alert_per_stay, policy_max_risk_one_per_stay]:
                    policy = policy_func(test_policy_df, target_col, "score", threshold)
                    policy.update({
                        "threshold_strategy": f"val_min_recall_{MIN_RECALL_TARGET:.2f}",
                        "threshold": threshold,
                        "label_frame": label_frame,
                        "horizon_hours": horizon,
                        "model": model_name,
                        "score_version": score_version,
                    })
                    policy_rows.append(policy)

                for cooldown in COOLDOWN_HOURS:
                    policy = policy_cooldown(test_policy_df, target_col, "score", threshold, cooldown)
                    policy.update({
                        "threshold_strategy": f"val_min_recall_{MIN_RECALL_TARGET:.2f}",
                        "threshold": threshold,
                        "label_frame": label_frame,
                        "horizon_hours": horizon,
                        "model": model_name,
                        "score_version": score_version,
                    })
                    policy_rows.append(policy)

                calibration_rows.append(
                    make_calibration_deciles(
                        y_test,
                        test_prob,
                        label_frame,
                        horizon,
                        model_name,
                        score_version,
                    )
                )

                for pct in TOP_RISK_PCTS:
                    for unit in ["assessment", "stay"]:
                        top_row = evaluate_top_risk(test_policy_df, target_col, "score", pct, unit)
                        top_row.update({
                            "label_frame": label_frame,
                            "horizon_hours": horizon,
                            "model": model_name,
                            "score_version": score_version,
                        })
                        top_risk_rows.append(top_row)

            model_file = os.path.join(
                MODEL_PATH,
                f"{clean_name(label_frame)}_h{horizon}_{clean_name(model_name)}.joblib",
            )
            joblib.dump({
                "model": model,
                "feature_cols": feature_cols,
                "calibrators": calibrators,
                "calibrator_scores": calibrator_scores,
                "selected_score_version": selected_score_version,
                "label_frame": label_frame,
                "horizon": horizon,
                "target_col": target_col,
            }, model_file)

            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv(
                os.path.join(
                    OUTPUT_PATH,
                    f"{clean_name(label_frame)}_h{horizon}_{clean_name(model_name)}_feature_importance.csv",
                ),
                index=False,
            )

            best_print = [
                row for row in model_rows
                if row["label_frame"] == label_frame
                and row["horizon_hours"] == horizon
                and row["model"] == model_name
                and row["score_version"] == selected_score_version
            ][0]
            print(
                f"    selected={selected_score_version} | "
                f"PR-AUC={best_print['pr_auc']:.4f} | "
                f"ROC-AUC={best_print['roc_auc']:.4f} | "
                f"Brier={best_print['brier_score']:.4f}"
            )


# %% Save tables

model_comparison = pd.DataFrame(model_rows).sort_values(
    ["label_frame", "horizon_hours", "pr_auc"],
    ascending=[True, True, False],
)
model_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_model_comparison.csv")
model_comparison.to_csv(model_file, index=False)

selected_model_comparison = model_comparison[model_comparison["selected_by_val_brier"]].copy()
best_by_frame_horizon = (
    selected_model_comparison
    .sort_values(["label_frame", "horizon_hours", "pr_auc"], ascending=[True, True, False])
    .groupby(["label_frame", "horizon_hours"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)
best_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_best_by_frame_horizon.csv")
best_by_frame_horizon.to_csv(best_file, index=False)

threshold_table = pd.DataFrame(threshold_rows)
threshold_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_threshold_table.csv")
threshold_table.to_csv(threshold_file, index=False)

policy_summary = pd.DataFrame(policy_rows)
policy_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_alert_policy_summary.csv")
policy_summary.to_csv(policy_file, index=False)

calibration_table = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
calibration_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_calibration_deciles.csv")
calibration_table.to_csv(calibration_file, index=False)

top_risk_table = pd.DataFrame(top_risk_rows)
top_risk_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_top_risk_review_table.csv")
top_risk_table.to_csv(top_risk_file, index=False)


# %% Plots

for label_frame in LABEL_FRAMES:
    frame_best = best_by_frame_horizon[best_by_frame_horizon["label_frame"].eq(label_frame)]
    if len(frame_best) == 0:
        continue

    plt.figure(figsize=(7, 5))
    plt.plot(frame_best["horizon_hours"], frame_best["pr_auc"], marker="o", label="PR-AUC")
    plt.plot(frame_best["horizon_hours"], frame_best["roc_auc"], marker="o", label="ROC-AUC")
    plt.xlabel("Prediction horizon (hours)")
    plt.ylabel("Metric")
    plt.title(f"v0.4F {label_frame} Best Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_PATH, f"v0_4f_{clean_name(label_frame)}_performance_by_horizon.png"),
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    best_policy = policy_summary.merge(
        frame_best[["label_frame", "horizon_hours", "model", "score_version"]],
        on=["label_frame", "horizon_hours", "model", "score_version"],
        how="inner",
    )
    for policy_name, group in best_policy.groupby("policy"):
        plt.plot(
            group["horizon_hours"],
            group["false_alerts_per_true_positive"].replace(np.inf, np.nan),
            marker="o",
            label=policy_name,
        )
    plt.xlabel("Prediction horizon (hours)")
    plt.ylabel("False alerts per true positive")
    plt.title(f"v0.4F {label_frame} Alert Policy Burden")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_PATH, f"v0_4f_{clean_name(label_frame)}_alert_policy_burden.png"),
        dpi=300,
    )
    plt.close()


# %% SHAP for overall best selected model

try:
    import shap

    overall_best = best_by_frame_horizon.sort_values("pr_auc", ascending=False).iloc[0]
    label_frame = overall_best["label_frame"]
    horizon = int(overall_best["horizon_hours"])
    model_name = overall_best["model"]
    score_version = overall_best["score_version"]
    print("")
    print(f"Generating SHAP for overall best selected model: {label_frame}, {horizon}h, {model_name}")

    df = make_label_frame(df_raw, horizon, label_frame)
    _, _, test_df = make_subject_level_splits(df, "subject_id", "run12_target")
    bundle = joblib.load(
        os.path.join(MODEL_PATH, f"{clean_name(label_frame)}_h{horizon}_{clean_name(model_name)}.joblib")
    )
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    shap_sample = test_df[feature_cols].sample(n=min(1000, len(test_df)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
    plt.tight_layout()
    shap_file = os.path.join(PLOT_PATH, "v0_4f_dynamic_alert_policy_best_model_shap.png")
    plt.savefig(shap_file, dpi=300, bbox_inches="tight")
    plt.close()
except ImportError:
    print("SHAP not installed. Skipping SHAP plot.")


# %% Manifest and console summary

manifest_rows = [
    ("Feature audit", os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_feature_audit.csv")),
    ("Model comparison", model_file),
    ("Best by label frame and horizon", best_file),
    ("Threshold table", threshold_file),
    ("Alert policy summary", policy_file),
    ("Calibration deciles", calibration_file),
    ("Top-risk review table", top_risk_file),
    ("Best model SHAP", os.path.join(PLOT_PATH, "v0_4f_dynamic_alert_policy_best_model_shap.png")),
]
manifest_file = os.path.join(OUTPUT_PATH, "v0_4f_dynamic_alert_policy_output_manifest.csv")
pd.DataFrame(manifest_rows, columns=["output", "path"]).to_csv(manifest_file, index=False)

display_cols = [
    "label_frame",
    "horizon_hours",
    "model",
    "score_version",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "base_rate",
    "threshold",
    "recall_sensitivity",
    "precision_ppv",
    "alerts_per_100_assessments",
    "false_alerts_per_true_positive",
]
print("")
print("v0.4F best calibrated model by label frame and horizon:")
print(best_by_frame_horizon[display_cols].round(4).to_string(index=False))

print("")
print("Saved outputs:")
for label, path in manifest_rows:
    print(f"  {label}: {path}")

print("")
print("Modeling 09 v0.4F Dynamic Alert Policy complete.")

