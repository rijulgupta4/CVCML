"""Run 29: outcome-validity and early-culture result-availability audit.

This run is intentionally development-only. It reproduces the frozen Run 23
primary target/model protocol on 2008-2019 data and never scores 2020-2022.
"""

from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT = Path(r"C:\path\to\CVCML")
MIMIC = Path(r"C:\path\to\mimic-iv")
DATA = PROJECT / "data" / "v0_5"
OUTPUT = PROJECT / "Outputs" / "Run 29 (v0.5 Outcome Validity and Leakage Audit)"
PLOTS = OUTPUT / "plots"
MODELS = OUTPUT / "models"

FEATURE_FILE = DATA / "v0_5_run20_dynamic_enriched_features.csv"
LABEL_FILE = DATA / "v0_5_run22_source_screened_daily_landmarks.csv"
EPISODE_FILE = DATA / "v0_5_catheter_exposure_periods.csv"
MICRO_FILE = MIMIC / "hosp" / "microbiologyevents.csv.gz"
REVIEW_SOURCE = (
    PROJECT / "Outputs" / "Run 27.1 (v0.5 ICD Discordance Supplement)"
    / "v0_5_run27_1_episode_discordance_phenotypes.csv"
)

TARGET = "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013"]
CALIB_GROUPS = ["2014 - 2016"]
VALID_GROUPS = ["2017 - 2019"]
LOCKBOX_GROUP = "2020 - 2022"
RANDOM_STATE = 42

STATIC_NUMERIC = ["landmark_hour", "landmark_day", "anchor_age"]
STATIC_CATEGORICAL = ["gender", "admission_type", "insurance", "race", "first_careunit"]
LAB_PREFIXES = ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"]
BLOOD_SPEC_TYPES = ["BLOOD CULTURE", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"]


def ensure_paths():
    for path in [OUTPUT, PLOTS, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def clip_prob(prob):
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)


def logit(prob):
    prob = clip_prob(prob)
    return np.log(prob / (1 - prob))


def sigmoid(score):
    score = np.clip(np.asarray(score, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-score))


def apply_platt(model, raw_prob):
    return sigmoid(logit(raw_prob) * float(model.coef_[0][0]) + float(model.intercept_[0]))


def calibration_intercept_slope(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    x = np.clip(logit(y_prob), -20, 20)
    fitted = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
    fitted.fit(x.reshape(-1, 1), y_true)
    return float(fitted.intercept_[0]), float(fitted.coef_[0][0])


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    reference = float(brier_score_loss(y_true, np.repeat(prevalence, len(y_true))))
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": pr_auc / prevalence if prevalence else np.nan,
        "brier_score": brier,
        "brier_reference_prevalence": reference,
        "brier_skill_score": 1 - brier / reference if reference else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_observed_ratio": float(y_prob.sum() / y_true.sum()) if y_true.sum() else np.nan,
    }


def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
            ]),
            categorical_cols,
        ),
    ])


def fit_model(train, numeric_cols, categorical_cols):
    positives = int(train[TARGET].sum())
    negatives = int(len(train) - positives)
    model = Pipeline([
        ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
        (
            "model",
            XGBClassifier(
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
                scale_pos_weight=negatives / positives if positives else 1.0,
                random_state=RANDOM_STATE,
                n_jobs=2,
                tree_method="hist",
            ),
        ),
    ])
    cols = numeric_cols + categorical_cols
    model.fit(train[cols], train[TARGET].astype(int))
    return model


def review_policy(scored, probability_col, model_variant):
    episode_best = (
        scored.sort_values(["episode_id", probability_col, "landmark_hour"], ascending=[True, False, True])
        .groupby("episode_id")
        .head(1)
        .copy()
    )
    episode_best["episode_positive"] = episode_best["episode_id"].map(
        scored.groupby("episode_id")[TARGET].max()
    ).fillna(0).astype(int)
    episode_best = episode_best.sort_values(probability_col, ascending=False).reset_index(drop=True)
    total_positive = int(episode_best["episode_positive"].sum())
    rows = []
    for pct in [1, 2, 5, 10]:
        n = max(1, int(np.ceil(len(episode_best) * pct / 100)))
        flagged = episode_best.head(n)
        tp = int(flagged["episode_positive"].sum())
        rows.append({
            "model_variant": model_variant,
            "policy": f"top_{pct}_percent_episodes_by_max_risk",
            "episodes_reviewed": int(len(flagged)),
            "reviewed_episode_fraction": len(flagged) / len(episode_best),
            "true_positive_episodes": tp,
            "false_positive_episodes": int(len(flagged) - tp),
            "episode_precision_ppv": tp / len(flagged) if len(flagged) else np.nan,
            "episode_recall_sensitivity": tp / total_positive if total_positive else np.nan,
            "false_episode_reviews_per_true_positive": (len(flagged) - tp) / tp if tp else np.nan,
        })
    return pd.DataFrame(rows)


def load_development_frame():
    print("Loading enriched features and source-screened target...", flush=True)
    features = pd.read_csv(FEATURE_FILE, parse_dates=["landmark_time", "exposure_start"])
    labels = pd.read_csv(LABEL_FILE, usecols=["landmark_id", TARGET])
    features = features.merge(labels, on="landmark_id", how="left", validate="one_to_one")
    features[TARGET] = features[TARGET].fillna(0).astype(int)
    lockbox_rows = int(features["anchor_year_group"].eq(LOCKBOX_GROUP).sum())
    development = features[
        features["run18_primary_model_frame"].eq(1)
        & features["split_role"].eq("development")
        & ~features["anchor_year_group"].eq(LOCKBOX_GROUP)
    ].copy()
    del features
    print(f"  Development rows retained: {len(development):,}", flush=True)
    print(f"  Lockbox rows excluded without scoring: {lockbox_rows:,}", flush=True)
    return development, lockbox_rows


def build_microbiology_availability(development):
    print("Loading positive blood-culture timestamps...", flush=True)
    usecols = [
        "microevent_id", "micro_specimen_id", "subject_id", "hadm_id", "charttime", "storetime",
        "spec_type_desc", "test_name", "org_name",
    ]
    micro = pd.read_csv(MICRO_FILE, usecols=usecols, low_memory=False)
    micro = micro[
        micro["spec_type_desc"].isin(BLOOD_SPEC_TYPES)
        & micro["org_name"].notna()
        & ~micro["org_name"].str.contains("CANCELLED", case=False, na=False)
    ].copy()
    micro["charttime"] = pd.to_datetime(micro["charttime"], errors="coerce")
    micro["storetime"] = pd.to_datetime(micro["storetime"], errors="coerce")
    micro = micro[micro["charttime"].notna()].copy()

    # Susceptibility rows repeat an organism. Collapse to specimen-organism events.
    culture_events = (
        micro.groupby(
            ["subject_id", "hadm_id", "micro_specimen_id", "charttime", "spec_type_desc", "org_name"],
            dropna=False,
        )
        .agg(
            earliest_positive_storetime=("storetime", "min"),
            latest_positive_storetime=("storetime", "max"),
            raw_micro_rows=("microevent_id", "size"),
        )
        .reset_index()
    )
    del micro

    episode_ids = development["episode_id"].drop_duplicates()
    episodes = pd.read_csv(
        EPISODE_FILE,
        usecols=[
            "episode_id", "subject_id", "hadm_id", "stay_id", "exposure_start",
            "exposure_end_observed", "early_positive_culture",
        ],
        parse_dates=["exposure_start", "exposure_end_observed"],
    )
    episodes = episodes[episodes["episode_id"].isin(episode_ids)].copy()
    episodes["early_window_end"] = episodes["exposure_start"] + pd.Timedelta(hours=48)
    matched = episodes.merge(culture_events, on=["subject_id", "hadm_id"], how="left")
    matched = matched[
        matched["charttime"].notna()
        & matched["charttime"].ge(matched["exposure_start"])
        & matched["charttime"].lt(matched["early_window_end"])
        & matched["charttime"].le(matched["exposure_end_observed"])
    ].copy()
    matched["earliest_turnaround_hours"] = (
        (matched["earliest_positive_storetime"] - matched["charttime"]).dt.total_seconds() / 3600
    )
    matched["latest_turnaround_hours"] = (
        (matched["latest_positive_storetime"] - matched["charttime"]).dt.total_seconds() / 3600
    )

    episode_audit = (
        matched.groupby("episode_id")
        .agg(
            early_positive_specimen_time=("charttime", "min"),
            earliest_positive_storetime=("earliest_positive_storetime", "min"),
            latest_positive_storetime=("latest_positive_storetime", "max"),
            early_positive_specimen_count=("micro_specimen_id", "nunique"),
            early_positive_organism_count=("org_name", "nunique"),
            earliest_turnaround_hours=("earliest_turnaround_hours", "min"),
            latest_turnaround_hours=("latest_turnaround_hours", "max"),
        )
        .reset_index()
    )
    episode_audit = episodes[
        ["episode_id", "subject_id", "hadm_id", "stay_id", "exposure_start", "early_positive_culture"]
    ].merge(episode_audit, on="episode_id", how="left")
    episode_audit["reconstructed_early_positive_specimen"] = episode_audit[
        "early_positive_specimen_time"
    ].notna().astype(int)
    episode_audit["original_reconstruction_agrees"] = (
        episode_audit["early_positive_culture"].fillna(0).astype(int)
        == episode_audit["reconstructed_early_positive_specimen"]
    ).astype(int)

    development = development.merge(
        episode_audit[
            [
                "episode_id", "early_positive_specimen_time", "earliest_positive_storetime",
                "latest_positive_storetime", "early_positive_specimen_count",
                "early_positive_organism_count", "reconstructed_early_positive_specimen",
            ]
        ],
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    development["early_positive_result_available_at_landmark"] = (
        development["earliest_positive_storetime"].notna()
        & development["earliest_positive_storetime"].le(development["landmark_time"])
    ).astype(int)
    development["early_positive_result_unavailable_at_landmark"] = (
        development["reconstructed_early_positive_specimen"].fillna(0).eq(1)
        & development["early_positive_result_available_at_landmark"].eq(0)
    ).astype(int)
    development["hours_from_specimen_to_landmark"] = (
        (development["landmark_time"] - development["early_positive_specimen_time"]).dt.total_seconds() / 3600
    )
    development["hours_from_storetime_to_landmark"] = (
        (development["landmark_time"] - development["earliest_positive_storetime"]).dt.total_seconds() / 3600
    )

    detail_cols = [
        "landmark_id", "episode_id", "subject_id", "anchor_year_group", "landmark_hour", "landmark_time",
        TARGET, "early_positive_culture", "reconstructed_early_positive_specimen",
        "early_positive_specimen_time", "earliest_positive_storetime", "latest_positive_storetime",
        "early_positive_result_available_at_landmark", "early_positive_result_unavailable_at_landmark",
        "hours_from_specimen_to_landmark", "hours_from_storetime_to_landmark",
    ]
    development[detail_cols].to_csv(OUTPUT / "v0_5_run29_early_culture_landmark_audit.csv", index=False)
    episode_audit.to_csv(OUTPUT / "v0_5_run29_early_culture_episode_audit.csv", index=False)

    by_landmark = (
        development.groupby("landmark_hour")
        .agg(
            rows=("landmark_id", "size"),
            target_positive_rows=(TARGET, "sum"),
            original_early_positive_rows=("early_positive_culture", "sum"),
            reconstructed_early_positive_rows=("reconstructed_early_positive_specimen", "sum"),
            result_available_rows=("early_positive_result_available_at_landmark", "sum"),
            result_unavailable_rows=("early_positive_result_unavailable_at_landmark", "sum"),
        )
        .reset_index()
    )
    by_landmark["unavailable_fraction_among_early_positive"] = (
        by_landmark["result_unavailable_rows"] / by_landmark["reconstructed_early_positive_rows"].replace(0, np.nan)
    )
    by_landmark.to_csv(OUTPUT / "v0_5_run29_early_culture_by_landmark.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "development_rows", "value": len(development)},
        {"metric": "development_episodes", "value": development["episode_id"].nunique()},
        {"metric": "original_early_positive_episodes", "value": int(episode_audit["early_positive_culture"].sum())},
        {"metric": "reconstructed_early_positive_episodes", "value": int(episode_audit["reconstructed_early_positive_specimen"].sum())},
        {"metric": "episode_reconstruction_agreement", "value": float(episode_audit["original_reconstruction_agrees"].mean())},
        {"metric": "early_positive_landmark_rows", "value": int(development["reconstructed_early_positive_specimen"].sum())},
        {"metric": "result_unavailable_early_positive_landmark_rows", "value": int(development["early_positive_result_unavailable_at_landmark"].sum())},
        {
            "metric": "result_unavailable_fraction_among_early_positive_landmark_rows",
            "value": float(
                development.loc[development["reconstructed_early_positive_specimen"].eq(1), "early_positive_result_unavailable_at_landmark"].mean()
            ),
        },
        {"metric": "median_earliest_positive_turnaround_hours", "value": float(episode_audit["earliest_turnaround_hours"].median())},
        {"metric": "median_latest_positive_turnaround_hours", "value": float(episode_audit["latest_turnaround_hours"].median())},
    ])
    summary.to_csv(OUTPUT / "v0_5_run29_early_culture_summary.csv", index=False)
    return development, episode_audit, by_landmark, summary


def model_ablation(development):
    print("Running frozen development-only model ablation...", flush=True)
    lab_cols = sorted({c for c in development.columns if any(c.startswith(prefix) for prefix in LAB_PREFIXES)})
    vital_cols = sorted({c for c in development.columns if c.startswith("vital_")})
    therapy_cols = sorted({c for c in development.columns if c.startswith("abx_") or c.startswith("vaso_")})
    dynamic_numeric = lab_cols + vital_cols + therapy_cols
    variants = {
        "original_episode_level_early_positive": STATIC_NUMERIC + ["early_positive_culture"] + dynamic_numeric,
        "safe_exclude_early_positive": STATIC_NUMERIC + dynamic_numeric,
        "exploratory_storetime_available_only": STATIC_NUMERIC + ["early_positive_result_available_at_landmark"] + dynamic_numeric,
    }
    train = development[development["anchor_year_group"].isin(TRAIN_GROUPS)].copy()
    calib = development[development["anchor_year_group"].isin(CALIB_GROUPS)].copy()
    valid = development[development["anchor_year_group"].isin(VALID_GROUPS)].copy()
    metric_rows = []
    policy_frames = []
    prediction_frames = []

    for name, numeric_cols in variants.items():
        print(f"  Fitting {name}...", flush=True)
        feature_cols = numeric_cols + STATIC_CATEGORICAL
        model = fit_model(train, numeric_cols, STATIC_CATEGORICAL)
        calib_raw = model.predict_proba(calib[feature_cols])[:, 1]
        valid_raw = model.predict_proba(valid[feature_cols])[:, 1]
        platt = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
        platt.fit(logit(calib_raw).reshape(-1, 1), calib[TARGET].astype(int))
        valid_platt = apply_platt(platt, valid_raw)
        joblib.dump(model, MODELS / f"run29_xgboost_{name}.joblib")
        joblib.dump(platt, MODELS / f"run29_platt_{name}.joblib")

        for split, frame, raw_prob in [("calibration", calib, calib_raw), ("validation", valid, valid_raw)]:
            for calibration, prob in [("raw", raw_prob), ("platt", apply_platt(platt, raw_prob))]:
                row = {"model_variant": name, "split": split, "calibration": calibration}
                row.update(metrics(frame[TARGET], prob))
                metric_rows.append(row)

        scored = valid[
            ["landmark_id", "episode_id", "subject_id", "anchor_year_group", "landmark_hour", "landmark_time", TARGET]
        ].copy()
        scored["model_variant"] = name
        scored["raw_probability"] = valid_raw
        scored["platt_probability"] = valid_platt
        prediction_frames.append(scored)
        policy_frames.append(review_policy(scored, "platt_probability", name))

    comparison = pd.DataFrame(metric_rows)
    policies = pd.concat(policy_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison.to_csv(OUTPUT / "v0_5_run29_ablation_model_comparison.csv", index=False)
    policies.to_csv(OUTPUT / "v0_5_run29_ablation_review_policy.csv", index=False)
    predictions.to_csv(OUTPUT / "v0_5_run29_validation_predictions.csv", index=False)
    return comparison, policies, predictions


def make_review_package():
    source = pd.read_csv(REVIEW_SOURCE)
    sampled = []
    for group_name, group in source.groupby("proxy_icd_group"):
        # Guarantee representation of rare source-screen strata, then fill to
        # an equal 25-case agreement-cell budget with a deterministic sample.
        selected_parts = []
        for _, stratum in group.groupby("source_screen_class", dropna=False):
            selected_parts.append(stratum.sample(n=min(5, len(stratum)), random_state=2029))
        selected = pd.concat(selected_parts).drop_duplicates("episode_id")
        remaining_n = max(0, 25 - len(selected))
        remaining = group[~group["episode_id"].isin(selected["episode_id"])]
        if remaining_n:
            selected = pd.concat([
                selected,
                remaining.sample(n=min(remaining_n, len(remaining)), random_state=2029),
            ])
        sampled.append(selected.head(25))
    review = pd.concat(sampled, ignore_index=True)
    review = review.sort_values(["proxy_icd_group", "source_screen_class", "episode_id"]).reset_index(drop=True)
    review["run29_review_id"] = [f"R29-{i:03d}" for i in range(1, len(review) + 1)]
    review["review_status"] = "not_started"
    review["adjudicator_id"] = ""
    review["eligible_cvc_exposure_confirmed"] = ""
    review["line_present_on_culture_day_or_prior_day"] = ""
    review["blood_culture_criterion_met"] = ""
    review["common_commensal_symptom_criterion_met"] = ""
    review["secondary_source_evidence"] = ""
    review["secondary_source_site"] = ""
    review["mbi_lcbi_possible"] = ""
    review["line_infection_documented_in_note"] = ""
    review["final_adjudication"] = ""
    review["confidence"] = ""
    review["review_comments"] = ""
    review.to_csv(OUTPUT / "v0_5_run29_adjudication_review_sample.csv", index=False)

    template_cols = [
        "run29_review_id", "episode_id", "adjudicator_id", "review_status",
        "eligible_cvc_exposure_confirmed", "line_present_on_culture_day_or_prior_day",
        "blood_culture_criterion_met", "common_commensal_symptom_criterion_met",
        "secondary_source_evidence", "secondary_source_site", "mbi_lcbi_possible",
        "line_infection_documented_in_note", "final_adjudication", "confidence", "review_comments",
    ]
    review[template_cols].to_csv(OUTPUT / "v0_5_run29_adjudication_template.csv", index=False)

    protocol = """# Run 29 adjudication protocol

## Purpose
Validate the strict CVC-associated BSI proxy against structured chart evidence and, when MIMIC-IV-Note becomes available, narrative evidence. This is not official NHSN adjudication.

## Blinding
Reviewers should not see model scores while adjudicating the outcome. The queue is stratified by proxy/ICD agreement and source-screen class; it is not a prevalence sample.

## Review sequence
1. Confirm the reconstructed CVC exposure episode and at least 48 hours of observed line exposure before the candidate blood culture.
2. Confirm that the line was present on the event day or the prior calendar day where the available data permit this.
3. Confirm the blood-culture organism rule: a recognized pathogen, or repeated qualifying common commensal cultures with compatible symptoms.
4. Search the +/-3-day window for a plausible secondary source using nonblood cultures, diagnoses, procedures, antimicrobial context, and notes when available.
5. Consider MBI-LCBI plausibility using malignancy/transplant, neutropenia, and organism context; mark uncertain when the structured record is insufficient.
6. Assign one final category: likely primary CVC-associated BSI, likely secondary BSI, contaminant, or insufficient evidence.
7. Record confidence as high, medium, or low and document the decisive evidence.

## Interpretation
The balanced queue estimates failure modes and agreement within strata; it must not be used directly to estimate population PPV without applying sampling weights. Two independent reviewers and consensus resolution are preferred for a manuscript-grade validation subset.

## Required future data
MIMIC-IV-Note is not present locally. Narrative source attribution and chills documentation therefore remain pending until note access is added.

## Sources
- MIMIC-IV microbiologyevents documentation: https://mimic.mit.edu/docs/IV/modules/hosp/microbiologyevents.html
- CDC NHSN Patient Safety Component Manual, Bloodstream Infection chapter: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf
"""
    (OUTPUT / "v0_5_run29_adjudication_protocol.md").write_text(protocol, encoding="utf-8")
    return review


def make_plots(by_landmark, comparison, predictions):
    def font(size, bold=False):
        path = r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    title_font = font(38, True)
    axis_font = font(26)
    tick_font = font(21)
    colors = [(217, 95, 2), (27, 158, 119), (117, 112, 179)]

    def canvas(title):
        image = Image.new("RGB", (1700, 1050), "white")
        draw = ImageDraw.Draw(image)
        draw.text((850, 35), title, font=title_font, fill="black", anchor="ma")
        return image, draw, (180, 130, 1580, 860)

    def axes(draw, box, x_label, y_label, y_max=1.0):
        left, top, right, bottom = box
        draw.line((left, bottom, right, bottom), fill="black", width=3)
        draw.line((left, top, left, bottom), fill="black", width=3)
        for frac in np.linspace(0, 1, 6):
            y = bottom - frac * (bottom - top)
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((left - 15, y), f"{frac * y_max:.2f}", font=tick_font, fill="black", anchor="rm")
        draw.text(((left + right) / 2, 990), x_label, font=axis_font, fill="black", anchor="ma")
        draw.text((left, 95), y_label, font=axis_font, fill="black", anchor="la")

    # Availability line chart.
    image, draw, box = canvas("Run 29 Early-Culture Result Availability by Landmark")
    axes(draw, box, "Landmark hour", "Fraction with result unavailable", 1.0)
    left, top, right, bottom = box
    early_landmarks = by_landmark[by_landmark["landmark_hour"].le(240)].copy()
    x_vals = early_landmarks["landmark_hour"].astype(float).to_numpy()
    y_vals = early_landmarks["unavailable_fraction_among_early_positive"].fillna(0).astype(float).to_numpy()
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    points = []
    for x, y in zip(x_vals, y_vals):
        px = left + (x - x_min) / max(x_max - x_min, 1) * (right - left)
        py = bottom - y * (bottom - top)
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=5)
    for px, py in points:
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(31, 119, 180))
    for x in np.linspace(x_min, x_max, 6):
        px = left + (x - x_min) / max(x_max - x_min, 1) * (right - left)
        draw.text((px, bottom + 15), f"{x:.0f}", font=tick_font, fill="black", anchor="ma")
    image.save(PLOTS / "v0_5_run29_result_unavailability_by_landmark.png")

    # PR-AUC lift bars.
    val = comparison[(comparison["split"].eq("validation")) & (comparison["calibration"].eq("platt"))].copy()
    y_max = max(2.0, float(np.ceil(val["pr_auc_lift_over_prevalence"].max() * 4) / 4))
    image, draw, box = canvas("Run 29 Development-Validation Ablation")
    axes(draw, box, "Feature variant", "PR-AUC lift over prevalence", y_max)
    left, top, right, bottom = box
    slot = (right - left) / len(val)
    for idx, row in enumerate(val.itertuples()):
        cx = left + slot * (idx + 0.5)
        height = row.pr_auc_lift_over_prevalence / y_max * (bottom - top)
        draw.rectangle((cx - slot * 0.28, bottom - height, cx + slot * 0.28, bottom), fill=colors[idx])
        label = row.model_variant.replace("original_episode_level_", "original\n").replace(
            "safe_exclude_", "safe exclude\n"
        ).replace("exploratory_storetime_", "storetime\n").replace("_", " ")
        draw.multiline_text((cx, bottom + 15), label, font=tick_font, fill="black", anchor="ma", align="center")
        draw.text((cx, bottom - height - 12), f"{row.pr_auc_lift_over_prevalence:.2f}x", font=tick_font, fill="black", anchor="ma")
    y_one = bottom - 1 / y_max * (bottom - top)
    draw.line((left, y_one, right, y_one), fill=(80, 80, 80), width=2)
    image.save(PLOTS / "v0_5_run29_ablation_pr_auc_lift.png")

    # Precision-recall curves.
    image, draw, box = canvas("Run 29 Early-Culture Feature Ablation PR Curves")
    axes(draw, box, "Recall / sensitivity", "Precision / PPV", 1.0)
    left, top, right, bottom = box
    legend_y = 155
    for idx, (name, group) in enumerate(predictions.groupby("model_variant")):
        precision, recall, _ = precision_recall_curve(group[TARGET], group["platt_probability"])
        ap = average_precision_score(group[TARGET], group["platt_probability"])
        pts = [(left + r * (right - left), bottom - p * (bottom - top)) for r, p in zip(recall, precision)]
        if len(pts) > 1:
            draw.line(pts, fill=colors[idx], width=4)
        draw.line((1020, legend_y, 1085, legend_y), fill=colors[idx], width=5)
        draw.text((1100, legend_y), f"{name} (AP={ap:.3f})", font=tick_font, fill="black", anchor="lm")
        legend_y += 38
    prevalence = float(predictions[TARGET].mean())
    base_y = bottom - prevalence * (bottom - top)
    for x in range(left, right, 22):
        draw.line((x, base_y, min(x + 11, right), base_y), fill=(100, 100, 100), width=2)
    draw.text((1100, legend_y), f"Base rate={prevalence:.3f}", font=tick_font, fill="black", anchor="lm")
    for frac in np.linspace(0, 1, 6):
        x = left + frac * (right - left)
        draw.text((x, bottom + 15), f"{frac:.1f}", font=tick_font, fill="black", anchor="ma")
    image.save(PLOTS / "v0_5_run29_ablation_pr_curves.png")


def write_notes(lockbox_rows, summary, comparison, policies, review):
    val = comparison[(comparison["split"].eq("validation")) & (comparison["calibration"].eq("platt"))].copy()
    safe = val[val["model_variant"].eq("safe_exclude_early_positive")].iloc[0]
    original = val[val["model_variant"].eq("original_episode_level_early_positive")].iloc[0]
    unavailable = summary.loc[
        summary["metric"].eq("result_unavailable_fraction_among_early_positive_landmark_rows"), "value"
    ].iloc[0]
    note = f"""# Run 29: Outcome validity and leakage audit

## Decision question
Does the frozen development model use knowledge that an early blood specimen eventually grew an organism before that result was available, and is performance robust to removing that feature?

## Locked design
- Primary target: `{TARGET}`.
- Train: 2008-2013; Platt calibration: 2014-2016; validation: 2017-2019.
- Model and hyperparameters copied from Run 23 without tuning.
- The 2020-2022 temporal lockbox was excluded before audit/modeling; {lockbox_rows:,} lockbox rows were not scored.

## Main findings
- {unavailable:.1%} of development landmark rows carrying an early-positive specimen flag did not yet have an organism-positive `storetime` by the landmark.
- Original-feature validation PR-AUC: {original['pr_auc']:.4f} ({original['pr_auc_lift_over_prevalence']:.2f}x prevalence).
- Safe exclusion validation PR-AUC: {safe['pr_auc']:.4f} ({safe['pr_auc_lift_over_prevalence']:.2f}x prevalence).
- PR-AUC change after exclusion: {safe['pr_auc'] - original['pr_auc']:+.4f}.
- The adjudication queue contains {len(review):,} stratified episodes and remains pending manual review.

## Interpretation
`early_positive_culture` is an outcome/result-derived episode flag, not a prospectively available specimen-order feature. `charttime` is specimen collection time, whereas `storetime` is the last known microbiology result update. Because the feature was copied onto all daily landmarks, it can reveal eventual culture positivity before the organism result was available. The manuscript-safe primary pipeline should exclude it. The storetime-aware replacement is exploratory because MIMIC documents `storetime` as the last known update rather than the first preliminary notification.

## Scope and limitations
- Run 29 is a development-only validity audit, not a new lockbox evaluation.
- `storetime` is an imperfect proxy for clinical notification time.
- MIMIC-IV-Note is not available locally, so narrative source adjudication is not yet performed.
- The balanced review sample supports failure-mode analysis, not unweighted population PPV estimation.

## Sources
- MIMIC-IV microbiologyevents documentation: https://mimic.mit.edu/docs/IV/modules/hosp/microbiologyevents.html
- CDC NHSN Bloodstream Infection Event guidance: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf
"""
    (OUTPUT / "v0_5_run29_notes.md").write_text(note, encoding="utf-8")

    manifest = pd.DataFrame([
        {"artifact": p.name, "path": str(p), "bytes": p.stat().st_size}
        for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "v0_5_run29_manifest.csv"
    ])
    manifest.to_csv(OUTPUT / "v0_5_run29_manifest.csv", index=False)

    metadata = {
        "run": 29,
        "name": "v0.5 Outcome Validity and Leakage Audit",
        "created_by": "Run 29 script",
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "lockbox_scored": False,
        "lockbox_group": LOCKBOX_GROUP,
        "review_sample_rows": int(len(review)),
    }
    (OUTPUT / "v0_5_run29_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    ensure_paths()
    development, lockbox_rows = load_development_frame()
    development, episode_audit, by_landmark, summary = build_microbiology_availability(development)
    comparison, policies, predictions = model_ablation(development)
    review = make_review_package()
    make_plots(by_landmark, comparison, predictions)
    write_notes(lockbox_rows, summary, comparison, policies, review)
    shutil.copy2(Path(__file__), OUTPUT / Path(__file__).name)
    print("", flush=True)
    print("Run 29 complete.", flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

