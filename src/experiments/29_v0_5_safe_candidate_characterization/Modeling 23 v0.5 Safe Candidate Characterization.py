"""Run 30: characterize the leakage-safe v0.5 development-validation candidate.

This run does not refit, retune, or score the 2020-2022 lockbox. It uses the
frozen Run 29 safe validation predictions and quantifies evaluation-sample
uncertainty with patient-clustered bootstrap resampling.
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(r"C:\path\to\CVCML")
DATA = PROJECT / "data" / "v0_5"
RUN29 = PROJECT / "Outputs" / "Run 29 (v0.5 Outcome Validity and Leakage Audit)"
OUTPUT = PROJECT / "Outputs" / "Run 30 (v0.5 Safe Candidate Characterization)"
PLOTS = OUTPUT / "plots"

FEATURE_FILE = DATA / "v0_5_run20_dynamic_enriched_features.csv"
PREDICTION_FILE = RUN29 / "v0_5_run29_validation_predictions.csv"
TARGET = "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"
MODEL_VARIANT = "safe_exclude_early_positive"
PROBABILITY = "platt_probability"
RANDOM_STATE = 2030
N_BOOTSTRAP = 2000
CI_LOWER = 0.025
CI_UPPER = 0.975

CONTEXT_COLUMNS = [
    "landmark_id",
    "gender",
    "race",
    "anchor_age",
    "first_careunit",
    "admission_type",
    "insurance",
    "cvc_types",
]


def ensure_paths():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def clip_prob(probability):
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)


def logit(probability):
    probability = clip_prob(probability)
    return np.log(probability / (1 - probability))


def calibration_intercept_slope(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    predictor = np.clip(logit(y_prob), -20, 20).reshape(-1, 1)
    model = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
    try:
        model.fit(predictor, y_true)
    except ValueError:
        return np.nan, np.nan
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metric_values(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else np.nan
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "prevalence": prevalence,
            "pr_auc_lift": np.nan,
            "brier_score": np.nan,
            "brier_skill_score": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "expected_observed_ratio": np.nan,
        }
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    reference = float(brier_score_loss(y_true, np.repeat(prevalence, len(y_true))))
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": safe_divide(pr_auc, prevalence),
        "brier_score": brier,
        "brier_skill_score": 1 - safe_divide(brier, reference),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_observed_ratio": safe_divide(float(y_prob.sum()), int(y_true.sum())),
    }


def cluster_index_map(frame):
    groups = frame.groupby("subject_id", sort=False).indices
    cluster_ids = np.asarray(list(groups.keys()))
    cluster_rows = [np.asarray(groups[cluster_id], dtype=int) for cluster_id in cluster_ids]
    return cluster_ids, cluster_rows


def bootstrap_metric_distribution(frame, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    local = frame.reset_index(drop=True)
    y = local[TARGET].to_numpy(dtype=int)
    probability = local[PROBABILITY].to_numpy(dtype=float)
    cluster_ids, cluster_rows = cluster_index_map(local)
    rng = np.random.default_rng(seed)
    metric_names = list(metric_values(y, probability).keys())
    values = {metric: np.full(n_bootstrap, np.nan) for metric in metric_names}

    for iteration in range(n_bootstrap):
        sampled_positions = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        sampled_rows = np.concatenate([cluster_rows[position] for position in sampled_positions])
        result = metric_values(y[sampled_rows], probability[sampled_rows])
        for metric in metric_names:
            values[metric][iteration] = result[metric]
    return values


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    lower, upper = np.quantile(values, [CI_LOWER, CI_UPPER])
    return float(lower), float(upper), int(len(values))


def summarize_metrics(frame, analysis, subgroup_variable="overall", subgroup_level="all", seed=RANDOM_STATE):
    point = metric_values(frame[TARGET], frame[PROBABILITY])
    bootstrap = bootstrap_metric_distribution(frame, seed=seed)
    positive_subjects = frame.loc[frame[TARGET].eq(1), "subject_id"].nunique()
    positive_episodes = frame.loc[frame[TARGET].eq(1), "episode_id"].nunique()
    rows = []
    for metric, estimate in point.items():
        lower, upper, valid_replicates = interval(bootstrap[metric])
        rows.append({
            "analysis": analysis,
            "subgroup_variable": subgroup_variable,
            "subgroup_level": subgroup_level,
            "metric": metric,
            "estimate": estimate,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "valid_bootstrap_replicates": valid_replicates,
            "bootstrap_replicates_requested": N_BOOTSTRAP,
            "landmark_rows": int(len(frame)),
            "positive_landmark_rows": int(frame[TARGET].sum()),
            "subjects": int(frame["subject_id"].nunique()),
            "positive_subjects": int(positive_subjects),
            "episodes": int(frame["episode_id"].nunique()),
            "positive_episodes": int(positive_episodes),
        })
    return pd.DataFrame(rows)


def collapse_race(value):
    value = str(value).upper()
    if "HISPANIC" in value or "LATINO" in value:
        return "Hispanic/Latino"
    if "BLACK" in value:
        return "Black"
    if "ASIAN" in value:
        return "Asian"
    if "WHITE" in value or "PORTUGUESE" in value:
        return "White"
    return "Other/unknown"


def collapse_careunit(value):
    value = str(value).upper()
    if "NEURO" in value:
        return "Neurologic"
    if "CARDIAC" in value or "CORONARY" in value or "CVICU" in value or "CCU" in value:
        return "Cardiac"
    if "SURG" in value or "TRAUMA" in value or "TSICU" in value:
        return "Surgical/trauma"
    if "MEDICAL" in value or "MICU" in value:
        return "Medical"
    return "Other"


def collapse_admission(value):
    value = str(value).upper()
    if "EMER" in value:
        return "Emergency"
    if "URGENT" in value:
        return "Urgent"
    if "OBSERVATION" in value:
        return "Observation"
    if "ELECTIVE" in value or "SAME DAY" in value:
        return "Elective/same-day"
    return "Other/unknown"


def collapse_catheter(value):
    value = str(value).upper()
    if "DIALYSIS" in value:
        return "Dialysis-involved"
    if "PA CATHETER" in value:
        return "PA-involved, no dialysis"
    if "PICC" in value:
        return "PICC, no dialysis/PA"
    return "Other"


def add_subgroups(frame):
    result = frame.copy()
    result["sex_group"] = result["gender"].fillna("Unknown")
    result["age_group"] = pd.cut(
        result["anchor_age"],
        bins=[-np.inf, 49, 64, 79, np.inf],
        labels=["<50", "50-64", "65-79", ">=80"],
    ).astype("object").fillna("Unknown")
    result["race_group"] = result["race"].fillna("Unknown").map(collapse_race)
    result["careunit_group"] = result["first_careunit"].fillna("Unknown").map(collapse_careunit)
    result["admission_group"] = result["admission_type"].fillna("Unknown").map(collapse_admission)
    result["insurance_group"] = result["insurance"].fillna("Unknown")
    result["catheter_group"] = result["cvc_types"].fillna("Unknown").map(collapse_catheter)
    return result


def reliability_flag(positive_subjects, subjects):
    if positive_subjects >= 20 and subjects >= 100:
        return "more_stable"
    if positive_subjects >= 10 and subjects >= 50:
        return "interpret_cautiously"
    return "descriptive_only_small_event_count"


def subgroup_characterization(frame):
    variables = [
        "sex_group",
        "age_group",
        "race_group",
        "careunit_group",
        "admission_group",
        "insurance_group",
        "catheter_group",
    ]
    summaries = []
    seed_offset = 100
    for variable in variables:
        for level, group in frame.groupby(variable, dropna=False, sort=True):
            if group["subject_id"].nunique() < 10 or group[TARGET].nunique() < 2:
                continue
            summary = summarize_metrics(
                group,
                analysis="prespecified_subgroup",
                subgroup_variable=variable,
                subgroup_level=str(level),
                seed=RANDOM_STATE + seed_offset,
            )
            summary["reliability_flag"] = reliability_flag(
                int(summary["positive_subjects"].iloc[0]),
                int(summary["subjects"].iloc[0]),
            )
            summaries.append(summary)
            seed_offset += 1
    result = pd.concat(summaries, ignore_index=True)
    sparse = result["reliability_flag"].eq("descriptive_only_small_event_count")
    result.loc[sparse, ["ci_lower_95", "ci_upper_95"]] = np.nan
    result.loc[sparse, "valid_bootstrap_replicates"] = 0
    return result


def make_episode_frame(frame):
    ordered = frame.sort_values(
        ["episode_id", PROBABILITY, "landmark_hour"],
        ascending=[True, False, True],
    )
    episode = ordered.groupby("episode_id", as_index=False).head(1).copy()
    episode["episode_positive"] = episode["episode_id"].map(
        frame.groupby("episode_id")[TARGET].max()
    ).fillna(0).astype(int)
    return episode[["episode_id", "subject_id", "episode_positive", PROBABILITY]].reset_index(drop=True)


def episode_review_values(episode_frame, top_percent):
    ranked = episode_frame.sort_values(PROBABILITY, ascending=False)
    n_review = max(1, int(np.ceil(len(ranked) * top_percent / 100)))
    reviewed = ranked.head(n_review)
    true_positive = int(reviewed["episode_positive"].sum())
    total_positive = int(ranked["episode_positive"].sum())
    return {
        "episodes_reviewed": int(n_review),
        "review_fraction": n_review / len(ranked),
        "true_positive_episodes": true_positive,
        "false_positive_episodes": int(n_review - true_positive),
        "episode_ppv": safe_divide(true_positive, n_review),
        "episode_recall": safe_divide(true_positive, total_positive),
        "false_reviews_per_true_positive": safe_divide(n_review - true_positive, true_positive),
    }


def episode_review_characterization(frame):
    episode = make_episode_frame(frame)
    cluster_ids, cluster_rows = cluster_index_map(episode)
    rng = np.random.default_rng(RANDOM_STATE + 9000)
    rows = []
    for top_percent in [1, 2, 5, 10, 20]:
        point = episode_review_values(episode, top_percent)
        bootstrap = {metric: np.full(N_BOOTSTRAP, np.nan) for metric in point}
        for iteration in range(N_BOOTSTRAP):
            sampled_positions = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
            sampled_rows = np.concatenate([cluster_rows[position] for position in sampled_positions])
            sample = episode.iloc[sampled_rows]
            values = episode_review_values(sample, top_percent)
            for metric, value in values.items():
                bootstrap[metric][iteration] = value
        for metric, estimate in point.items():
            lower, upper, valid_replicates = interval(bootstrap[metric])
            rows.append({
                "top_percent": top_percent,
                "metric": metric,
                "estimate": estimate,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "valid_bootstrap_replicates": valid_replicates,
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "episodes": int(len(episode)),
                "positive_episodes": int(episode["episode_positive"].sum()),
                "subjects": int(episode["subject_id"].nunique()),
            })
    return pd.DataFrame(rows), episode


def calibration_deciles(frame):
    result = frame.copy()
    result["risk_decile"] = pd.qcut(
        result[PROBABILITY].rank(method="first"),
        q=10,
        labels=range(1, 11),
    ).astype(int)
    return (
        result.groupby("risk_decile")
        .agg(
            landmark_rows=("landmark_id", "size"),
            subjects=("subject_id", "nunique"),
            episodes=("episode_id", "nunique"),
            positive_landmark_rows=(TARGET, "sum"),
            mean_predicted_risk=(PROBABILITY, "mean"),
            observed_event_rate=(TARGET, "mean"),
            minimum_predicted_risk=(PROBABILITY, "min"),
            maximum_predicted_risk=(PROBABILITY, "max"),
        )
        .reset_index()
    )


def load_frame():
    print("Loading frozen Run 29 safe validation predictions...", flush=True)
    predictions = pd.read_csv(PREDICTION_FILE, parse_dates=["landmark_time"])
    predictions = predictions[predictions["model_variant"].eq(MODEL_VARIANT)].copy()
    context = pd.read_csv(FEATURE_FILE, usecols=CONTEXT_COLUMNS)
    frame = predictions.merge(context, on="landmark_id", how="left", validate="one_to_one")
    if not frame["anchor_year_group"].eq("2017 - 2019").all():
        raise ValueError("Run 30 expected only the frozen 2017-2019 validation frame.")
    if frame[PROBABILITY].isna().any():
        raise ValueError("Safe validation probabilities contain missing values.")
    if frame["subject_id"].isna().any():
        raise ValueError("Patient-cluster identifier is missing.")
    print(f"  Landmark rows: {len(frame):,}", flush=True)
    print(f"  Subjects:      {frame['subject_id'].nunique():,}", flush=True)
    print(f"  Episodes:      {frame['episode_id'].nunique():,}", flush=True)
    print(f"  Positive rows: {int(frame[TARGET].sum()):,} ({frame[TARGET].mean():.2%})", flush=True)
    return add_subgroups(frame)


def font(size, bold=False):
    path = r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def draw_calibration_plot(deciles):
    image = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(image)
    title = font(38, True)
    axis = font(25)
    tick = font(20)
    left, top, right, bottom = 170, 125, 1390, 860
    draw.text((750, 35), "Run 30 Leakage-Safe Validation Calibration", font=title, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    max_value = max(0.20, float(max(deciles["mean_predicted_risk"].max(), deciles["observed_event_rate"].max()) * 1.15))
    for fraction in np.linspace(0, 1, 6):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        value = fraction * max_value
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((x, bottom + 15), f"{value:.2f}", font=tick, fill="black", anchor="ma")
        draw.text((left - 15, y), f"{value:.2f}", font=tick, fill="black", anchor="rm")
    draw.line((left, bottom, right, top), fill=(110, 110, 110), width=3)
    points = []
    for row in deciles.itertuples():
        x = left + row.mean_predicted_risk / max_value * (right - left)
        y = bottom - row.observed_event_rate / max_value * (bottom - top)
        points.append((x, y))
    draw.line(points, fill=(31, 119, 180), width=5)
    for x, y in points:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(31, 119, 180))
    draw.text(((left + right) / 2, 985), "Mean predicted risk", font=axis, fill="black", anchor="ma")
    draw.text((left, 92), "Observed 7-day proxy event rate", font=axis, fill="black", anchor="la")
    image.save(PLOTS / "v0_5_run30_calibration_deciles.png")


def draw_review_policy_plot(review):
    pivot = review[review["metric"].isin(["episode_ppv", "episode_recall"])].copy()
    image = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(image)
    title = font(38, True)
    axis = font(25)
    tick = font(20)
    left, top, right, bottom = 170, 125, 1390, 860
    draw.text((750, 35), "Run 30 Episode Review Policy with Clustered 95% CIs", font=title, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    for fraction in np.linspace(0, 1, 6):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 15, y), f"{fraction:.1f}", font=tick, fill="black", anchor="rm")
    x_values = sorted(pivot["top_percent"].unique())
    x_min, x_max = min(x_values), max(x_values)
    colors = {"episode_ppv": (31, 119, 180), "episode_recall": (217, 95, 2)}
    labels = {"episode_ppv": "PPV", "episode_recall": "Recall"}
    for metric in ["episode_ppv", "episode_recall"]:
        rows = pivot[pivot["metric"].eq(metric)].sort_values("top_percent")
        points = []
        for row in rows.itertuples():
            x = left + (row.top_percent - x_min) / (x_max - x_min) * (right - left)
            y = bottom - row.estimate * (bottom - top)
            y_low = bottom - row.ci_lower_95 * (bottom - top)
            y_high = bottom - row.ci_upper_95 * (bottom - top)
            draw.line((x, y_low, x, y_high), fill=colors[metric], width=3)
            draw.line((x - 7, y_low, x + 7, y_low), fill=colors[metric], width=3)
            draw.line((x - 7, y_high, x + 7, y_high), fill=colors[metric], width=3)
            points.append((x, y))
        draw.line(points, fill=colors[metric], width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[metric])
    for value in x_values:
        x = left + (value - x_min) / (x_max - x_min) * (right - left)
        draw.text((x, bottom + 15), f"{value}%", font=tick, fill="black", anchor="ma")
    legend_y = 165
    for metric in ["episode_ppv", "episode_recall"]:
        draw.line((1030, legend_y, 1095, legend_y), fill=colors[metric], width=5)
        draw.text((1110, legend_y), labels[metric], font=tick, fill="black", anchor="lm")
        legend_y += 40
    draw.text(((left + right) / 2, 985), "Top fraction of episodes reviewed", font=axis, fill="black", anchor="ma")
    draw.text((left, 92), "Episode-level proportion", font=axis, fill="black", anchor="la")
    image.save(PLOTS / "v0_5_run30_episode_review_policy.png")


def draw_subgroup_lift_plot(subgroups):
    plot_data = subgroups[subgroups["metric"].eq("pr_auc_lift")].copy()
    plot_data = plot_data[
        ~plot_data["reliability_flag"].eq("descriptive_only_small_event_count")
    ].copy()
    plot_data = plot_data.sort_values(["subgroup_variable", "estimate"], ascending=[True, False])
    row_height = 42
    height = max(1100, 270 + row_height * len(plot_data))
    image = Image.new("RGB", (1800, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(38, True)
    tick = font(19)
    small = font(17)
    left, top, right = 610, 130, 1450
    bottom = height - 120
    draw.text((900, 35), "Run 30 Subgroup PR-AUC Lift with Patient-Clustered 95% CIs", font=title, fill="black", anchor="ma")
    draw.ellipse((650, 90, 664, 104), fill=(31, 119, 180))
    draw.text((676, 97), "more stable", font=small, fill="black", anchor="lm")
    draw.ellipse((850, 90, 864, 104), fill=(217, 95, 2))
    draw.text((876, 97), "interpret cautiously", font=small, fill="black", anchor="lm")
    finite_upper = plot_data["ci_upper_95"].replace([np.inf, -np.inf], np.nan).dropna()
    x_max = max(3.0, float(finite_upper.max() * 1.10)) if len(finite_upper) else 3.0
    for value in np.linspace(0, x_max, 6):
        x = left + value / x_max * (right - left)
        draw.line((x, top, x, bottom), fill=(225, 225, 225), width=1)
        draw.text((x, bottom + 15), f"{value:.1f}x", font=tick, fill="black", anchor="ma")
    x_one = left + 1 / x_max * (right - left)
    draw.line((x_one, top, x_one, bottom), fill=(90, 90, 90), width=3)
    for index, row in enumerate(plot_data.itertuples()):
        y = top + 30 + index * row_height
        variable = str(row.subgroup_variable).replace("_group", "").replace("_", " ")
        label = f"{variable}: {row.subgroup_level}"
        draw.text((left - 18, y), label, font=tick, fill="black", anchor="rm")
        color = (31, 119, 180) if row.reliability_flag == "more_stable" else (
            (217, 95, 2) if row.reliability_flag == "interpret_cautiously" else (130, 130, 130)
        )
        x = left + row.estimate / x_max * (right - left)
        x_low = left + max(0, row.ci_lower_95) / x_max * (right - left)
        x_high = left + min(x_max, row.ci_upper_95) / x_max * (right - left)
        draw.line((x_low, y, x_high, y), fill=color, width=4)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.text((right + 12, y), f"{int(row.positive_subjects)} positive patients", font=small, fill="black", anchor="lm")
    draw.text(((left + right) / 2, height - 45), "PR-AUC / subgroup prevalence", font=tick, fill="black", anchor="ma")
    image.save(PLOTS / "v0_5_run30_subgroup_pr_auc_lift.png")


def write_notes(overall, subgroups, review, deciles, frame):
    overall_metrics = overall.set_index("metric")
    review_10 = review[(review["top_percent"].eq(10))].set_index("metric")
    stable_groups = int(
        subgroups[subgroups["metric"].eq("pr_auc_lift")]["reliability_flag"].eq("more_stable").sum()
    )
    note = f"""# Run 30: Leakage-safe candidate characterization

## Decision question
How uncertain and heterogeneous is the frozen Run 29 leakage-safe candidate on the untouched 2017-2019 development-validation period, before external validation?

## Locked design
- Predictions: Run 29 `safe_exclude_early_positive`, Platt calibrated.
- Target: `{TARGET}` (7-day strict primary-or-uncertain CVC-associated BSI proxy).
- Evaluation period: 2017-2019 validation only.
- No model refitting, retuning, threshold optimization, or 2020-2022 scoring.
- Uncertainty: {N_BOOTSTRAP:,} patient-clustered bootstrap replicates. All landmarks and episodes for a sampled patient move together.

## Overall validation performance
- ROC-AUC: {overall_metrics.loc['roc_auc', 'estimate']:.4f} (95% CI {overall_metrics.loc['roc_auc', 'ci_lower_95']:.4f}-{overall_metrics.loc['roc_auc', 'ci_upper_95']:.4f}).
- PR-AUC: {overall_metrics.loc['pr_auc', 'estimate']:.4f} (95% CI {overall_metrics.loc['pr_auc', 'ci_lower_95']:.4f}-{overall_metrics.loc['pr_auc', 'ci_upper_95']:.4f}).
- PR-AUC lift over {overall_metrics.loc['prevalence', 'estimate']:.2%} prevalence: {overall_metrics.loc['pr_auc_lift', 'estimate']:.2f}x (95% CI {overall_metrics.loc['pr_auc_lift', 'ci_lower_95']:.2f}-{overall_metrics.loc['pr_auc_lift', 'ci_upper_95']:.2f}x).
- Brier Skill Score: {overall_metrics.loc['brier_skill_score', 'estimate']:.4f} (95% CI {overall_metrics.loc['brier_skill_score', 'ci_lower_95']:.4f}-{overall_metrics.loc['brier_skill_score', 'ci_upper_95']:.4f}).
- Calibration intercept: {overall_metrics.loc['calibration_intercept', 'estimate']:.3f}; slope: {overall_metrics.loc['calibration_slope', 'estimate']:.3f}; E:O: {overall_metrics.loc['expected_observed_ratio', 'estimate']:.3f}.

## Episode review policy
- Top 10% review PPV: {review_10.loc['episode_ppv', 'estimate']:.1%} (95% CI {review_10.loc['episode_ppv', 'ci_lower_95']:.1%}-{review_10.loc['episode_ppv', 'ci_upper_95']:.1%}).
- Top 10% positive-episode recall: {review_10.loc['episode_recall', 'estimate']:.1%} (95% CI {review_10.loc['episode_recall', 'ci_lower_95']:.1%}-{review_10.loc['episode_recall', 'ci_upper_95']:.1%}).
- False reviews per true positive at top 10%: {review_10.loc['false_reviews_per_true_positive', 'estimate']:.2f} (95% CI {review_10.loc['false_reviews_per_true_positive', 'ci_lower_95']:.2f}-{review_10.loc['false_reviews_per_true_positive', 'ci_upper_95']:.2f}).

## Subgroup interpretation
- Subgroups were prespecified from available demographic and episode context: sex, age, race, first ICU type, admission type, insurance, and catheter context.
- {stable_groups} subgroup levels met the descriptive `more_stable` rule (at least 20 positive patients and 100 patients).
- Cells with fewer than 10 positive patients retain descriptive point estimates, but their confidence intervals are suppressed and they are excluded from the forest plot. These are heterogeneity checks, not formal fairness claims or evidence of causal differences.
- No multiplicity-adjusted hypothesis testing was performed.

## Scope and limitations
- The confidence intervals measure sampling uncertainty conditional on the already fitted model and calibration map; they do not include model-development uncertainty.
- Repeated landmarks are handled by patient-clustered resampling, but single-center label error and transportability remain unresolved.
- Subgroup prevalence differs, so PR-AUC is interpreted alongside PR-AUC lift and event counts.
- The 2020-2022 period remains a post-hoc sensitivity cohort for the revised safe pipeline, not a pristine lockbox.
- External validation remains the required confirmation step.

## Data inventory
- {len(frame):,} landmark rows, {frame['subject_id'].nunique():,} patients, and {frame['episode_id'].nunique():,} episodes.
- {int(frame[TARGET].sum()):,} positive landmark rows ({frame[TARGET].mean():.2%}).
- Calibration table: {len(deciles)} equal-frequency risk groups.
"""
    (OUTPUT / "v0_5_run30_notes.md").write_text(note, encoding="utf-8")


def write_metadata(frame):
    metadata = {
        "run": 30,
        "name": "v0.5 Safe Candidate Characterization",
        "source_predictions": str(PREDICTION_FILE),
        "model_variant": MODEL_VARIANT,
        "probability": PROBABILITY,
        "target": TARGET,
        "validation_period": "2017-2019",
        "bootstrap_unit": "subject_id",
        "bootstrap_replicates": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "model_refit": False,
        "lockbox_scored": False,
        "landmark_rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "episodes": int(frame["episode_id"].nunique()),
    }
    (OUTPUT / "v0_5_run30_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_manifest():
    manifest = pd.DataFrame([
        {"artifact": path.name, "path": str(path), "bytes": path.stat().st_size}
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "v0_5_run30_manifest.csv"
    ])
    manifest.to_csv(OUTPUT / "v0_5_run30_manifest.csv", index=False)


def main():
    ensure_paths()
    frame = load_frame()

    print("Estimating overall patient-clustered confidence intervals...", flush=True)
    overall = summarize_metrics(frame, analysis="overall_validation")
    overall["reliability_flag"] = "primary"
    overall.to_csv(OUTPUT / "v0_5_run30_overall_metric_ci.csv", index=False)

    print("Characterizing prespecified subgroups...", flush=True)
    subgroups = subgroup_characterization(frame)
    subgroups.to_csv(OUTPUT / "v0_5_run30_subgroup_performance.csv", index=False)

    print("Characterizing episode review policies...", flush=True)
    review, episode = episode_review_characterization(frame)
    review.to_csv(OUTPUT / "v0_5_run30_episode_review_policy_ci.csv", index=False)
    episode.to_csv(OUTPUT / "v0_5_run30_episode_max_risk_predictions.csv", index=False)

    deciles = calibration_deciles(frame)
    deciles.to_csv(OUTPUT / "v0_5_run30_calibration_deciles.csv", index=False)

    draw_calibration_plot(deciles)
    draw_review_policy_plot(review)
    draw_subgroup_lift_plot(subgroups)
    write_notes(overall, subgroups, review, deciles, frame)
    write_metadata(frame)
    shutil.copy2(Path(__file__), OUTPUT / Path(__file__).name)
    write_manifest()

    print("", flush=True)
    print("Run 30 complete.", flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

