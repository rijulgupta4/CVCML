"""
Run 25: v0.5 locked temporal evaluation.

This is the first opened evaluation of the 2020-2022 temporal lockbox for the
frozen v0.5 candidate:

- Label: source-screened primary_or_uncertain strict CVC-associated BSI proxy
- Horizon: 7 days / 168 hours
- Model: Run 23 static_labs_vitals_therapy XGBoost
- Calibration: Run 23 Platt calibrator
- Use case: daily infection-prevention review list for active CVC episodes

The script must not refit the model, recalibrate, tune thresholds, or change
the feature set. It only scores the lockbox and reports the pre-specified
validation-compatible metrics and review-list policies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_ROOT / "data" / "v0_5"
RUN23_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 23 (v0.5 Label Sensitivity Modeling)"
RUN24_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 24 (v0.5 Operating Policy Characterization)"
RUN25_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 25 (v0.5 Locked Temporal Evaluation)"
PLOT_DIR = RUN25_OUTPUT_DIR / "plots"

FEATURE_FILE = DATA_PATH / "v0_5_run20_dynamic_enriched_features.csv"
LABEL_FILE = DATA_PATH / "v0_5_run22_source_screened_daily_landmarks.csv"
MODEL_FILE = RUN23_OUTPUT_DIR / "models" / "run23_xgboost_static_labs_vitals_therapy_primary_or_uncertain.joblib"
PLATT_FILE = RUN23_OUTPUT_DIR / "models" / "run23_platt_static_labs_vitals_therapy_primary_or_uncertain.joblib"

TARGET_LABEL = "primary_or_uncertain"
TARGET_ROLE = "candidate_primary"
TARGET_COL = "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"
LOCKBOX_YEAR_GROUP = "2020 - 2022"
SCORE_RAW = "raw_probability"
SCORE_PLATT = "platt_probability"

STATIC_NUMERIC = ["landmark_hour", "landmark_day", "anchor_age", "early_positive_culture"]
STATIC_CATEGORICAL = ["gender", "admission_type", "insurance", "race", "first_careunit"]
LAB_PREFIXES = ["wbc", "lactate", "hemoglobin", "platelets", "creatinine"]


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_platt(platt: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    return platt.predict_proba(logit(raw_prob).reshape(-1, 1))[:, 1]


def calibration_intercept_slope(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    X = logit(y_prob).reshape(-1, 1)
    try:
        model = LogisticRegression(solver="liblinear", C=1e6, max_iter=1000)
        model.fit(X, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def brier_skill_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    reference = float(brier_score_loss(y_true, np.full_like(y_true, prevalence, dtype=float)))
    return safe_div(reference - brier, reference), brier, reference


def expected_observed_ratio(y_true, y_prob):
    observed = float(np.sum(y_true))
    expected = float(np.sum(y_prob))
    return safe_div(expected, observed)


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def evaluate_predictions(df: pd.DataFrame, split_name: str, score_col: str, calibration: str) -> dict:
    y_true = df[TARGET_COL].astype(int).to_numpy()
    y_prob = df[score_col].to_numpy()
    prevalence = float(np.mean(y_true))
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    bss, brier, brier_reference = brier_skill_score(y_true, y_prob)
    cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "split": split_name,
        "target_label": TARGET_LABEL,
        "target_role": TARGET_ROLE,
        "target_col": TARGET_COL,
        "feature_set": "static_labs_vitals_therapy",
        "calibration": calibration,
        "score_name": f"static_labs_vitals_therapy_{calibration}_{TARGET_LABEL}",
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()),
        "patients": int(df["subject_id"].nunique()),
        "positive_rows": int(np.sum(y_true)),
        "positive_episodes": int(df.loc[df[TARGET_COL].eq(1), "episode_id"].nunique()),
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": safe_div(pr_auc, prevalence),
        "brier_score": brier,
        "brier_reference_prevalence": brier_reference,
        "brier_skill_score": bss,
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "expected_observed_ratio": expected_observed_ratio(y_true, y_prob),
    }


def review_summary(
    flagged: pd.DataFrame,
    all_rows: pd.DataFrame,
    policy_family: str,
    policy: str,
    *,
    score_col: str = SCORE_PLATT,
    policy_value: str | int | float | None = None,
    threshold: float | None = None,
    cooldown_hours: float | None = None,
) -> dict:
    flagged = flagged.copy()
    total_positive_rows = int(all_rows[TARGET_COL].sum())
    positive_episodes = set(all_rows.loc[all_rows[TARGET_COL].eq(1), "episode_id"])
    rows_reviewed = int(len(flagged))
    episodes_reviewed = int(flagged["episode_id"].nunique()) if rows_reviewed else 0
    tp_rows = int(flagged[TARGET_COL].sum()) if rows_reviewed else 0
    fp_rows = rows_reviewed - tp_rows
    captured_positive_episodes = set(flagged.loc[flagged[TARGET_COL].eq(1), "episode_id"]) if rows_reviewed else set()

    return {
        "split": "temporal_lockbox",
        "target_label": TARGET_LABEL,
        "target_role": TARGET_ROLE,
        "calibration": "platt",
        "score_name": f"static_labs_vitals_therapy_platt_{TARGET_LABEL}",
        "policy_family": policy_family,
        "policy": policy,
        "policy_value": policy_value,
        "threshold": threshold,
        "cooldown_hours": cooldown_hours,
        "rows_total": int(len(all_rows)),
        "episodes_total": int(all_rows["episode_id"].nunique()),
        "positive_rows_total": total_positive_rows,
        "positive_episodes_total": int(len(positive_episodes)),
        "base_row_prevalence": safe_div(total_positive_rows, len(all_rows)),
        "rows_reviewed": rows_reviewed,
        "reviewed_row_fraction": safe_div(rows_reviewed, len(all_rows)),
        "reviews_per_100_landmark_rows": 100 * safe_div(rows_reviewed, len(all_rows)),
        "episodes_reviewed": episodes_reviewed,
        "reviewed_episode_fraction": safe_div(episodes_reviewed, all_rows["episode_id"].nunique()),
        "reviews_per_100_episodes": 100 * safe_div(episodes_reviewed, all_rows["episode_id"].nunique()),
        "true_positive_rows": tp_rows,
        "false_positive_rows": fp_rows,
        "precision_ppv": safe_div(tp_rows, rows_reviewed),
        "ppv_lift_over_prevalence": safe_div(safe_div(tp_rows, rows_reviewed), safe_div(total_positive_rows, len(all_rows))),
        "row_recall_sensitivity": safe_div(tp_rows, total_positive_rows),
        "positive_episodes_captured": int(len(captured_positive_episodes)),
        "episode_recall_sensitivity": safe_div(len(captured_positive_episodes), len(positive_episodes)),
        "false_reviews_per_true_positive": safe_div(fp_rows, tp_rows),
        "repeat_review_rows": rows_reviewed - episodes_reviewed,
        "mean_reviews_per_reviewed_episode": safe_div(rows_reviewed, episodes_reviewed),
        "max_reviews_per_episode": int(flagged.groupby("episode_id").size().max()) if rows_reviewed else 0,
        "mean_review_score": flagged[score_col].mean() if rows_reviewed else np.nan,
        "min_review_score": flagged[score_col].min() if rows_reviewed else np.nan,
        "max_review_score": flagged[score_col].max() if rows_reviewed else np.nan,
    }


def make_topk_row_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(SCORE_PLATT, ascending=False)
    for pct in [0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]:
        n = max(1, math.ceil(len(ranked) * pct))
        policy = f"top_{pct:.1%}_rows".replace(".0%", "%")
        rows.append(review_summary(ranked.head(n), df, "top_row_budget", policy, policy_value=pct))
    for n in [25, 50, 100, 150, 250, 500, 750, 1000]:
        n_eff = min(n, len(ranked))
        rows.append(review_summary(ranked.head(n_eff), df, "top_row_budget", f"top_{n_eff}_rows", policy_value=n_eff))
    return pd.DataFrame(rows)


def make_topk_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_idx = df.groupby("episode_id")[SCORE_PLATT].idxmax()
    ranked = df.loc[best_idx].sort_values(SCORE_PLATT, ascending=False)
    for pct in [0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]:
        n = max(1, math.ceil(len(ranked) * pct))
        policy = f"top_{pct:.1%}_episodes".replace(".0%", "%")
        rows.append(review_summary(ranked.head(n), df, "top_episode_budget", policy, policy_value=pct))
    for n in [25, 50, 100, 150, 250, 500, 750, 1000]:
        n_eff = min(n, len(ranked))
        rows.append(review_summary(ranked.head(n_eff), df, "top_episode_budget", f"top_{n_eff}_episodes", policy_value=n_eff))
    return pd.DataFrame(rows)


def make_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]:
        flagged = df[df[SCORE_PLATT].ge(threshold)]
        rows.append(
            review_summary(
                flagged,
                df,
                "threshold_all_landmarks",
                f"risk_ge_{threshold:.3f}",
                policy_value=threshold,
                threshold=threshold,
            )
        )
    return pd.DataFrame(rows)


def make_first_alert_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]:
        alerts = (
            df[df[SCORE_PLATT].ge(threshold)]
            .sort_values(["episode_id", "landmark_hour", SCORE_PLATT], ascending=[True, True, False])
            .groupby("episode_id")
            .head(1)
        )
        rows.append(
            review_summary(
                alerts,
                df,
                "first_alert_per_episode",
                f"first_alert_risk_ge_{threshold:.3f}",
                policy_value=threshold,
                threshold=threshold,
            )
        )
    return pd.DataFrame(rows)


def cooldown_select(group: pd.DataFrame, cooldown_hours: float) -> pd.DataFrame:
    group = group.sort_values(["landmark_hour", SCORE_PLATT])
    selected = []
    last_hour = -np.inf
    for idx, row in group.iterrows():
        hour = row["landmark_hour"]
        if pd.isna(hour) or hour - last_hour >= cooldown_hours:
            selected.append(idx)
            last_hour = hour
    return group.loc[selected]


def make_cooldown_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]:
        over = df[df[SCORE_PLATT].ge(threshold)].copy()
        for cooldown in [48, 72, 168]:
            if over.empty:
                flagged = over
            else:
                flagged = pd.concat([cooldown_select(g, cooldown) for _, g in over.groupby("episode_id")], ignore_index=False)
            rows.append(
                review_summary(
                    flagged,
                    df,
                    "threshold_with_cooldown",
                    f"risk_ge_{threshold:.3f}_cooldown_{cooldown}h",
                    policy_value=f"{threshold}|{cooldown}",
                    threshold=threshold,
                    cooldown_hours=cooldown,
                )
            )
    return pd.DataFrame(rows)


def make_calibration_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["landmark_id", TARGET_COL, SCORE_PLATT]].copy()
    out["target_label"] = TARGET_LABEL
    out["target_role"] = TARGET_ROLE
    out["calibration"] = "platt"
    out["score_name"] = f"static_labs_vitals_therapy_platt_{TARGET_LABEL}"
    try:
        out["risk_decile"] = pd.qcut(out[SCORE_PLATT], q=10, labels=False, duplicates="drop")
    except ValueError:
        out["risk_decile"] = pd.cut(out[SCORE_PLATT], bins=10, labels=False)
    out["risk_decile"] = out["risk_decile"].astype("Int64")
    return (
        out
        .groupby(["target_label", "target_role", "calibration", "score_name", "risk_decile"], observed=True, dropna=True)
        .agg(
            rows=("landmark_id", "count"),
            mean_predicted_risk=(SCORE_PLATT, "mean"),
            observed_event_rate=(TARGET_COL, "mean"),
            positive_rows=(TARGET_COL, "sum"),
        )
        .reset_index()
    )


def make_selected_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    ranked_rows = df.sort_values(SCORE_PLATT, ascending=False)
    for policy, n in [
        ("top_5%_rows", max(1, math.ceil(len(df) * 0.05))),
        ("top_10%_rows", max(1, math.ceil(len(df) * 0.10))),
    ]:
        frames.append(ranked_rows.head(n).assign(policy_family="top_row_budget", policy=policy))

    best_idx = df.groupby("episode_id")[SCORE_PLATT].idxmax()
    ranked_episodes = df.loc[best_idx].sort_values(SCORE_PLATT, ascending=False)
    for policy, n in [
        ("top_100_episodes", 100),
        ("top_150_episodes", 150),
        ("top_250_episodes", 250),
    ]:
        frames.append(ranked_episodes.head(min(n, len(ranked_episodes))).assign(policy_family="top_episode_budget", policy=policy))

    keep = [
        "policy_family", "policy", "landmark_id", "episode_id", "subject_id", "hadm_id",
        "stay_id", "anchor_year_group", "landmark_hour", "landmark_day", "landmark_time",
        TARGET_COL, "target", SCORE_RAW, SCORE_PLATT,
    ]
    return pd.concat(frames, ignore_index=True)[[c for c in keep if c in df.columns or c in ["policy_family", "policy"]]]


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


FONT_TITLE = load_font(42, True)
FONT_LABEL = load_font(28)
FONT_TICK = load_font(23)


def draw_validation_lockbox_bars(comparison: pd.DataFrame, output_file: Path) -> None:
    metrics = [
        ("pr_auc_lift_over_prevalence", "PR-AUC lift"),
        ("precision_ppv_top_5_rows", "Top 5% PPV"),
        ("episode_recall_top_5_rows", "Top 5% episode recall"),
        ("brier_skill_score", "Brier skill"),
    ]
    img = Image.new("RGB", (1700, 1000), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 160, 160, 1540, 790
    draw.text((850, 55), "Run 25 Validation vs Locked Temporal Evaluation", font=FONT_TITLE, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    max_val = max(float(comparison[m[0]].max()) for m in metrics if m[0] in comparison.columns)
    y_max = max(1.0, math.ceil(max_val * 10) / 10)
    for frac in np.linspace(0, 1, 6):
        y = bottom - frac * (bottom - top)
        val = y_max * frac
        draw.line((left - 10, y, left, y), fill="black", width=2)
        draw.text((left - 20, y), f"{val:.2f}", font=FONT_TICK, fill="black", anchor="rm")
        if frac > 0:
            draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
    slot = (right - left) / len(metrics)
    colors = {"validation": (31, 119, 180), "temporal_lockbox": (214, 39, 40)}
    for i, (col, label) in enumerate(metrics):
        cx = left + slot * (i + 0.5)
        for j, split in enumerate(["validation", "temporal_lockbox"]):
            value = float(comparison.loc[comparison["split"].eq(split), col].iloc[0])
            bar_w = slot * 0.25
            x0 = cx + (j - 0.5) * bar_w * 1.4 - bar_w / 2
            x1 = x0 + bar_w
            h = value / y_max * (bottom - top)
            draw.rectangle((x0, bottom - h, x1, bottom), fill=colors[split])
            draw.text((x0 + bar_w / 2, bottom - h - 10), f"{value:.2f}", font=FONT_TICK, fill="black", anchor="ma")
        draw.text((cx, bottom + 24), label, font=FONT_TICK, fill="black", anchor="ma")
    draw.rectangle((1160, 125, 1190, 155), fill=colors["validation"])
    draw.text((1205, 120), "Validation", font=FONT_TICK, fill="black")
    draw.rectangle((1160, 165, 1190, 195), fill=colors["temporal_lockbox"])
    draw.text((1205, 160), "Temporal lockbox", font=FONT_TICK, fill="black")
    img.save(output_file)


def draw_calibration_deciles(deciles: pd.DataFrame, output_file: Path) -> None:
    img = Image.new("RGB", (1300, 950), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 140, 130, 1160, 800
    draw.text((650, 45), "Run 25 Lockbox Calibration by Risk Decile", font=FONT_TITLE, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)
    max_val = float(max(deciles["mean_predicted_risk"].max(), deciles["observed_event_rate"].max(), 0.01))
    y_max = min(1.0, max(0.10, math.ceil(max_val * 20) / 20))
    for frac in np.linspace(0, 1, 6):
        y = bottom - frac * (bottom - top)
        val = y_max * frac
        draw.text((left - 20, y), f"{val * 100:.0f}%", font=FONT_TICK, fill="black", anchor="rm")
        draw.line((left - 8, y, left, y), fill="black", width=2)
        if frac > 0:
            draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
    def xy(x, y):
        px = left + int(x / 9 * (right - left))
        py = bottom - int(y / y_max * (bottom - top))
        return px, py
    pred_points = [xy(i, row.mean_predicted_risk) for i, row in enumerate(deciles.itertuples())]
    obs_points = [xy(i, row.observed_event_rate) for i, row in enumerate(deciles.itertuples())]
    if len(pred_points) > 1:
        draw.line(pred_points, fill=(31, 119, 180), width=4)
        draw.line(obs_points, fill=(214, 39, 40), width=4)
    for p in pred_points:
        draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=(31, 119, 180))
    for p in obs_points:
        draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=(214, 39, 40))
    draw.text((650, 865), "Risk decile", font=FONT_LABEL, fill="black", anchor="ma")
    draw.text((30, 450), "Risk", font=FONT_LABEL, fill="black", anchor="lm")
    draw.rectangle((820, 135, 850, 165), fill=(31, 119, 180))
    draw.text((865, 130), "Mean predicted", font=FONT_TICK, fill="black")
    draw.rectangle((820, 175, 850, 205), fill=(214, 39, 40))
    draw.text((865, 170), "Observed", font=FONT_TICK, fill="black")
    img.save(output_file)


def write_notes(metrics: pd.DataFrame, top_rows: pd.DataFrame, top_episodes: pd.DataFrame, comparison: pd.DataFrame, output_file: Path) -> None:
    lockbox = metrics[(metrics["split"].eq("temporal_lockbox")) & (metrics["calibration"].eq("platt"))].iloc[0]
    top5 = top_rows[top_rows["policy"].eq("top_5%_rows")].iloc[0]
    top10 = top_rows[top_rows["policy"].eq("top_10%_rows")].iloc[0]
    ep150 = top_episodes[top_episodes["policy"].eq("top_150_episodes")].iloc[0] if (top_episodes["policy"].eq("top_150_episodes")).any() else None
    ep250 = top_episodes[top_episodes["policy"].eq("top_250_episodes")].iloc[0] if (top_episodes["policy"].eq("top_250_episodes")).any() else None

    lines = [
        "# Run 25 - v0.5 Locked Temporal Evaluation",
        "",
        "## Purpose",
        "",
        "Run 25 opens the 2020-2022 temporal lockbox once for the frozen v0.5 candidate. It loads the saved Run 23 XGBoost model and Platt calibrator, scores eligible lockbox landmark rows, and reports the same discrimination, calibration, and review-list metrics used during development.",
        "",
        "## Frozen Specification",
        "",
        f"- Label: `{TARGET_COL}` (`{TARGET_LABEL}`)",
        "- Horizon: 7 days / 168 hours",
        "- Feature set: static + labs + vitals + therapy context",
        "- Model: Run 23 XGBoost, no refitting",
        "- Calibration: Run 23 Platt calibrator, no recalibration",
        "- Use case: daily infection-prevention review list for active CVC episodes",
        "",
        "## Locked Test Performance",
        "",
        f"- Rows: {int(lockbox['rows']):,}; positive rows: {int(lockbox['positive_rows']):,}; prevalence: {fmt_pct(lockbox['prevalence'])}",
        f"- Episodes: {int(lockbox['episodes']):,}; positive episodes: {int(lockbox['positive_episodes']):,}",
        f"- ROC-AUC: {lockbox['roc_auc']:.3f}",
        f"- PR-AUC: {lockbox['pr_auc']:.3f}",
        f"- PR-AUC lift over prevalence: {lockbox['pr_auc_lift_over_prevalence']:.2f}x",
        f"- Brier Skill Score: {lockbox['brier_skill_score']:.3f}",
        f"- Expected:Observed ratio: {lockbox['expected_observed_ratio']:.2f}",
        "",
        "## Review-List Policies",
        "",
        f"- Top 5% rows: {int(top5['rows_reviewed'])} reviews, PPV {fmt_pct(top5['precision_ppv'])}, row recall {fmt_pct(top5['row_recall_sensitivity'])}, episode recall {fmt_pct(top5['episode_recall_sensitivity'])}, false reviews/TP {top5['false_reviews_per_true_positive']:.2f}.",
        f"- Top 10% rows: {int(top10['rows_reviewed'])} reviews, PPV {fmt_pct(top10['precision_ppv'])}, row recall {fmt_pct(top10['row_recall_sensitivity'])}, episode recall {fmt_pct(top10['episode_recall_sensitivity'])}, false reviews/TP {top10['false_reviews_per_true_positive']:.2f}.",
    ]

    if ep150 is not None:
        lines.append(f"- Top 150 episodes: {int(ep150['rows_reviewed'])} reviews, PPV {fmt_pct(ep150['precision_ppv'])}, episode recall {fmt_pct(ep150['episode_recall_sensitivity'])}, false reviews/TP {ep150['false_reviews_per_true_positive']:.2f}.")
    if ep250 is not None:
        lines.append(f"- Top 250 episodes: {int(ep250['rows_reviewed'])} reviews, PPV {fmt_pct(ep250['precision_ppv'])}, episode recall {fmt_pct(ep250['episode_recall_sensitivity'])}, false reviews/TP {ep250['false_reviews_per_true_positive']:.2f}.")

    lines.extend([
        "",
        "## Initial Interpretation Template",
        "",
        "Use the lockbox result to decide whether the project is strong, modest-but-useful, or weak. Do not tune on these results. If performance is modest but top-k yield remains above prevalence, the defensible claim is review-list prioritization rather than bedside alerting.",
    ])
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RUN25_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_FILE.exists() or not PLATT_FILE.exists():
        raise FileNotFoundError("Frozen Run 23 model/calibrator artifacts are missing. Run 25 must not refit them.")

    print("Loading features and source-screened labels...")
    features = pd.read_csv(FEATURE_FILE, parse_dates=["landmark_time"])
    labels = pd.read_csv(LABEL_FILE, usecols=["landmark_id", TARGET_COL])
    features = features.merge(labels, on="landmark_id", how="left", validate="one_to_one")
    features[TARGET_COL] = features[TARGET_COL].fillna(0).astype(int)

    lab_cols = sorted({c for c in features.columns if any(c.startswith(prefix) for prefix in LAB_PREFIXES)})
    vital_cols = sorted({c for c in features.columns if c.startswith("vital_")})
    therapy_cols = sorted({c for c in features.columns if c.startswith("abx_") or c.startswith("vaso_")})
    numeric_cols = STATIC_NUMERIC + lab_cols + vital_cols + therapy_cols
    categorical_cols = STATIC_CATEGORICAL
    feature_cols = numeric_cols + categorical_cols

    print("Loading frozen Run 23 model and Platt calibrator...")
    pipeline = joblib.load(MODEL_FILE)
    platt = joblib.load(PLATT_FILE)

    lockbox = features[
        features["anchor_year_group"].eq(LOCKBOX_YEAR_GROUP)
        & features["run18_primary_model_frame"].eq(1)
    ].copy()
    if lockbox.empty:
        raise ValueError("No eligible lockbox rows found.")

    print(f"Scoring lockbox rows: {len(lockbox):,}")
    raw_prob = pipeline.predict_proba(lockbox[feature_cols])[:, 1]
    platt_prob = apply_platt(platt, raw_prob)

    keep_cols = [
        "landmark_id", "episode_id", "subject_id", "hadm_id", "stay_id",
        "anchor_year_group", "split_role", "landmark_hour", "landmark_day",
        "landmark_time", TARGET_COL,
    ]
    scored = lockbox[keep_cols].copy()
    scored["target_label"] = TARGET_LABEL
    scored["target_role"] = TARGET_ROLE
    scored["target"] = scored[TARGET_COL]
    scored[SCORE_RAW] = raw_prob
    scored[SCORE_PLATT] = platt_prob

    metrics = pd.DataFrame([
        evaluate_predictions(scored, "temporal_lockbox", SCORE_RAW, "raw"),
        evaluate_predictions(scored, "temporal_lockbox", SCORE_PLATT, "platt"),
    ])

    top_rows = make_topk_row_table(scored)
    top_episodes = make_topk_episode_table(scored)
    thresholds = make_threshold_table(scored)
    first_alerts = make_first_alert_table(scored)
    cooldowns = make_cooldown_table(scored)
    policies = pd.concat([top_rows, top_episodes, thresholds, first_alerts, cooldowns], ignore_index=True)
    deciles = make_calibration_deciles(scored)
    selected = make_selected_review_rows(scored)

    validation_metrics = pd.read_csv(RUN23_OUTPUT_DIR / "v0_5_run23_label_sensitivity_model_comparison.csv")
    validation_platt = validation_metrics[
        validation_metrics["split"].eq("validation")
        & validation_metrics["target_label"].eq(TARGET_LABEL)
        & validation_metrics["calibration"].eq("platt")
    ].copy()

    run24_top_rows = pd.read_csv(RUN24_OUTPUT_DIR / "v0_5_run24_topk_row_policy.csv")
    val_top5 = run24_top_rows[run24_top_rows["policy"].eq("top_5%_rows")].iloc[0]
    lock_top5 = top_rows[top_rows["policy"].eq("top_5%_rows")].iloc[0]

    comparison = pd.concat([validation_platt, metrics[metrics["calibration"].eq("platt")]], ignore_index=True, sort=False)
    comparison["precision_ppv_top_5_rows"] = [val_top5["precision_ppv"], lock_top5["precision_ppv"]]
    comparison["episode_recall_top_5_rows"] = [val_top5["episode_recall_sensitivity"], lock_top5["episode_recall_sensitivity"]]
    comparison["false_reviews_per_tp_top_5_rows"] = [val_top5["false_reviews_per_true_positive"], lock_top5["false_reviews_per_true_positive"]]

    print("Saving Run 25 outputs...")
    metrics.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_model_comparison.csv", index=False)
    comparison.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_validation_lockbox_comparison.csv", index=False)
    top_rows.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_topk_row_policy.csv", index=False)
    top_episodes.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_topk_episode_policy.csv", index=False)
    thresholds.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_threshold_policy.csv", index=False)
    first_alerts.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_first_alert_policy.csv", index=False)
    cooldowns.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_cooldown_policy.csv", index=False)
    policies.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_operating_policy_summary.csv", index=False)
    deciles.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_calibration_deciles.csv", index=False)
    scored.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_predictions.csv", index=False)
    selected.to_csv(RUN25_OUTPUT_DIR / "v0_5_run25_selected_review_rows.csv", index=False)

    draw_validation_lockbox_bars(comparison, PLOT_DIR / "v0_5_run25_validation_vs_lockbox.png")
    draw_calibration_deciles(deciles, PLOT_DIR / "v0_5_run25_lockbox_calibration_deciles.png")
    write_notes(metrics, top_rows, top_episodes, comparison, RUN25_OUTPUT_DIR / "run_25_v0_5_locked_temporal_evaluation_notes.md")

    manifest = {
        "run": "Run 25 (v0.5 Locked Temporal Evaluation)",
        "lockbox_opened": True,
        "lockbox_year_group": LOCKBOX_YEAR_GROUP,
        "frozen_label": TARGET_COL,
        "frozen_model": str(MODEL_FILE),
        "frozen_calibrator": str(PLATT_FILE),
        "feature_file": str(FEATURE_FILE),
        "label_file": str(LABEL_FILE),
        "eligible_lockbox_rows": int(len(scored)),
        "positive_lockbox_rows": int(scored[TARGET_COL].sum()),
        "outputs": sorted([p.name for p in RUN25_OUTPUT_DIR.glob("v0_5_run25_*.csv")]) + [
            "run_25_v0_5_locked_temporal_evaluation_notes.md",
            "plots/v0_5_run25_validation_vs_lockbox.png",
            "plots/v0_5_run25_lockbox_calibration_deciles.png",
        ],
    }
    (RUN25_OUTPUT_DIR / "v0_5_run25_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("Run 25 lockbox evaluation complete.")
    print(metrics[metrics["calibration"].eq("platt")].round(4).to_string(index=False))
    print(f"Outputs saved to: {RUN25_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

