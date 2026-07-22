"""
Run 26: v0.5 locked error analysis.

This run characterizes the already-opened Run 25 temporal lockbox result.
It does not train, recalibrate, tune thresholds, or alter labels. It joins
the frozen lockbox predictions to source-screen labels, episode context,
culture details, and selected feature values to answer:

- Which lockbox positives were captured by the review-list policies?
- Which positives were missed?
- What clinical/context features distinguish TP, FP, FN, and low-risk TN rows?
- Are high-risk negatives clinically plausible false positives or possible label noise?
- Which organisms, source-screen classes, line types, and ICUs dominate errors?
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"C:\path\to\CVCML")
DATA_PATH = PROJECT_ROOT / "data" / "v0_5"
RUN25_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 25 (v0.5 Locked Temporal Evaluation)"
RUN26_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 26 (v0.5 Locked Error Analysis)"
PLOT_DIR = RUN26_OUTPUT_DIR / "plots"

PREDICTIONS_FILE = RUN25_OUTPUT_DIR / "v0_5_run25_lockbox_predictions.csv"
FEATURE_FILE = DATA_PATH / "v0_5_run20_dynamic_enriched_features.csv"
DAILY_LABEL_FILE = DATA_PATH / "v0_5_run22_source_screened_daily_landmarks.csv"
EPISODE_LABEL_FILE = DATA_PATH / "v0_5_run22_source_screened_episode_labels.csv"
CULTURE_DETAIL_FILE = DATA_PATH / "v0_5_episode_culture_detail.csv"

TARGET_COL = "future_strict_primary_or_uncertain_cvc_bsi_proxy_7d"
SCORE_COL = "platt_probability"

ROW_POLICY_PCTS = {
    "top_5_percent_rows": 0.05,
    "top_10_percent_rows": 0.10,
}
EPISODE_POLICY_COUNTS = {
    "top_100_episodes": 100,
    "top_150_episodes": 150,
    "top_250_episodes": 250,
}

CONTEXT_COLS = [
    "landmark_id",
    "episode_id",
    "exposure_start",
    "exposure_end_observed",
    "end_reason_observed",
    "prediction_window_end",
    "observed_window_end",
    "hours_to_observed_end",
    "full_7d_followup_observed",
    "future_strict_cvc_bsi_proxy_7d",
    "future_broad_cvc_bsi_proxy_7d",
    "future_strict_primary_likely_cvc_bsi_proxy_7d",
    "future_strict_secondary_possible_cvc_bsi_proxy_7d",
    "landmark_outcome_status",
    "hours_to_strict_event",
    "strict_proxy_culture_time",
    "strict_primary_or_uncertain_culture_time",
    "strict_secondary_possible_culture_time",
    "observed_exposure_hours",
    "raw_cvc_event_count",
    "cvc_types",
    "locations",
    "first_careunit",
    "last_careunit",
    "admission_type",
    "insurance",
    "race",
    "gender",
    "anchor_age",
    "early_positive_culture",
    "strict_proxy_positive_orgs",
    "strict_proxy_label_reason",
    "source_screen_class",
    "nearby_nonblood_source_culture_count",
    "concordant_nonblood_source_culture_count",
    "nonconcordant_nonblood_source_culture_count",
    "nearby_nonblood_source_buckets",
    "hadm_source_icd_count",
    "hadm_source_icd_buckets",
]

FEATURE_COLS = [
    "landmark_id",
    "lactate_last",
    "lactate_mean",
    "lactate_trend",
    "wbc_last",
    "wbc_mean",
    "wbc_trend",
    "platelets_last",
    "platelets_mean",
    "platelets_trend",
    "creatinine_last",
    "creatinine_mean",
    "creatinine_trend",
    "hemoglobin_last",
    "hemoglobin_mean",
    "hemoglobin_trend",
    "vital_temperature_c_max_24h",
    "vital_temperature_c_mean_24h",
    "vital_heart_rate_max_24h",
    "vital_heart_rate_mean_24h",
    "vital_respiratory_rate_max_24h",
    "vital_map_min_24h",
    "vital_sbp_min_24h",
    "vital_spo2_min_24h",
    "abx_antibiotic_any_active_24h",
    "abx_antibiotic_any_starts_24h",
    "abx_broad_antibiotic_active_24h",
    "abx_broad_antibiotic_starts_24h",
    "abx_anti_mrsa_antibiotic_active_24h",
    "abx_antipseudomonal_antibiotic_active_24h",
    "vaso_vasopressor_any_active_24h",
    "vaso_vasopressor_any_starts_24h",
    "vaso_rate_max_24h",
    "lactate_lab_count",
    "wbc_lab_count",
    "platelets_lab_count",
    "creatinine_lab_count",
]


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def get_font(size=28, bold=False):
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


FONT_TITLE = get_font(42, True)
FONT_LABEL = get_font(28)
FONT_TICK = get_font(22)
FONT_SMALL = get_font(18)


def load_analysis_frame() -> pd.DataFrame:
    print("Loading Run 25 lockbox predictions...")
    preds = pd.read_csv(PREDICTIONS_FILE, parse_dates=["landmark_time"])
    preds[TARGET_COL] = preds[TARGET_COL].astype(int)
    preds["target"] = preds["target"].astype(int)

    print("Loading source-screened daily labels...")
    available_context = pd.read_csv(DAILY_LABEL_FILE, nrows=0).columns.tolist()
    context_cols = [c for c in CONTEXT_COLS if c in available_context]
    labels = pd.read_csv(
        DAILY_LABEL_FILE,
        usecols=context_cols,
        parse_dates=[
            c for c in [
                "exposure_start",
                "exposure_end_observed",
                "prediction_window_end",
                "observed_window_end",
                "strict_proxy_culture_time",
                "strict_primary_or_uncertain_culture_time",
                "strict_secondary_possible_culture_time",
            ] if c in context_cols
        ],
    )

    print("Loading selected dynamic feature columns...")
    available_features = pd.read_csv(FEATURE_FILE, nrows=0).columns.tolist()
    feature_cols = [c for c in FEATURE_COLS if c in available_features]
    feature_df = pd.read_csv(FEATURE_FILE, usecols=feature_cols)

    df = preds.merge(labels, on=["landmark_id", "episode_id"], how="left", validate="one_to_one")
    df = df.merge(feature_df, on="landmark_id", how="left", validate="one_to_one")

    ranked = df.sort_values(SCORE_COL, ascending=False)
    for policy, pct in ROW_POLICY_PCTS.items():
        n = max(1, math.ceil(len(ranked) * pct))
        top_ids = set(ranked.head(n)["landmark_id"])
        df[policy] = df["landmark_id"].isin(top_ids).astype(int)

    episode_best_idx = df.groupby("episode_id")[SCORE_COL].idxmax()
    episode_ranked = df.loc[episode_best_idx].sort_values(SCORE_COL, ascending=False)
    for policy, n in EPISODE_POLICY_COUNTS.items():
        top_episode_ids = set(episode_ranked.head(min(n, len(episode_ranked)))["episode_id"])
        df[policy] = df["episode_id"].isin(top_episode_ids).astype(int)

    df["top_10_error_group"] = np.select(
        [
            df["top_10_percent_rows"].eq(1) & df[TARGET_COL].eq(1),
            df["top_10_percent_rows"].eq(1) & df[TARGET_COL].eq(0),
            df["top_10_percent_rows"].eq(0) & df[TARGET_COL].eq(1),
            df["top_10_percent_rows"].eq(0) & df[TARGET_COL].eq(0),
        ],
        ["TP_reviewed_positive", "FP_reviewed_negative", "FN_missed_positive", "TN_not_reviewed_negative"],
        default="unclassified",
    )
    df["top_5_error_group"] = np.select(
        [
            df["top_5_percent_rows"].eq(1) & df[TARGET_COL].eq(1),
            df["top_5_percent_rows"].eq(1) & df[TARGET_COL].eq(0),
            df["top_5_percent_rows"].eq(0) & df[TARGET_COL].eq(1),
            df["top_5_percent_rows"].eq(0) & df[TARGET_COL].eq(0),
        ],
        ["TP_reviewed_positive", "FP_reviewed_negative", "FN_missed_positive", "TN_not_reviewed_negative"],
        default="unclassified",
    )
    return df


def build_row_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in ["top_5_percent_rows", "top_10_percent_rows"]:
        group_col = policy.replace("percent_rows", "error_group")
        # Correct the generated group names.
        group_col = "top_5_error_group" if policy.startswith("top_5") else "top_10_error_group"
        for group, g in df.groupby(group_col, dropna=False):
            rows.append({
                "policy": policy,
                "error_group": group,
                "rows": int(len(g)),
                "row_fraction": safe_div(len(g), len(df)),
                "positive_rows": int(g[TARGET_COL].sum()),
                "mean_score": float(g[SCORE_COL].mean()),
                "median_score": float(g[SCORE_COL].median()),
                "mean_landmark_hour": float(g["landmark_hour"].mean()),
                "median_landmark_hour": float(g["landmark_hour"].median()),
                "mean_hours_to_strict_event": float(g["hours_to_strict_event"].mean()) if "hours_to_strict_event" in g else np.nan,
                "median_hours_to_strict_event": float(g["hours_to_strict_event"].median()) if "hours_to_strict_event" in g else np.nan,
                "unique_episodes": int(g["episode_id"].nunique()),
                "source_screen_primary_likely_rows": int(g.get("future_strict_primary_likely_cvc_bsi_proxy_7d", pd.Series(dtype=int)).fillna(0).sum()) if "future_strict_primary_likely_cvc_bsi_proxy_7d" in g else np.nan,
                "source_screen_secondary_possible_rows": int(g.get("future_strict_secondary_possible_cvc_bsi_proxy_7d", pd.Series(dtype=int)).fillna(0).sum()) if "future_strict_secondary_possible_cvc_bsi_proxy_7d" in g else np.nan,
            })
    return pd.DataFrame(rows)


def build_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.sort_values(SCORE_COL, ascending=False)
        .groupby("episode_id")
        .agg(
            subject_id=("subject_id", "first"),
            hadm_id=("hadm_id", "first"),
            stay_id=("stay_id", "first"),
            anchor_year_group=("anchor_year_group", "first"),
            positive_episode=(TARGET_COL, "max"),
            max_score=(SCORE_COL, "max"),
            mean_score=(SCORE_COL, "mean"),
            landmark_rows=("landmark_id", "count"),
            positive_landmark_rows=(TARGET_COL, "sum"),
            first_landmark_hour=("landmark_hour", "min"),
            last_landmark_hour=("landmark_hour", "max"),
            max_score_landmark_hour=("landmark_hour", "first"),
            top_5_percent_rows=("top_5_percent_rows", "max"),
            top_10_percent_rows=("top_10_percent_rows", "max"),
            top_100_episodes=("top_100_episodes", "max"),
            top_150_episodes=("top_150_episodes", "max"),
            top_250_episodes=("top_250_episodes", "max"),
            cvc_types=("cvc_types", "first"),
            locations=("locations", "first"),
            first_careunit=("first_careunit", "first"),
            last_careunit=("last_careunit", "first"),
            source_screen_class=("source_screen_class", "first"),
            strict_proxy_positive_orgs=("strict_proxy_positive_orgs", "first"),
            strict_proxy_label_reason=("strict_proxy_label_reason", "first"),
            strict_primary_or_uncertain_culture_time=("strict_primary_or_uncertain_culture_time", "first"),
            nearby_nonblood_source_culture_count=("nearby_nonblood_source_culture_count", "max"),
            concordant_nonblood_source_culture_count=("concordant_nonblood_source_culture_count", "max"),
            hadm_source_icd_count=("hadm_source_icd_count", "max"),
            nearby_nonblood_source_buckets=("nearby_nonblood_source_buckets", "first"),
            hadm_source_icd_buckets=("hadm_source_icd_buckets", "first"),
        )
        .reset_index()
    )
    agg["capture_group_top_10_rows"] = np.select(
        [
            agg["positive_episode"].eq(1) & agg["top_10_percent_rows"].eq(1),
            agg["positive_episode"].eq(1) & agg["top_10_percent_rows"].eq(0),
            agg["positive_episode"].eq(0) & agg["top_10_percent_rows"].eq(1),
            agg["positive_episode"].eq(0) & agg["top_10_percent_rows"].eq(0),
        ],
        ["captured_positive_episode", "missed_positive_episode", "reviewed_negative_episode", "not_reviewed_negative_episode"],
        default="unclassified",
    )
    return agg.sort_values(["positive_episode", "max_score"], ascending=[False, False])


def build_episode_capture_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in ["top_5_percent_rows", "top_10_percent_rows", "top_100_episodes", "top_150_episodes", "top_250_episodes"]:
        positive = episodes[episodes["positive_episode"].eq(1)]
        reviewed = episodes[episodes[policy].eq(1)]
        captured = positive[positive[policy].eq(1)]
        rows.append({
            "policy": policy,
            "episodes_reviewed": int(len(reviewed)),
            "positive_episodes_total": int(len(positive)),
            "positive_episodes_captured": int(len(captured)),
            "episode_recall": safe_div(len(captured), len(positive)),
            "reviewed_episode_ppv": safe_div(reviewed["positive_episode"].sum(), len(reviewed)),
            "false_reviewed_episodes": int(len(reviewed) - reviewed["positive_episode"].sum()),
            "false_episode_reviews_per_true_positive": safe_div(len(reviewed) - reviewed["positive_episode"].sum(), reviewed["positive_episode"].sum()),
            "median_max_score_reviewed": float(reviewed["max_score"].median()) if len(reviewed) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_categorical(df: pd.DataFrame, group_col: str, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for cat_col in cat_cols:
        if cat_col not in df.columns:
            continue
        temp = df[[group_col, cat_col]].copy()
        temp[cat_col] = temp[cat_col].fillna("Missing").astype(str)
        for group, g in temp.groupby(group_col, dropna=False):
            counts = g[cat_col].value_counts(dropna=False).head(12)
            total = len(g)
            for value, count in counts.items():
                rows.append({
                    "group_col": group_col,
                    "group": group,
                    "field": cat_col,
                    "value": value,
                    "rows": int(count),
                    "group_rows": int(total),
                    "within_group_fraction": safe_div(count, total),
                })
    return pd.DataFrame(rows)


def summarize_organisms(episodes: pd.DataFrame) -> pd.DataFrame:
    positive = episodes[episodes["positive_episode"].eq(1)].copy()
    rows = []
    for _, row in positive.iterrows():
        org_text = str(row.get("strict_proxy_positive_orgs", "") or "")
        if org_text in ["", "nan", "None"]:
            orgs = ["Missing"]
        else:
            orgs = [part.strip() for part in org_text.replace("|", ";").split(";") if part.strip()]
            if not orgs:
                orgs = [org_text]
        capture = "captured_top10" if row["top_10_percent_rows"] == 1 else "missed_top10"
        for org in orgs:
            rows.append({
                "episode_id": row["episode_id"],
                "capture_group": capture,
                "organism": org,
                "max_score": row["max_score"],
                "cvc_types": row.get("cvc_types"),
                "first_careunit": row.get("first_careunit"),
                "source_screen_class": row.get("source_screen_class"),
            })
    org_df = pd.DataFrame(rows)
    if org_df.empty:
        return org_df
    return (
        org_df
        .groupby(["capture_group", "organism"], dropna=False)
        .agg(
            episodes=("episode_id", "nunique"),
            mean_max_score=("max_score", "mean"),
        )
        .reset_index()
        .sort_values(["capture_group", "episodes"], ascending=[True, False])
    )


def feature_contrast(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in FEATURE_COLS if c in df.columns and c != "landmark_id"]
    groups = ["TP_reviewed_positive", "FP_reviewed_negative", "FN_missed_positive", "TN_not_reviewed_negative"]
    rows = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        for group in groups:
            mask = df["top_10_error_group"].eq(group)
            g = values[mask]
            rows.append({
                "feature": col,
                "error_group": group,
                "rows_nonmissing": int(g.notna().sum()),
                "mean": float(g.mean()) if g.notna().any() else np.nan,
                "median": float(g.median()) if g.notna().any() else np.nan,
                "missing_fraction": float(g.isna().mean()) if len(g) else np.nan,
            })
    out = pd.DataFrame(rows)

    # Add a simple standardized contrast: reviewed negatives vs missed positives.
    pivot = out.pivot(index="feature", columns="error_group", values="mean")
    overall_std = df[feature_cols].apply(pd.to_numeric, errors="coerce").std(axis=0)
    contrast = []
    for feature in feature_cols:
        std = overall_std.get(feature, np.nan)
        fp_mean = pivot.loc[feature].get("FP_reviewed_negative", np.nan) if feature in pivot.index else np.nan
        fn_mean = pivot.loc[feature].get("FN_missed_positive", np.nan) if feature in pivot.index else np.nan
        tp_mean = pivot.loc[feature].get("TP_reviewed_positive", np.nan) if feature in pivot.index else np.nan
        tn_mean = pivot.loc[feature].get("TN_not_reviewed_negative", np.nan) if feature in pivot.index else np.nan
        contrast.append({
            "feature": feature,
            "fp_minus_fn_standardized": safe_div(fp_mean - fn_mean, std),
            "tp_minus_fn_standardized": safe_div(tp_mean - fn_mean, std),
            "fp_minus_tn_standardized": safe_div(fp_mean - tn_mean, std),
        })
    return out.merge(pd.DataFrame(contrast), on="feature", how="left")


def build_high_risk_negative_table(df: pd.DataFrame) -> pd.DataFrame:
    high_fp = df[df["top_10_error_group"].eq("FP_reviewed_negative")].copy()
    cols = [
        "landmark_id", "episode_id", "subject_id", "hadm_id", "stay_id",
        "landmark_hour", "landmark_day", SCORE_COL,
        "future_broad_cvc_bsi_proxy_7d",
        "future_strict_cvc_bsi_proxy_7d",
        "future_strict_secondary_possible_cvc_bsi_proxy_7d",
        "landmark_outcome_status",
        "hours_to_observed_end",
        "full_7d_followup_observed",
        "source_screen_class",
        "nearby_nonblood_source_culture_count",
        "concordant_nonblood_source_culture_count",
        "nearby_nonblood_source_buckets",
        "hadm_source_icd_count",
        "hadm_source_icd_buckets",
        "first_careunit",
        "cvc_types",
        "lactate_last",
        "wbc_last",
        "platelets_last",
        "vital_temperature_c_max_24h",
        "vital_heart_rate_max_24h",
        "abx_broad_antibiotic_active_24h",
        "vaso_vasopressor_any_active_24h",
    ]
    cols = [c for c in cols if c in high_fp.columns]
    return high_fp.sort_values(SCORE_COL, ascending=False)[cols]


def build_missed_positive_table(df: pd.DataFrame) -> pd.DataFrame:
    missed = df[df["top_10_error_group"].eq("FN_missed_positive")].copy()
    cols = [
        "landmark_id", "episode_id", "subject_id", "hadm_id", "stay_id",
        "landmark_hour", "landmark_day", SCORE_COL,
        "hours_to_strict_event",
        "strict_primary_or_uncertain_culture_time",
        "strict_proxy_positive_orgs",
        "strict_proxy_label_reason",
        "source_screen_class",
        "first_careunit",
        "cvc_types",
        "lactate_last",
        "wbc_last",
        "platelets_last",
        "vital_temperature_c_max_24h",
        "vital_heart_rate_max_24h",
        "abx_broad_antibiotic_active_24h",
        "vaso_vasopressor_any_active_24h",
    ]
    cols = [c for c in cols if c in missed.columns]
    return missed.sort_values([SCORE_COL, "hours_to_strict_event"], ascending=[False, True])[cols]


def culture_context() -> pd.DataFrame:
    cultures = pd.read_csv(CULTURE_DETAIL_FILE, parse_dates=["charttime", "exposure_start", "exposure_end_observed"])
    lockbox_episode_ids = pd.read_csv(PREDICTIONS_FILE, usecols=["episode_id"])["episode_id"].unique()
    cultures = cultures[cultures["episode_id"].isin(lockbox_episode_ids)].copy()
    return (
        cultures
        .groupby(["episode_id", "spec_type_desc", "org_name"], dropna=False)
        .agg(
            culture_rows=("charttime", "count"),
            first_culture_time=("charttime", "min"),
            qualifying_cvc_associated_rows=("qualifying_cvc_associated_culture", "sum"),
            early_positive_rows=("early_positive_culture", "sum"),
            min_hours_from_exposure_start=("hours_from_exposure_start", "min"),
        )
        .reset_index()
        .sort_values(["episode_id", "culture_rows"], ascending=[True, False])
    )


def draw_score_distribution(df: pd.DataFrame, output_file: Path) -> None:
    img = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 130, 120, 1250, 760
    draw.text((700, 45), "Run 26 Lockbox Score Distribution by Outcome", font=FONT_TITLE, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)

    pos = df[df[TARGET_COL].eq(1)][SCORE_COL].dropna().to_numpy()
    neg = df[df[TARGET_COL].eq(0)][SCORE_COL].dropna().to_numpy()
    bins = np.linspace(0, max(df[SCORE_COL].max(), 0.001), 21)
    pos_hist, _ = np.histogram(pos, bins=bins)
    neg_hist, _ = np.histogram(neg, bins=bins)
    pos_frac = pos_hist / pos_hist.sum() if pos_hist.sum() else pos_hist
    neg_frac = neg_hist / neg_hist.sum() if neg_hist.sum() else neg_hist
    y_max = max(pos_frac.max(), neg_frac.max(), 0.01) * 1.15
    bar_slot = (right - left) / (len(bins) - 1)
    for i in range(len(bins) - 1):
        x0 = left + i * bar_slot
        neg_h = neg_frac[i] / y_max * (bottom - top)
        pos_h = pos_frac[i] / y_max * (bottom - top)
        draw.rectangle((x0 + 2, bottom - neg_h, x0 + bar_slot * 0.45, bottom), fill=(31, 119, 180))
        draw.rectangle((x0 + bar_slot * 0.50, bottom - pos_h, x0 + bar_slot - 2, bottom), fill=(214, 39, 40))
    draw.text((700, 820), "Calibrated predicted risk", font=FONT_LABEL, fill="black", anchor="ma")
    draw.text((30, 420), "Fraction within class", font=FONT_LABEL, fill="black", anchor="lm")
    draw.rectangle((920, 130, 950, 160), fill=(31, 119, 180))
    draw.text((965, 125), "Negative rows", font=FONT_TICK, fill="black")
    draw.rectangle((920, 175, 950, 205), fill=(214, 39, 40))
    draw.text((965, 170), "Positive rows", font=FONT_TICK, fill="black")
    img.save(output_file)


def draw_error_group_feature_bars(feature_summary: pd.DataFrame, output_file: Path) -> None:
    contrast = (
        feature_summary[["feature", "fp_minus_fn_standardized", "tp_minus_fn_standardized"]]
        .drop_duplicates()
        .assign(abs_contrast=lambda d: d["fp_minus_fn_standardized"].abs())
        .sort_values("abs_contrast", ascending=False)
        .head(14)
    )
    img = Image.new("RGB", (1500, 1000), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 420, 100, 1380, 880
    draw.text((750, 35), "Run 26 Largest Feature Contrasts: FP vs Missed Positive", font=FONT_TITLE, fill="black", anchor="ma")
    max_abs = max(1.0, float(contrast["fp_minus_fn_standardized"].abs().max()) * 1.1)
    zero_x = left + (right - left) / 2
    draw.line((zero_x, top, zero_x, bottom), fill="black", width=2)
    row_h = (bottom - top) / max(len(contrast), 1)
    for i, row in enumerate(contrast.itertuples()):
        y = top + i * row_h + row_h * 0.5
        draw.text((left - 15, y), row.feature.replace("_", " "), font=FONT_SMALL, fill="black", anchor="rm")
        val = row.fp_minus_fn_standardized
        x = zero_x + val / max_abs * (right - left) / 2
        color = (214, 39, 40) if val > 0 else (31, 119, 180)
        draw.rectangle((min(zero_x, x), y - 12, max(zero_x, x), y + 12), fill=color)
        draw.text((x + (8 if val >= 0 else -8), y), f"{val:.2f}", font=FONT_SMALL, fill="black", anchor="lm" if val >= 0 else "rm")
    draw.text((750, 940), "Standardized mean difference: high-risk negatives minus missed positives", font=FONT_LABEL, fill="black", anchor="ma")
    img.save(output_file)


def write_notes(row_summary: pd.DataFrame, episode_capture: pd.DataFrame, feature_summary: pd.DataFrame, output_file: Path) -> None:
    top5_tp = row_summary[(row_summary["policy"].eq("top_5_percent_rows")) & (row_summary["error_group"].eq("TP_reviewed_positive"))].iloc[0]
    top5_fp = row_summary[(row_summary["policy"].eq("top_5_percent_rows")) & (row_summary["error_group"].eq("FP_reviewed_negative"))].iloc[0]
    top10_tp = row_summary[(row_summary["policy"].eq("top_10_percent_rows")) & (row_summary["error_group"].eq("TP_reviewed_positive"))].iloc[0]
    top10_fp = row_summary[(row_summary["policy"].eq("top_10_percent_rows")) & (row_summary["error_group"].eq("FP_reviewed_negative"))].iloc[0]
    top10_fn = row_summary[(row_summary["policy"].eq("top_10_percent_rows")) & (row_summary["error_group"].eq("FN_missed_positive"))].iloc[0]
    ep150 = episode_capture[episode_capture["policy"].eq("top_150_episodes")].iloc[0]

    feature_contrast_top = (
        feature_summary[["feature", "fp_minus_fn_standardized"]]
        .drop_duplicates()
        .dropna()
        .assign(abs_contrast=lambda d: d["fp_minus_fn_standardized"].abs())
        .sort_values("abs_contrast", ascending=False)
        .head(8)
    )

    lines = [
        "# Run 26 - v0.5 Locked Error Analysis",
        "",
        "## Purpose",
        "",
        "Run 26 characterizes the locked Run 25 temporal evaluation. It does not refit, recalibrate, tune thresholds, or revise labels. It joins the frozen lockbox predictions to source-screen labels, episode context, culture detail, and selected feature values.",
        "",
        "## Review-List Error Groups",
        "",
        f"- Top 5% rows reviewed {int(top5_tp['rows'] + top5_fp['rows'])} rows: {int(top5_tp['rows'])} true-positive rows and {int(top5_fp['rows'])} false-positive rows.",
        f"- Top 10% rows reviewed {int(top10_tp['rows'] + top10_fp['rows'])} rows: {int(top10_tp['rows'])} true-positive rows, {int(top10_fp['rows'])} false-positive rows, and left {int(top10_fn['rows'])} positive rows unreviewed.",
        f"- Top 150 episode review captured {fmt_pct(ep150['episode_recall'])} of positive episodes with reviewed-episode PPV {fmt_pct(ep150['reviewed_episode_ppv'])}.",
        "",
        "## Feature Contrast Signal",
        "",
        "Largest standardized contrasts between high-risk negatives and missed positive rows:",
    ]
    for _, row in feature_contrast_top.iterrows():
        lines.append(f"- {row['feature']}: FP - FN standardized mean difference {row['fp_minus_fn_standardized']:.2f}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This run should be used to decide whether high-risk false positives are clinically plausible review candidates or mostly noise.",
        "- If high-risk negatives show infection-like physiology, therapy exposure, source-culture context, or care intensity, the modest PPV is less damaging because the review list may still enrich for clinically concerning CVC episodes.",
        "- Missed positives define the next label/feature-improvement target, but the lockbox itself should not be used for tuning.",
    ])
    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RUN26_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_analysis_frame()
    episodes = build_episode_summary(df)
    row_summary = build_row_error_summary(df)
    episode_capture = build_episode_capture_summary(episodes)
    categorical_summary = summarize_categorical(
        df,
        "top_10_error_group",
        [
            "source_screen_class",
            "strict_proxy_label_reason",
            "first_careunit",
            "last_careunit",
            "cvc_types",
            "locations",
            "nearby_nonblood_source_buckets",
            "hadm_source_icd_buckets",
            "landmark_outcome_status",
        ],
    )
    organism_summary = summarize_organisms(episodes)
    feature_summary = feature_contrast(df)
    high_risk_negatives = build_high_risk_negative_table(df)
    missed_positives = build_missed_positive_table(df)
    cultures = culture_context()

    print("Saving Run 26 outputs...")
    df.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_lockbox_error_analysis_rows.csv", index=False)
    episodes.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_lockbox_episode_error_summary.csv", index=False)
    row_summary.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_row_error_group_summary.csv", index=False)
    episode_capture.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_episode_capture_summary.csv", index=False)
    categorical_summary.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_categorical_error_summary.csv", index=False)
    organism_summary.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_organism_capture_summary.csv", index=False)
    feature_summary.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_feature_error_contrast.csv", index=False)
    high_risk_negatives.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_high_risk_negative_review_rows.csv", index=False)
    missed_positives.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_missed_positive_rows.csv", index=False)
    cultures.to_csv(RUN26_OUTPUT_DIR / "v0_5_run26_lockbox_culture_context.csv", index=False)

    draw_score_distribution(df, PLOT_DIR / "v0_5_run26_score_distribution_by_outcome.png")
    draw_error_group_feature_bars(feature_summary, PLOT_DIR / "v0_5_run26_feature_contrast_fp_vs_fn.png")
    write_notes(row_summary, episode_capture, feature_summary, RUN26_OUTPUT_DIR / "run_26_v0_5_locked_error_analysis_notes.md")

    manifest = {
        "run": "Run 26 (v0.5 Locked Error Analysis)",
        "source_predictions": str(PREDICTIONS_FILE),
        "model_changed": False,
        "calibration_changed": False,
        "label_changed": False,
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()),
        "positive_rows": int(df[TARGET_COL].sum()),
        "positive_episodes": int(episodes["positive_episode"].sum()),
        "outputs": [
            "v0_5_run26_lockbox_error_analysis_rows.csv",
            "v0_5_run26_lockbox_episode_error_summary.csv",
            "v0_5_run26_row_error_group_summary.csv",
            "v0_5_run26_episode_capture_summary.csv",
            "v0_5_run26_categorical_error_summary.csv",
            "v0_5_run26_organism_capture_summary.csv",
            "v0_5_run26_feature_error_contrast.csv",
            "v0_5_run26_high_risk_negative_review_rows.csv",
            "v0_5_run26_missed_positive_rows.csv",
            "v0_5_run26_lockbox_culture_context.csv",
            "plots/v0_5_run26_score_distribution_by_outcome.png",
            "plots/v0_5_run26_feature_contrast_fp_vs_fn.png",
            "run_26_v0_5_locked_error_analysis_notes.md",
        ],
    }
    (RUN26_OUTPUT_DIR / "v0_5_run26_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("Run 26 locked error analysis complete.")
    print(row_summary.round(4).to_string(index=False))
    print("")
    print(episode_capture.round(4).to_string(index=False))
    print(f"Outputs saved to: {RUN26_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

