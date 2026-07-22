"""
Run 24: v0.5 operating policy characterization.

This run does not train a model. It freezes the Run 23 candidate target
(`primary_or_uncertain`) and converts validation-set probabilities into
review-list operating policies: top-k rows, top-k episodes, risk thresholds,
first-alert-only policies, and cooldown policies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(r"C:\path\to\CVCML")
RUN23_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 23 (v0.5 Label Sensitivity Modeling)"
RUN24_OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Run 24 (v0.5 Operating Policy Characterization)"
PLOTS_DIR = RUN24_OUTPUT_DIR / "plots"

PREDICTION_FILE = RUN23_OUTPUT_DIR / "v0_5_run23_label_sensitivity_validation_predictions.csv"

TARGET_LABEL = "primary_or_uncertain"
SCORE_COL = "platt_probability"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def load_predictions() -> pd.DataFrame:
    print("Loading Run 23 validation predictions...")
    df = pd.read_csv(PREDICTION_FILE)
    df = df[df["target_label"].eq(TARGET_LABEL)].copy()
    if df.empty:
        raise ValueError(f"No validation predictions found for target_label={TARGET_LABEL!r}.")

    df["target"] = df["target"].astype(int)
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    if df[SCORE_COL].isna().any():
        raise ValueError(f"{SCORE_COL} contains missing/non-numeric values.")

    if "landmark_time" in df.columns:
        df["landmark_time"] = pd.to_datetime(df["landmark_time"], errors="coerce")
    if "landmark_hour" in df.columns:
        df["landmark_hour"] = pd.to_numeric(df["landmark_hour"], errors="coerce")

    df = df.sort_values([SCORE_COL, "episode_id", "landmark_hour"], ascending=[False, True, True])
    print(f"  Rows: {len(df):,}")
    print(f"  Positive rows: {df['target'].sum():,} ({df['target'].mean():.2%})")
    print(f"  Episodes: {df['episode_id'].nunique():,}")
    print(f"  Positive episodes: {df.groupby('episode_id')['target'].max().sum():,}")
    return df


def summarize_policy(
    all_rows: pd.DataFrame,
    flagged_rows: pd.DataFrame,
    policy_family: str,
    policy: str,
    *,
    policy_value: float | int | str | None = None,
    threshold: float | None = None,
    cooldown_hours: float | None = None,
) -> dict:
    flagged_rows = flagged_rows.copy()
    n_rows = len(all_rows)
    n_episodes = all_rows["episode_id"].nunique()

    total_positive_rows = int(all_rows["target"].sum())
    episode_positive = all_rows.groupby("episode_id")["target"].max()
    total_positive_episodes = int(episode_positive.sum())

    rows_reviewed = int(len(flagged_rows))
    episodes_reviewed = int(flagged_rows["episode_id"].nunique()) if rows_reviewed else 0
    tp_rows = int(flagged_rows["target"].sum()) if rows_reviewed else 0
    fp_rows = rows_reviewed - tp_rows

    captured_positive_episodes = (
        int(flagged_rows.loc[flagged_rows["target"].eq(1), "episode_id"].nunique())
        if rows_reviewed
        else 0
    )

    eventual_positive_episodes_reviewed = 0
    if rows_reviewed and total_positive_episodes:
        positive_episode_ids = set(episode_positive[episode_positive.eq(1)].index)
        eventual_positive_episodes_reviewed = len(
            set(flagged_rows["episode_id"].unique()).intersection(positive_episode_ids)
        )

    mean_reviews_per_reviewed_episode = safe_div(rows_reviewed, episodes_reviewed)
    max_reviews_per_episode = (
        int(flagged_rows.groupby("episode_id").size().max()) if rows_reviewed else 0
    )

    return {
        "target_label": TARGET_LABEL,
        "score_col": SCORE_COL,
        "policy_family": policy_family,
        "policy": policy,
        "policy_value": policy_value,
        "threshold": threshold,
        "cooldown_hours": cooldown_hours,
        "rows_total": n_rows,
        "episodes_total": n_episodes,
        "positive_rows_total": total_positive_rows,
        "positive_episodes_total": total_positive_episodes,
        "base_row_prevalence": safe_div(total_positive_rows, n_rows),
        "rows_reviewed": rows_reviewed,
        "reviewed_row_fraction": safe_div(rows_reviewed, n_rows),
        "reviews_per_100_landmark_rows": 100 * safe_div(rows_reviewed, n_rows),
        "episodes_reviewed": episodes_reviewed,
        "reviewed_episode_fraction": safe_div(episodes_reviewed, n_episodes),
        "reviews_per_100_episodes": 100 * safe_div(episodes_reviewed, n_episodes),
        "true_positive_rows": tp_rows,
        "false_positive_rows": fp_rows,
        "precision_ppv": safe_div(tp_rows, rows_reviewed),
        "ppv_lift_over_prevalence": safe_div(safe_div(tp_rows, rows_reviewed), safe_div(total_positive_rows, n_rows)),
        "row_recall_sensitivity": safe_div(tp_rows, total_positive_rows),
        "positive_episodes_captured": captured_positive_episodes,
        "episode_recall_sensitivity": safe_div(captured_positive_episodes, total_positive_episodes),
        "eventual_positive_episodes_reviewed": eventual_positive_episodes_reviewed,
        "eventual_positive_episode_review_fraction": safe_div(eventual_positive_episodes_reviewed, total_positive_episodes),
        "false_reviews_per_true_positive": safe_div(fp_rows, tp_rows),
        "repeat_review_rows": rows_reviewed - episodes_reviewed,
        "mean_reviews_per_reviewed_episode": mean_reviews_per_reviewed_episode,
        "max_reviews_per_episode": max_reviews_per_episode,
        "mean_review_score": flagged_rows[SCORE_COL].mean() if rows_reviewed else np.nan,
        "min_review_score": flagged_rows[SCORE_COL].min() if rows_reviewed else np.nan,
        "max_review_score": flagged_rows[SCORE_COL].max() if rows_reviewed else np.nan,
    }


def top_row_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    selected_rows = []
    sorted_df = df.sort_values(SCORE_COL, ascending=False)

    for pct in [0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]:
        n = max(1, int(math.ceil(len(sorted_df) * pct)))
        flagged = sorted_df.head(n).copy()
        policy = f"top_{pct:.1%}_rows".replace(".0%", "%")
        summaries.append(
            summarize_policy(df, flagged, "top_row_budget", policy, policy_value=pct)
        )
        selected_rows.append(flagged.assign(policy_family="top_row_budget", policy=policy))

    for n in [25, 50, 100, 150, 250, 500, 750, 1000]:
        n_eff = min(n, len(sorted_df))
        flagged = sorted_df.head(n_eff).copy()
        policy = f"top_{n_eff}_rows"
        summaries.append(
            summarize_policy(df, flagged, "top_row_budget", policy, policy_value=n_eff)
        )
        selected_rows.append(flagged.assign(policy_family="top_row_budget", policy=policy))

    return pd.DataFrame(summaries), pd.concat(selected_rows, ignore_index=True)


def top_episode_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    selected_rows = []
    best_row_idx = df.groupby("episode_id")[SCORE_COL].idxmax()
    episode_best = df.loc[best_row_idx].sort_values(SCORE_COL, ascending=False).copy()

    for pct in [0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]:
        n = max(1, int(math.ceil(len(episode_best) * pct)))
        flagged = episode_best.head(n).copy()
        policy = f"top_{pct:.1%}_episodes".replace(".0%", "%")
        summaries.append(
            summarize_policy(df, flagged, "top_episode_budget", policy, policy_value=pct)
        )
        selected_rows.append(flagged.assign(policy_family="top_episode_budget", policy=policy))

    for n in [25, 50, 100, 150, 250, 500, 750, 1000]:
        n_eff = min(n, len(episode_best))
        flagged = episode_best.head(n_eff).copy()
        policy = f"top_{n_eff}_episodes"
        summaries.append(
            summarize_policy(df, flagged, "top_episode_budget", policy, policy_value=n_eff)
        )
        selected_rows.append(flagged.assign(policy_family="top_episode_budget", policy=policy))

    return pd.DataFrame(summaries), pd.concat(selected_rows, ignore_index=True)


def threshold_policies(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    for threshold in thresholds:
        flagged = df[df[SCORE_COL].ge(threshold)].copy()
        summaries.append(
            summarize_policy(
                df,
                flagged,
                "threshold_all_landmarks",
                f"risk_ge_{threshold:.3f}",
                policy_value=threshold,
                threshold=threshold,
            )
        )
    return pd.DataFrame(summaries)


def first_alert_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    selected_rows = []
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
    sort_cols = ["episode_id", "landmark_hour", "landmark_time", SCORE_COL]
    available_sort_cols = [col for col in sort_cols if col in df.columns]

    for threshold in thresholds:
        over = df[df[SCORE_COL].ge(threshold)].sort_values(available_sort_cols).copy()
        flagged = over.groupby("episode_id", as_index=False).head(1).copy()
        policy = f"first_alert_risk_ge_{threshold:.3f}"
        summaries.append(
            summarize_policy(
                df,
                flagged,
                "first_alert_per_episode",
                policy,
                policy_value=threshold,
                threshold=threshold,
            )
        )
        selected_rows.append(flagged.assign(policy_family="first_alert_per_episode", policy=policy))

    return pd.DataFrame(summaries), pd.concat(selected_rows, ignore_index=True)


def cooldown_select(group: pd.DataFrame, cooldown_hours: float) -> pd.DataFrame:
    group = group.sort_values(["landmark_hour", "landmark_time", SCORE_COL])
    selected = []
    last_hour = -np.inf
    for idx, row in group.iterrows():
        hour = row.get("landmark_hour")
        if pd.isna(hour):
            selected.append(idx)
            continue
        if hour - last_hour >= cooldown_hours:
            selected.append(idx)
            last_hour = hour
    return group.loc[selected]


def cooldown_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    selected_rows = []
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]
    cooldowns = [48, 72, 168]

    for threshold in thresholds:
        over = df[df[SCORE_COL].ge(threshold)].copy()
        for cooldown in cooldowns:
            if over.empty:
                flagged = over.copy()
            else:
                pieces = [cooldown_select(g, cooldown) for _, g in over.groupby("episode_id")]
                flagged = pd.concat(pieces, ignore_index=False).copy() if pieces else over.iloc[0:0].copy()
            policy = f"risk_ge_{threshold:.3f}_cooldown_{cooldown}h"
            summaries.append(
                summarize_policy(
                    df,
                    flagged,
                    "threshold_with_cooldown",
                    policy,
                    policy_value=f"{threshold}|{cooldown}",
                    threshold=threshold,
                    cooldown_hours=cooldown,
                )
            )
            selected_rows.append(flagged.assign(policy_family="threshold_with_cooldown", policy=policy))

    return pd.DataFrame(summaries), pd.concat(selected_rows, ignore_index=True)


def build_recommended_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    usable = summary[
        summary["rows_reviewed"].gt(0)
        & summary["true_positive_rows"].gt(0)
        & summary["reviewed_row_fraction"].le(0.20)
    ].copy()

    candidates = []
    if usable.empty:
        return usable

    base = usable["base_row_prevalence"].iloc[0]

    low_burden = usable[usable["reviewed_row_fraction"].le(0.02)].sort_values(
        ["precision_ppv", "episode_recall_sensitivity"], ascending=False
    ).head(3)
    candidates.append(low_burden.assign(candidate_role="low_burden_high_ppv"))

    balanced = usable[
        usable["reviewed_row_fraction"].between(0.03, 0.08)
        & usable["precision_ppv"].ge(base * 1.5)
    ].sort_values(["episode_recall_sensitivity", "precision_ppv"], ascending=False).head(5)
    candidates.append(balanced.assign(candidate_role="balanced_review_list"))

    surveillance = usable[
        usable["reviewed_row_fraction"].between(0.08, 0.15)
        & usable["precision_ppv"].ge(base * 1.25)
    ].sort_values(["episode_recall_sensitivity", "false_reviews_per_true_positive"], ascending=[False, True]).head(5)
    candidates.append(surveillance.assign(candidate_role="higher_recall_surveillance"))

    episode_limited = usable[usable["policy_family"].eq("top_episode_budget")].sort_values(
        ["episode_recall_sensitivity", "precision_ppv"], ascending=False
    ).head(5)
    candidates.append(episode_limited.assign(candidate_role="episode_limited_review"))

    result = pd.concat(candidates, ignore_index=True)
    result = result.drop_duplicates(subset=["policy_family", "policy"])
    return result.sort_values(
        ["candidate_role", "reviewed_row_fraction", "precision_ppv"],
        ascending=[True, True, False],
    )


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    *,
    families: list[str],
    show_base_prevalence: bool = True,
    y_max_override: float | None = None,
) -> None:
    width, height = 1500, 950
    margin_left, margin_right = 150, 70
    margin_top, margin_bottom = 110, 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(44, bold=True)
    label_font = get_font(28)
    tick_font = get_font(23)
    legend_font = get_font(25)

    colors = {
        "top_row_budget": "#1f77b4",
        "top_episode_budget": "#ff7f0e",
        "threshold_all_landmarks": "#2ca02c",
        "first_alert_per_episode": "#d62728",
        "threshold_with_cooldown": "#9467bd",
    }

    plot_data = data[data["policy_family"].isin(families)].copy()
    plot_data = plot_data[np.isfinite(plot_data[x_col]) & np.isfinite(plot_data[y_col])]
    x_max = max(0.01, float(plot_data[x_col].max()) * 1.08)
    if y_max_override is None:
        y_max = max(0.01, float(plot_data[y_col].max()) * 1.15)
    else:
        y_max = y_max_override

    def to_xy(x: float, y: float) -> tuple[int, int]:
        px = margin_left + int((x / x_max) * plot_w)
        py = margin_top + plot_h - int((y / y_max) * plot_h)
        return px, py

    draw.text((margin_left, 30), title, fill="#111111", font=title_font)

    # Axes
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#222222", width=3)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="#222222", width=3)

    for i in range(6):
        x_val = x_max * i / 5
        x, y = to_xy(x_val, 0)
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 8), fill="#222222", width=2)
        draw.text((x - 30, margin_top + plot_h + 18), f"{x_val * 100:.0f}%", fill="#222222", font=tick_font)

        y_val = y_max * i / 5
        x0, y0 = to_xy(0, y_val)
        draw.line((margin_left - 8, y0, margin_left, y0), fill="#222222", width=2)
        draw.text((margin_left - 95, y0 - 12), f"{y_val * 100:.0f}%", fill="#222222", font=tick_font)
        if i > 0:
            draw.line((margin_left, y0, margin_left + plot_w, y0), fill="#eeeeee", width=1)

    draw.text((margin_left + plot_w // 2 - 170, height - 75), "Reviewed landmark-row fraction", fill="#111111", font=label_font)
    draw.text((30, margin_top + plot_h // 2 - 30), y_label, fill="#111111", font=label_font)

    if show_base_prevalence:
        base = float(data["base_row_prevalence"].dropna().iloc[0])
        _, base_y = to_xy(0, base)
        draw.line((margin_left, base_y, margin_left + plot_w, base_y), fill="#777777", width=3)
        draw.text((margin_left + plot_w - 250, base_y - 32), f"Base {base * 100:.1f}%", fill="#666666", font=legend_font)

    legend_x = margin_left + plot_w - 420
    legend_y = margin_top + 10
    for j, family in enumerate(families):
        family_data = plot_data[plot_data["policy_family"].eq(family)].sort_values(x_col)
        if family_data.empty:
            continue
        points = [to_xy(float(row[x_col]), float(row[y_col])) for _, row in family_data.iterrows()]
        color = colors.get(family, "#333333")
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
        draw.rectangle((legend_x, legend_y + j * 38, legend_x + 24, legend_y + 24 + j * 38), fill=color)
        label = family.replace("_", " ")
        draw.text((legend_x + 34, legend_y - 2 + j * 38), label, fill="#111111", font=legend_font)

    img.save(output_path)


def write_text_summary(summary: pd.DataFrame, candidates: pd.DataFrame, output_path: Path) -> None:
    base = summary["base_row_prevalence"].dropna().iloc[0]
    top5 = summary[(summary["policy_family"].eq("top_row_budget")) & (summary["policy"].eq("top_5%_rows"))]
    top10 = summary[(summary["policy_family"].eq("top_row_budget")) & (summary["policy"].eq("top_10%_rows"))]

    lines = [
        "# Run 24 - v0.5 Operating Policy Characterization",
        "",
        "## Purpose",
        "",
        "Run 24 freezes the Run 23 candidate label (`primary_or_uncertain`) and evaluates how the calibrated validation risk scores behave as operational review policies. It does not retrain the model or inspect the temporal lockbox.",
        "",
        "## Key Validation Context",
        "",
        f"- Landmark rows evaluated: {int(summary['rows_total'].iloc[0]):,}",
        f"- Positive landmark rows: {int(summary['positive_rows_total'].iloc[0]):,}",
        f"- Base row prevalence: {format_percent(base)}",
        f"- Positive episodes: {int(summary['positive_episodes_total'].iloc[0]):,}",
    ]

    if not top5.empty:
        row = top5.iloc[0]
        lines.extend([
            "",
            "## Top 5% Row Review Policy",
            "",
            f"- Reviews: {int(row['rows_reviewed']):,} landmark rows ({format_percent(row['reviewed_row_fraction'])})",
            f"- PPV: {format_percent(row['precision_ppv'])} ({row['ppv_lift_over_prevalence']:.2f}x base prevalence)",
            f"- Row recall: {format_percent(row['row_recall_sensitivity'])}",
            f"- Episode recall: {format_percent(row['episode_recall_sensitivity'])}",
            f"- False reviews per true-positive row: {row['false_reviews_per_true_positive']:.2f}",
        ])

    if not top10.empty:
        row = top10.iloc[0]
        lines.extend([
            "",
            "## Top 10% Row Review Policy",
            "",
            f"- Reviews: {int(row['rows_reviewed']):,} landmark rows ({format_percent(row['reviewed_row_fraction'])})",
            f"- PPV: {format_percent(row['precision_ppv'])} ({row['ppv_lift_over_prevalence']:.2f}x base prevalence)",
            f"- Row recall: {format_percent(row['row_recall_sensitivity'])}",
            f"- Episode recall: {format_percent(row['episode_recall_sensitivity'])}",
            f"- False reviews per true-positive row: {row['false_reviews_per_true_positive']:.2f}",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The model is best framed as a prioritized review-list or infection-prevention surveillance aid, not a bedside interruptive alarm.",
        "- Top-row budgets provide the clearest operating point because they directly control review burden.",
        "- Episode-limited policies reduce repeated reviews, but they can miss useful repeated risk signals when a patient's status evolves over time.",
        "- Threshold policies are less portable at this stage because calibration remains development-set dependent; use review budgets until calibration is externally tested.",
        "",
        "## Recommended Candidate Policies",
        "",
    ])

    if candidates.empty:
        lines.append("No candidate policies met the default screening criteria.")
    else:
        keep_cols = [
            "candidate_role",
            "policy_family",
            "policy",
            "rows_reviewed",
            "reviewed_row_fraction",
            "precision_ppv",
            "episode_recall_sensitivity",
            "false_reviews_per_true_positive",
        ]
        for _, row in candidates[keep_cols].head(12).iterrows():
            lines.append(
                f"- {row['candidate_role']}: {row['policy_family']} / {row['policy']} | "
                f"reviews={int(row['rows_reviewed'])}, PPV={format_percent(row['precision_ppv'])}, "
                f"episode recall={format_percent(row['episode_recall_sensitivity'])}, "
                f"false reviews/TP={row['false_reviews_per_true_positive']:.2f}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RUN24_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions()

    print("")
    print("Building review policies...")
    top_rows, selected_top_rows = top_row_policies(df)
    top_episodes, selected_top_episodes = top_episode_policies(df)
    thresholds = threshold_policies(df)
    first_alerts, selected_first_alerts = first_alert_policies(df)
    cooldowns, selected_cooldowns = cooldown_policies(df)

    summary = pd.concat(
        [top_rows, top_episodes, thresholds, first_alerts, cooldowns],
        ignore_index=True,
    )
    candidates = build_recommended_candidates(summary)

    selected_review_rows = pd.concat(
        [selected_top_rows, selected_top_episodes, selected_first_alerts, selected_cooldowns],
        ignore_index=True,
    )

    print("Saving tables...")
    top_rows.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_topk_row_policy.csv", index=False)
    top_episodes.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_topk_episode_policy.csv", index=False)
    thresholds.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_threshold_policy.csv", index=False)
    first_alerts.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_first_alert_policy.csv", index=False)
    cooldowns.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_cooldown_policy.csv", index=False)
    summary.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_operating_policy_summary.csv", index=False)
    candidates.to_csv(RUN24_OUTPUT_DIR / "v0_5_run24_recommended_policy_candidates.csv", index=False)

    selected_cols = [
        "policy_family",
        "policy",
        "landmark_id",
        "episode_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "anchor_year_group",
        "landmark_hour",
        "landmark_time",
        "target",
        SCORE_COL,
    ]
    selected_cols = [col for col in selected_cols if col in selected_review_rows.columns]
    selected_review_rows[selected_cols].to_csv(
        RUN24_OUTPUT_DIR / "v0_5_run24_selected_review_rows.csv",
        index=False,
    )

    print("Drawing plots...")
    primary_families = ["top_row_budget", "top_episode_budget", "first_alert_per_episode", "threshold_with_cooldown"]
    draw_line_plot(
        summary,
        "reviewed_row_fraction",
        "precision_ppv",
        "Run 24 PPV vs Review Burden",
        "PPV",
        PLOTS_DIR / "v0_5_run24_ppv_vs_review_burden.png",
        families=primary_families,
    )
    draw_line_plot(
        summary,
        "reviewed_row_fraction",
        "episode_recall_sensitivity",
        "Run 24 Episode Recall vs Review Burden",
        "Episode recall",
        PLOTS_DIR / "v0_5_run24_episode_recall_vs_review_burden.png",
        families=primary_families,
        show_base_prevalence=False,
        y_max_override=1.0,
    )

    write_text_summary(
        summary,
        candidates,
        RUN24_OUTPUT_DIR / "run_24_v0_5_operating_policy_characterization_notes.md",
    )

    manifest = {
        "run": "Run 24 (v0.5 Operating Policy Characterization)",
        "created_from": str(PREDICTION_FILE),
        "target_label": TARGET_LABEL,
        "score_col": SCORE_COL,
        "rows": int(len(df)),
        "positive_rows": int(df["target"].sum()),
        "base_row_prevalence": float(df["target"].mean()),
        "outputs": [
            "v0_5_run24_operating_policy_summary.csv",
            "v0_5_run24_topk_row_policy.csv",
            "v0_5_run24_topk_episode_policy.csv",
            "v0_5_run24_threshold_policy.csv",
            "v0_5_run24_first_alert_policy.csv",
            "v0_5_run24_cooldown_policy.csv",
            "v0_5_run24_recommended_policy_candidates.csv",
            "v0_5_run24_selected_review_rows.csv",
            "plots/v0_5_run24_ppv_vs_review_burden.png",
            "plots/v0_5_run24_episode_recall_vs_review_burden.png",
            "run_24_v0_5_operating_policy_characterization_notes.md",
        ],
    }
    (RUN24_OUTPUT_DIR / "v0_5_run24_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("")
    print("Run 24 complete.")
    print(f"Outputs saved to: {RUN24_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

