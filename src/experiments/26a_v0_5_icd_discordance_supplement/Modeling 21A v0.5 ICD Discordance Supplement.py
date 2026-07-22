from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(r"C:\path\to\CVCML")
RUN27_DIR = PROJECT_DIR / "Outputs" / "Run 27 (v0.5 ICD Agreement Label Validation)"
OUT_DIR = PROJECT_DIR / "Outputs" / "Run 27.1 (v0.5 ICD Discordance Supplement)"
PLOT_DIR = OUT_DIR / "plots"

EPISODE_AGREEMENT = RUN27_DIR / "v0_5_run27_episode_icd_agreement.csv"

PROXY_COL = "cvc_bsi_strict_primary_or_uncertain_proxy"
ICD_COL = "icd_cvc_bsi_specific"
RANDOM_SEED = 2028


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def semicolon_join(values: pd.Series, max_items: int = 20) -> str:
    cleaned = []
    for value in values:
        text = safe_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return "; ".join(cleaned[:max_items])


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = safe_text(value)
        if text:
            return text
    return ""


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    datetime_cols = [
        "exposure_start",
        "exposure_end_observed",
        "broad_proxy_culture_time",
        "strict_proxy_culture_time",
        "strict_primary_likely_culture_time",
        "strict_primary_or_uncertain_culture_time",
        "strict_secondary_possible_culture_time",
    ]
    for col in datetime_cols:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")

    work["observed_exposure_hours"] = (
        work["exposure_end_observed"] - work["exposure_start"]
    ).dt.total_seconds() / 3600.0
    work["strict_culture_hours_from_line_start"] = (
        work["strict_proxy_culture_time"] - work["exposure_start"]
    ).dt.total_seconds() / 3600.0
    work["target_culture_hours_from_line_start"] = (
        work["strict_primary_or_uncertain_culture_time"] - work["exposure_start"]
    ).dt.total_seconds() / 3600.0
    return work


def assign_discordance_groups(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    proxy = safe_int(work[PROXY_COL])
    icd = safe_int(work[ICD_COL])
    work["proxy_icd_group"] = np.select(
        [
            (proxy == 1) & (icd == 1),
            (proxy == 1) & (icd == 0),
            (proxy == 0) & (icd == 1),
        ],
        ["proxy_pos_icd_pos", "proxy_pos_icd_neg", "proxy_neg_icd_pos"],
        default="proxy_neg_icd_neg",
    )
    return work


def proxy_positive_icd_negative_phenotype(row: pd.Series) -> str:
    source_class = safe_text(row.get("source_screen_class"))
    if safe_int(pd.Series([row.get("cvc_bsi_strict_primary_likely_proxy", 0)])).iloc[0] == 1:
        return "structured_primary_likely_not_icd_coded"
    if "uncertain_nonconcordant_source_culture" in source_class:
        return "structured_uncertain_nonconcordant_source_not_icd_coded"
    if "uncertain_icd_only" in source_class:
        return "structured_uncertain_source_icd_not_cvc_icd_coded"
    return "structured_proxy_not_icd_coded_other"


def icd_positive_proxy_negative_phenotype(row: pd.Series) -> str:
    if int(row.get("cvc_bsi_strict_secondary_possible_proxy", 0) or 0) == 1:
        return "strict_event_excluded_as_secondary_possible"
    if int(row.get("cvc_bsi_broad_proxy", 0) or 0) == 1 and int(
        row.get("cvc_bsi_strict_proxy", 0) or 0
    ) == 0:
        return "broad_blood_culture_failed_strict_organism_logic"
    if int(row.get("early_positive_culture", 0) or 0) == 1:
        return "positive_culture_before_48h_eligibility"
    if int(row.get("eligible_48h_observed_exposure", 0) or 0) == 0:
        return "no_observed_48h_eligible_cvc_exposure"
    return "no_structured_qualifying_blood_culture_during_episode"


def assign_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["discordance_phenotype"] = "not_discordant"

    proxy_only = work["proxy_icd_group"].eq("proxy_pos_icd_neg")
    icd_only = work["proxy_icd_group"].eq("proxy_neg_icd_pos")
    work.loc[proxy_only, "discordance_phenotype"] = work.loc[proxy_only].apply(
        proxy_positive_icd_negative_phenotype, axis=1
    )
    work.loc[icd_only, "discordance_phenotype"] = work.loc[icd_only].apply(
        icd_positive_proxy_negative_phenotype, axis=1
    )

    priority = {
        "structured_primary_likely_not_icd_coded": 1,
        "strict_event_excluded_as_secondary_possible": 1,
        "broad_blood_culture_failed_strict_organism_logic": 2,
        "structured_uncertain_nonconcordant_source_not_icd_coded": 2,
        "positive_culture_before_48h_eligibility": 3,
        "structured_uncertain_source_icd_not_cvc_icd_coded": 3,
        "no_observed_48h_eligible_cvc_exposure": 4,
        "no_structured_qualifying_blood_culture_during_episode": 5,
        "structured_proxy_not_icd_coded_other": 5,
        "not_discordant": 9,
    }
    work["manual_review_priority"] = work["discordance_phenotype"].map(priority).fillna(9).astype(int)
    return work


def add_admission_attribution_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    admission = (
        work.groupby("hadm_id", dropna=False)
        .agg(
            admission_episode_count=("episode_id", "nunique"),
            admission_proxy_positive_episode_count=(PROXY_COL, "sum"),
            admission_strict_positive_episode_count=("cvc_bsi_strict_proxy", "sum"),
            admission_broad_positive_episode_count=("cvc_bsi_broad_proxy", "sum"),
            admission_icd_specific=(ICD_COL, "max"),
        )
        .reset_index()
    )
    admission["icd_episode_attribution_ambiguous"] = (
        (safe_int(admission["admission_icd_specific"]) == 1)
        & (admission["admission_episode_count"] > 1)
    ).astype(int)
    return work.merge(admission, on="hadm_id", how="left")


def binary_agreement(df: pd.DataFrame, proxy_col: str, icd_col: str, grain: str) -> dict[str, object]:
    proxy = safe_int(df[proxy_col])
    icd = safe_int(df[icd_col])
    both = int(((proxy == 1) & (icd == 1)).sum())
    proxy_only = int(((proxy == 1) & (icd == 0)).sum())
    icd_only = int(((proxy == 0) & (icd == 1)).sum())
    neither = int(((proxy == 0) & (icd == 0)).sum())
    proxy_pos = both + proxy_only
    icd_pos = both + icd_only
    union = both + proxy_only + icd_only
    return {
        "grain": grain,
        "n_units": len(df),
        "proxy_positive": proxy_pos,
        "icd_positive": icd_pos,
        "both_positive": both,
        "proxy_positive_icd_negative": proxy_only,
        "proxy_negative_icd_positive": icd_only,
        "both_negative": neither,
        "positive_set_jaccard": both / union if union else np.nan,
        "proxy_ppv_vs_icd": both / proxy_pos if proxy_pos else np.nan,
        "proxy_sensitivity_vs_icd": both / icd_pos if icd_pos else np.nan,
    }


def admission_level_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, object]] = {
        "subject_id": ("subject_id", "first"),
        "anchor_year_group": ("anchor_year_group", first_nonempty),
        "episode_count": ("episode_id", "nunique"),
        "episode_ids": ("episode_id", lambda x: "; ".join(x.astype(str).unique())),
        "proxy_positive": (PROXY_COL, "max"),
        "proxy_positive_episode_count": (PROXY_COL, "sum"),
        "strict_proxy_positive": ("cvc_bsi_strict_proxy", "max"),
        "broad_proxy_positive": ("cvc_bsi_broad_proxy", "max"),
        "early_positive_culture": ("early_positive_culture", "max"),
        "eligible_48h_any_episode": ("eligible_48h_observed_exposure", "max"),
        "icd_cvc_bsi_specific": (ICD_COL, "max"),
        "icd_cvc_infection_any": ("icd_cvc_infection_any", "max"),
        "icd_broad": ("icd_cvc_or_vascular_infection_broad", "max"),
        "source_screen_classes": ("source_screen_class", semicolon_join),
        "strict_proxy_positive_orgs": ("strict_proxy_positive_orgs", semicolon_join),
        "icd_specific_codes": ("icd_cvc_bsi_specific_codes", semicolon_join),
        "icd_specific_titles": ("icd_cvc_bsi_specific_titles", semicolon_join),
    }
    admission = df.groupby("hadm_id", dropna=False).agg(**aggregations).reset_index()
    admission["proxy_positive"] = safe_int(admission["proxy_positive"])
    admission["icd_cvc_bsi_specific"] = safe_int(admission["icd_cvc_bsi_specific"])
    admission["proxy_icd_group"] = np.select(
        [
            admission["proxy_positive"].eq(1) & admission["icd_cvc_bsi_specific"].eq(1),
            admission["proxy_positive"].eq(1) & admission["icd_cvc_bsi_specific"].eq(0),
            admission["proxy_positive"].eq(0) & admission["icd_cvc_bsi_specific"].eq(1),
        ],
        ["proxy_pos_icd_pos", "proxy_pos_icd_neg", "proxy_neg_icd_pos"],
        default="proxy_neg_icd_neg",
    )
    admission["icd_episode_attribution_ambiguous"] = (
        admission["icd_cvc_bsi_specific"].eq(1) & admission["episode_count"].gt(1)
    ).astype(int)
    return admission


def summarize_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    discordant = df[df["proxy_icd_group"].isin(["proxy_pos_icd_neg", "proxy_neg_icd_pos"])].copy()
    out = (
        discordant.groupby(["proxy_icd_group", "discordance_phenotype"], dropna=False)
        .agg(
            n_episodes=("episode_id", "count"),
            n_admissions=("hadm_id", "nunique"),
            n_patients=("subject_id", "nunique"),
            eligible_48h_rate=("eligible_48h_observed_exposure", "mean"),
            multi_episode_admission_rate=("icd_episode_attribution_ambiguous", "mean"),
            median_observed_exposure_hours=("observed_exposure_hours", "median"),
            median_nearby_nonblood_source_cultures=("nearby_nonblood_source_culture_count", "median"),
            median_hadm_source_icd_count=("hadm_source_icd_count", "median"),
            lockbox_episodes=("anchor_year_group", lambda x: int(x.eq("2020 - 2022").sum())),
        )
        .reset_index()
    )
    totals = out.groupby("proxy_icd_group")["n_episodes"].transform("sum")
    out["share_within_discordance_direction"] = out["n_episodes"] / totals
    return out.sort_values(["proxy_icd_group", "n_episodes"], ascending=[True, False])


def summarize_by_period(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, group in df.groupby("anchor_year_group", dropna=False):
        n = len(group)
        for discordance_group in ["proxy_pos_icd_neg", "proxy_neg_icd_pos", "proxy_pos_icd_pos"]:
            count = int(group["proxy_icd_group"].eq(discordance_group).sum())
            rows.append(
                {
                    "anchor_year_group": period,
                    "proxy_icd_group": discordance_group,
                    "n_episodes": n,
                    "group_count": count,
                    "group_rate": count / n if n else np.nan,
                    "group_per_1000_episodes": 1000 * count / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_admission_ambiguity(admission: pd.DataFrame) -> pd.DataFrame:
    icd = admission[admission["icd_cvc_bsi_specific"].eq(1)].copy()
    if len(icd) == 0:
        return pd.DataFrame()
    icd["episode_count_band"] = pd.cut(
        icd["episode_count"],
        bins=[0, 1, 2, np.inf],
        labels=["1 episode", "2 episodes", "3+ episodes"],
    )
    out = (
        icd.groupby("episode_count_band", observed=True)
        .agg(
            icd_positive_admissions=("hadm_id", "count"),
            represented_episodes=("episode_count", "sum"),
            admissions_with_proxy_positive=("proxy_positive", "sum"),
            admissions_with_broad_proxy_positive=("broad_proxy_positive", "sum"),
        )
        .reset_index()
    )
    out["proxy_overlap_rate"] = out["admissions_with_proxy_positive"] / out["icd_positive_admissions"]
    return out


REVIEW_COLUMNS = [
    "episode_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "anchor_year_group",
    "proxy_icd_group",
    "discordance_phenotype",
    "manual_review_priority",
    "source_screen_class",
    "exposure_start",
    "exposure_end_observed",
    "observed_exposure_hours",
    "eligible_48h_observed_exposure",
    "cvc_bsi_broad_proxy",
    "broad_proxy_culture_time",
    "cvc_bsi_strict_proxy",
    "strict_proxy_culture_time",
    "strict_proxy_positive_orgs",
    "strict_proxy_label_reason",
    "early_positive_culture",
    "nearby_nonblood_source_culture_count",
    "concordant_nonblood_source_culture_count",
    "nearby_nonblood_source_buckets",
    "nearby_nonblood_source_specimens",
    "nearby_nonblood_source_orgs",
    "hadm_source_icd_count",
    "hadm_source_icd_buckets",
    "hadm_source_icd_titles",
    "icd_cvc_bsi_specific_codes",
    "icd_cvc_bsi_specific_titles",
    "admission_episode_count",
    "admission_proxy_positive_episode_count",
    "icd_episode_attribution_ambiguous",
    "max_score",
    "mean_score",
    "top_150_episodes",
    "capture_group_top_10_rows",
]


def build_review_queues(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = [col for col in REVIEW_COLUMNS if col in df.columns]
    sort_cols = ["manual_review_priority", "anchor_year_group"]
    ascending = [True, False]
    if "max_score" in df.columns:
        df = df.copy()
        df["max_score_sort"] = pd.to_numeric(df["max_score"], errors="coerce").fillna(-1)
        sort_cols.append("max_score_sort")
        ascending.append(False)
    sort_cols.append("episode_id")
    ascending.append(True)

    proxy_only = (
        df[df["proxy_icd_group"].eq("proxy_pos_icd_neg")]
        .sort_values(sort_cols, ascending=ascending)
        .loc[:, available]
    )
    icd_only = (
        df[df["proxy_icd_group"].eq("proxy_neg_icd_pos")]
        .sort_values(sort_cols, ascending=ascending)
        .loc[:, available]
    )

    lockbox = pd.concat(
        [
            proxy_only[proxy_only["anchor_year_group"].eq("2020 - 2022")],
            icd_only[icd_only["anchor_year_group"].eq("2020 - 2022")],
        ],
        ignore_index=True,
    )

    review_sample_parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for direction_df in [proxy_only, icd_only]:
        target_n = min(30, len(direction_df))
        if target_n == 0:
            continue
        shuffled = direction_df.copy()
        shuffled["_random_order"] = rng.random(len(shuffled))

        # One case per phenotype-period cell first, then fill to the target.
        first_pass = (
            shuffled.sort_values("_random_order")
            .groupby(["discordance_phenotype", "anchor_year_group"], dropna=False)
            .head(1)
        )
        if len(first_pass) > target_n:
            first_pass = first_pass.sort_values(
                ["manual_review_priority", "_random_order"]
            ).head(target_n)
        remaining_n = target_n - len(first_pass)
        remaining = shuffled[~shuffled["episode_id"].isin(first_pass["episode_id"])]
        if remaining_n > 0:
            fill = remaining.sort_values(
                ["manual_review_priority", "_random_order"]
            ).head(remaining_n)
            selected = pd.concat([first_pass, fill], ignore_index=True)
        else:
            selected = first_pass.copy()
        selected["review_sample_seed"] = RANDOM_SEED
        review_sample_parts.append(selected.drop(columns=["_random_order"], errors="ignore"))

    review_sample = pd.concat(review_sample_parts, ignore_index=True) if review_sample_parts else pd.DataFrame()
    return proxy_only, icd_only, review_sample, lockbox


def get_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def display_label(text: str) -> str:
    labels = {
        "structured_primary_likely_not_icd_coded": "Primary-likely proxy, no CVC-BSI ICD code",
        "structured_uncertain_nonconcordant_source_not_icd_coded": "Uncertain: nonconcordant source culture, no CVC-BSI ICD code",
        "structured_uncertain_source_icd_not_cvc_icd_coded": "Uncertain: source ICD evidence, no CVC-BSI ICD code",
        "structured_proxy_not_icd_coded_other": "Other proxy-positive / ICD-negative",
        "strict_event_excluded_as_secondary_possible": "Strict event excluded as possible secondary BSI",
        "broad_blood_culture_failed_strict_organism_logic": "Broad culture failed strict organism logic",
        "positive_culture_before_48h_eligibility": "Positive culture before 48h eligibility",
        "no_observed_48h_eligible_cvc_exposure": "No observed 48h eligible CVC exposure",
        "no_structured_qualifying_blood_culture_during_episode": "No structured qualifying blood culture during episode",
    }
    return labels.get(text, text.replace("_", " "))


def plot_phenotypes(summary: pd.DataFrame) -> None:
    if len(summary) == 0:
        return
    plot_df = summary.sort_values(["proxy_icd_group", "n_episodes"], ascending=[True, True]).copy()
    width = 1800
    row_h = 74
    height = 190 + row_h * len(plot_df) + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42, True)
    label_font = get_font(23)
    value_font = get_font(24, True)
    note_font = get_font(19)
    draw.text((65, 40), "Run 27.1 Structured Reasons for Proxy/ICD Discordance", font=title_font, fill=(25, 25, 25))
    draw.text(
        (65, 100),
        "ICD diagnoses are admission-level; phenotype counts below are catheter-episode-level.",
        font=note_font,
        fill=(90, 90, 90),
    )

    max_n = max(int(plot_df["n_episodes"].max()), 1)
    x0 = 850
    bar_w = 700
    colors = {"proxy_neg_icd_pos": (226, 124, 66), "proxy_pos_icd_neg": (70, 135, 205)}
    y = 170
    previous_group = None
    for row in plot_df.itertuples(index=False):
        if previous_group is not None and row.proxy_icd_group != previous_group:
            draw.line((65, y - 12, width - 65, y - 12), fill=(210, 210, 210), width=2)
        label = display_label(str(row.discordance_phenotype))
        draw.text((65, y + 12), label, font=label_font, fill=(40, 40, 40))
        length = int(bar_w * int(row.n_episodes) / max_n)
        color = colors.get(str(row.proxy_icd_group), (120, 120, 120))
        draw.rounded_rectangle((x0, y + 9, x0 + max(length, 4), y + 48), radius=6, fill=color)
        draw.text((x0 + length + 18, y + 12), f"{int(row.n_episodes):,}", font=value_font, fill=(45, 45, 45))
        y += row_h
        previous_group = row.proxy_icd_group

    draw.rectangle((65, height - 62, 95, height - 36), fill=colors["proxy_pos_icd_neg"])
    draw.text((108, height - 67), "Proxy-positive / ICD-negative", font=note_font, fill=(60, 60, 60))
    draw.rectangle((470, height - 62, 500, height - 36), fill=colors["proxy_neg_icd_pos"])
    draw.text((513, height - 67), "Proxy-negative / ICD-positive", font=note_font, fill=(60, 60, 60))
    img.save(PLOT_DIR / "v0_5_run27_1_discordance_phenotypes.png")


def plot_temporal_discordance(period_summary: pd.DataFrame) -> None:
    if len(period_summary) == 0:
        return
    periods = sorted(period_summary["anchor_year_group"].dropna().astype(str).unique())
    directions = ["proxy_pos_icd_neg", "proxy_neg_icd_pos"]
    colors = {"proxy_pos_icd_neg": (70, 135, 205), "proxy_neg_icd_pos": (226, 124, 66)}
    labels = {"proxy_pos_icd_neg": "Proxy+/ICD-", "proxy_neg_icd_pos": "Proxy-/ICD+"}

    width, height = 1500, 880
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42, True)
    axis_font = get_font(22)
    value_font = get_font(18)
    draw.text((65, 35), "Run 27.1 Discordance Across MIMIC Eras", font=title_font, fill=(25, 25, 25))
    draw.text((65, 92), "Rate per 1,000 reconstructed catheter episodes", font=axis_font, fill=(80, 80, 80))

    x0, y0 = 130, 690
    chart_w, chart_h = 1250, 500
    max_value = period_summary[period_summary["proxy_icd_group"].isin(directions)][
        "group_per_1000_episodes"
    ].max()
    max_value = max(float(max_value), 1.0)

    for tick in range(6):
        value = max_value * tick / 5
        y = y0 - chart_h * tick / 5
        draw.line((x0, y, x0 + chart_w, y), fill=(225, 225, 225), width=1)
        draw.text((30, y - 12), f"{value:.0f}", font=value_font, fill=(80, 80, 80))

    group_w = chart_w / max(len(periods), 1)
    bar_w = min(58, group_w * 0.28)
    for i, period in enumerate(periods):
        center = x0 + group_w * (i + 0.5)
        draw.text((center - 55, y0 + 25), period, font=value_font, fill=(50, 50, 50))
        for j, direction in enumerate(directions):
            row = period_summary[
                period_summary["anchor_year_group"].astype(str).eq(period)
                & period_summary["proxy_icd_group"].eq(direction)
            ]
            value = float(row["group_per_1000_episodes"].iloc[0]) if len(row) else 0.0
            bar_h = chart_h * value / max_value
            left = center + (j - 1) * bar_w
            draw.rectangle((left, y0 - bar_h, left + bar_w - 6, y0), fill=colors[direction])
            draw.text((left - 2, y0 - bar_h - 25), f"{value:.1f}", font=value_font, fill=(55, 55, 55))

    legend_y = 800
    for i, direction in enumerate(directions):
        left = 480 + i * 300
        draw.rectangle((left, legend_y, left + 30, legend_y + 24), fill=colors[direction])
        draw.text((left + 42, legend_y - 3), labels[direction], font=axis_font, fill=(50, 50, 50))
    img.save(PLOT_DIR / "v0_5_run27_1_discordance_by_era.png")


def write_notes(
    episode_agreement: pd.DataFrame,
    admission_agreement: pd.DataFrame,
    phenotype_summary: pd.DataFrame,
    ambiguity: pd.DataFrame,
    review_sample: pd.DataFrame,
) -> None:
    episode_row = episode_agreement.iloc[0]
    admission_row = admission_agreement.iloc[0]
    icd_multi = ambiguity[ambiguity["episode_count_band"].isin(["2 episodes", "3+ episodes"])]
    multi_admissions = int(icd_multi["icd_positive_admissions"].sum()) if len(icd_multi) else 0
    total_icd_admissions = int(ambiguity["icd_positive_admissions"].sum()) if len(ambiguity) else 0

    phenotype_lines = []
    for row in phenotype_summary.itertuples(index=False):
        phenotype_lines.append(
            f"- {display_label(str(row.discordance_phenotype))}: {int(row.n_episodes):,} episodes "
            f"({row.share_within_discordance_direction:.1%} within {row.proxy_icd_group})."
        )

    lines = [
        "# Run 27.1 - Structured Discordance Phenotyping Supplement",
        "",
        "## Purpose",
        "",
        "Run 27.1 supplements Run 27 by reconciling the grain mismatch, assigning transparent structured reasons to proxy/ICD discordance, and creating deterministic review queues for later chart validation. It does not refit the model, change thresholds, or redefine the primary outcome.",
        "",
        "## Grain Reconciliation",
        "",
        f"- Episode-level comparison: {int(episode_row['n_units']):,} episodes, {int(episode_row['both_positive']):,} both-positive, positive-set Jaccard {episode_row['positive_set_jaccard']:.1%}.",
        f"- Admission-level comparison: {int(admission_row['n_units']):,} admissions, {int(admission_row['both_positive']):,} both-positive, positive-set Jaccard {admission_row['positive_set_jaccard']:.1%}.",
        f"- ICD-specific CVC-BSI codes appeared in {total_icd_admissions:,} admissions; {multi_admissions:,} ({multi_admissions / total_icd_admissions:.1%}) contained multiple reconstructed catheter episodes." if total_icd_admissions else "- No ICD-specific CVC-BSI admissions were found.",
        "- Because diagnoses_icd is admission-level and assigned at discharge, it cannot identify which catheter episode generated the code. Admission-level agreement is therefore the fairer external agreement analysis.",
        "",
        "## Discordance Phenotypes",
        "",
        *phenotype_lines,
        "",
        "## Interpretation",
        "",
        "The dominant ICD-positive/proxy-negative phenotype is absence of a structured qualifying blood culture during the reconstructed episode. This does not establish that the ICD code is wrong: possible explanations include incomplete procedureevents line documentation, an infection tied to a different catheter episode in the same admission, culture timing outside the proxy window, or administrative coding that cannot be temporally localized.",
        "",
        "Proxy-positive/ICD-negative episodes are mostly source-screened uncertain events rather than the small primary-likely subset. They remain appropriate for sensitivity analysis and targeted review, but they should not be described as administratively confirmed CLABSI.",
        "",
        "## Manual Review Frame",
        "",
        f"- Deterministic balanced sample: {len(review_sample):,} episodes, generated with seed {RANDOM_SEED}.",
        "- The sample spans both discordance directions, structured phenotypes, and MIMIC eras where cases are available.",
        "- Structured data alone cannot adjudicate NHSN CLABSI. A future review should examine clinical notes and source attribution while preserving the existing proxy and ICD labels as separate evidence streams.",
        "",
        "## Sources and Rationale",
        "",
        "- CDC NHSN January 2026 BSI/CLABSI manual: https://www.cdc.gov/nhsn/pdfs/pscmanual/4psc_clabscurrent.pdf",
        "- MIMIC diagnoses_icd documentation: https://mimic.mit.edu/docs/IV/modules/hosp/diagnoses_icd.html",
        "- MIMIC procedureevents documentation: https://mimic.mit.edu/docs/IV/modules/icu/procedureevents.html",
        "",
        "## Key Output Files",
        "",
        "- `v0_5_run27_1_grain_reconciled_agreement.csv`",
        "- `v0_5_run27_1_admission_level_icd_agreement.csv`",
        "- `v0_5_run27_1_icd_episode_attribution_ambiguity.csv`",
        "- `v0_5_run27_1_discordance_phenotype_summary.csv`",
        "- `v0_5_run27_1_discordance_by_era.csv`",
        "- `v0_5_run27_1_proxy_positive_icd_negative_review_queue.csv`",
        "- `v0_5_run27_1_icd_positive_proxy_negative_review_queue.csv`",
        "- `v0_5_run27_1_balanced_manual_review_sample.csv`",
        "- `v0_5_run27_1_lockbox_discordance_review_queue.csv`",
        "- `plots/v0_5_run27_1_discordance_phenotypes.png`",
        "- `plots/v0_5_run27_1_discordance_by_era.png`",
    ]
    (OUT_DIR / "run_27_1_v0_5_discordance_phenotyping_notes.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    print("Run 27.1: structured proxy/ICD discordance phenotyping supplement")
    print("Loading Run 27 episode agreement table...")
    episodes = pd.read_csv(EPISODE_AGREEMENT)
    print(f"  Episodes: {len(episodes):,}")
    print(f"  Admissions: {episodes['hadm_id'].nunique():,}")

    binary_cols = [
        PROXY_COL,
        ICD_COL,
        "cvc_bsi_broad_proxy",
        "cvc_bsi_strict_proxy",
        "cvc_bsi_strict_primary_likely_proxy",
        "cvc_bsi_strict_secondary_possible_proxy",
        "early_positive_culture",
        "eligible_48h_observed_exposure",
        "icd_cvc_infection_any",
        "icd_cvc_or_vascular_infection_broad",
    ]
    for col in binary_cols:
        if col in episodes.columns:
            episodes[col] = safe_int(episodes[col])

    episodes = parse_datetimes(episodes)
    episodes = assign_discordance_groups(episodes)
    episodes = assign_phenotypes(episodes)
    episodes = add_admission_attribution_fields(episodes)

    admission = admission_level_table(episodes)
    admission.to_csv(OUT_DIR / "v0_5_run27_1_admission_level_icd_agreement.csv", index=False)

    episode_agreement = pd.DataFrame([binary_agreement(episodes, PROXY_COL, ICD_COL, "catheter_episode")])
    admission_agreement = pd.DataFrame(
        [binary_agreement(admission, "proxy_positive", "icd_cvc_bsi_specific", "hospital_admission")]
    )
    grain_agreement = pd.concat([episode_agreement, admission_agreement], ignore_index=True)
    grain_agreement.to_csv(OUT_DIR / "v0_5_run27_1_grain_reconciled_agreement.csv", index=False)

    phenotype_summary = summarize_phenotypes(episodes)
    phenotype_summary.to_csv(OUT_DIR / "v0_5_run27_1_discordance_phenotype_summary.csv", index=False)

    period_summary = summarize_by_period(episodes)
    period_summary.to_csv(OUT_DIR / "v0_5_run27_1_discordance_by_era.csv", index=False)

    ambiguity = summarize_admission_ambiguity(admission)
    ambiguity.to_csv(OUT_DIR / "v0_5_run27_1_icd_episode_attribution_ambiguity.csv", index=False)

    proxy_only, icd_only, review_sample, lockbox = build_review_queues(episodes)
    proxy_only.to_csv(OUT_DIR / "v0_5_run27_1_proxy_positive_icd_negative_review_queue.csv", index=False)
    icd_only.to_csv(OUT_DIR / "v0_5_run27_1_icd_positive_proxy_negative_review_queue.csv", index=False)
    review_sample.to_csv(OUT_DIR / "v0_5_run27_1_balanced_manual_review_sample.csv", index=False)
    lockbox.to_csv(OUT_DIR / "v0_5_run27_1_lockbox_discordance_review_queue.csv", index=False)

    episodes.to_csv(OUT_DIR / "v0_5_run27_1_episode_discordance_phenotypes.csv", index=False)
    plot_phenotypes(phenotype_summary)
    plot_temporal_discordance(period_summary)
    write_notes(episode_agreement, admission_agreement, phenotype_summary, ambiguity, review_sample)

    ep = episode_agreement.iloc[0]
    adm = admission_agreement.iloc[0]
    print("")
    print("Grain-reconciled agreement:")
    print(f"  Episode-level positive Jaccard:   {ep['positive_set_jaccard']:.1%}")
    print(f"  Admission-level positive Jaccard: {adm['positive_set_jaccard']:.1%}")
    print(f"  Proxy+/ICD- episode queue: {len(proxy_only):,}")
    print(f"  Proxy-/ICD+ episode queue: {len(icd_only):,}")
    print(f"  Balanced manual-review sample: {len(review_sample):,}")
    print(f"  Lockbox discordant episodes: {len(lockbox):,}")
    print("")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

