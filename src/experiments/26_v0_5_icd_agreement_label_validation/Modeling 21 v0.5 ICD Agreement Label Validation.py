from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(r"C:\path\to\CVCML")
MIMIC_DIR = Path(r"C:\path\to\mimic-iv")

DATA_V05 = PROJECT_DIR / "data" / "v0_5"
RUN25_DIR = PROJECT_DIR / "Outputs" / "Run 25 (v0.5 Locked Temporal Evaluation)"
RUN26_DIR = PROJECT_DIR / "Outputs" / "Run 26 (v0.5 Locked Error Analysis)"
OUT_DIR = PROJECT_DIR / "Outputs" / "Run 27 (v0.5 ICD Agreement Label Validation)"
PLOT_DIR = OUT_DIR / "plots"

EPISODE_LABELS = DATA_V05 / "v0_5_run22_source_screened_episode_labels.csv"
LOCKBOX_PREDICTIONS = RUN25_DIR / "v0_5_run25_lockbox_predictions.csv"
RUN26_EPISODES = RUN26_DIR / "v0_5_run26_lockbox_episode_error_summary.csv"

DIAGNOSES = MIMIC_DIR / "hosp" / "diagnoses_icd.csv.gz"
D_ICD_DIAGNOSES = MIMIC_DIR / "hosp" / "d_icd_diagnoses.csv.gz"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace(".", "").strip()


def semicolon_join(values: pd.Series, max_items: int = 12) -> str:
    cleaned = [str(v) for v in values.dropna().astype(str).unique() if str(v).strip()]
    cleaned = cleaned[:max_items]
    return "; ".join(cleaned)


def classify_icd_row(code: str, version: int, title: str) -> dict[str, int]:
    norm = normalize_code(code)
    title_l = (title or "").lower()

    # Specific CVC bloodstream / CVC infection coding.
    icd9_specific = norm in {"99931", "99932"}
    icd9_cvc_any = norm in {"99931", "99932", "99933"}
    icd10_cvc_bsi = norm.startswith("T80211")
    icd10_cvc_unspecified = norm.startswith("T80219")
    icd10_cvc_other = norm.startswith("T80218")
    icd10_cvc_local = norm.startswith("T80212")
    icd10_cvc_root = norm == "T8021"

    cvc_bsi_specific = int(icd9_specific or icd10_cvc_bsi)
    cvc_infection_any = int(
        icd9_cvc_any
        or icd10_cvc_bsi
        or icd10_cvc_unspecified
        or icd10_cvc_other
        or icd10_cvc_local
        or icd10_cvc_root
    )

    vascular_device_broad = int(norm == "99662" or norm.startswith("T827"))
    catheter_infection_text = int(
        ("central venous catheter" in title_l and "infection" in title_l)
        or ("bloodstream infection" in title_l and "catheter" in title_l)
    )

    return {
        "icd_cvc_bsi_specific": cvc_bsi_specific,
        "icd_cvc_infection_any": cvc_infection_any,
        "icd_vascular_device_infection_broad": vascular_device_broad,
        "icd_cvc_or_vascular_infection_broad": int(cvc_infection_any or vascular_device_broad),
        "icd_catheter_infection_text_candidate": catheter_infection_text,
    }


def load_icd_candidate_dictionary() -> pd.DataFrame:
    d_icd = pd.read_csv(D_ICD_DIAGNOSES, dtype={"icd_code": str})
    d_icd["icd_code_norm"] = d_icd["icd_code"].map(normalize_code)

    regex = re.compile(
        r"central venous catheter|central line|bloodstream infection.*catheter|"
        r"catheter.*bloodstream|vascular device|vascular catheter|infection.*catheter|catheter.*infection",
        re.I,
    )
    candidate_mask = (
        d_icd["long_title"].fillna("").str.contains(regex)
        | d_icd["icd_code_norm"].str.startswith(("9993", "99662", "T8021", "T827"))
    )
    candidates = d_icd.loc[candidate_mask].copy()
    flags = candidates.apply(
        lambda row: pd.Series(
            classify_icd_row(row["icd_code"], int(row["icd_version"]), row.get("long_title", ""))
        ),
        axis=1,
    )
    candidates = pd.concat([candidates, flags], axis=1)
    candidates = candidates.sort_values(["icd_version", "icd_code_norm", "long_title"])
    return candidates


def load_relevant_diagnoses(hadm_ids: set[int]) -> pd.DataFrame:
    print("Loading MIMIC diagnoses for v0.5 admissions...")
    chunks = []
    usecols = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    for chunk in pd.read_csv(
        DIAGNOSES,
        usecols=usecols,
        dtype={"icd_code": str, "hadm_id": "Int64", "subject_id": "Int64"},
        chunksize=500_000,
    ):
        filtered = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
        if len(filtered):
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    diagnoses = pd.concat(chunks, ignore_index=True)

    d_icd = pd.read_csv(D_ICD_DIAGNOSES, dtype={"icd_code": str})
    diagnoses = diagnoses.merge(d_icd, on=["icd_code", "icd_version"], how="left")
    flags = diagnoses.apply(
        lambda row: pd.Series(
            classify_icd_row(row["icd_code"], int(row["icd_version"]), row.get("long_title", ""))
        ),
        axis=1,
    )
    diagnoses = pd.concat([diagnoses, flags], axis=1)
    diagnoses["icd_code_norm"] = diagnoses["icd_code"].map(normalize_code)
    return diagnoses


def aggregate_hadm_flags(diagnoses: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [
        "icd_cvc_bsi_specific",
        "icd_cvc_infection_any",
        "icd_vascular_device_infection_broad",
        "icd_cvc_or_vascular_infection_broad",
        "icd_catheter_infection_text_candidate",
    ]
    if len(diagnoses) == 0:
        return pd.DataFrame(columns=["hadm_id", *flag_cols])

    flagged = diagnoses[diagnoses[flag_cols].sum(axis=1) > 0].copy()
    if len(flagged) == 0:
        return pd.DataFrame(columns=["hadm_id", *flag_cols])

    grouped = flagged.groupby("hadm_id", as_index=False)[flag_cols].max()

    for flag in flag_cols:
        sub = flagged[flagged[flag] == 1].copy()
        code_col = f"{flag}_codes"
        title_col = f"{flag}_titles"
        if len(sub):
            text = (
                sub.groupby("hadm_id")
                .agg(
                    **{
                        code_col: ("icd_code", semicolon_join),
                        title_col: ("long_title", semicolon_join),
                    }
                )
                .reset_index()
            )
            grouped = grouped.merge(text, on="hadm_id", how="left")
        else:
            grouped[code_col] = np.nan
            grouped[title_col] = np.nan

    return grouped


def agreement_counts(df: pd.DataFrame, proxy_col: str, icd_col: str, group_col: str | None = None) -> pd.DataFrame:
    work = df.copy()
    if group_col is None:
        work["_group"] = "all"
        group_col = "_group"

    rows = []
    for group_value, group in work.groupby(group_col, dropna=False):
        proxy = group[proxy_col].fillna(0).astype(int)
        icd = group[icd_col].fillna(0).astype(int)

        both_pos = int(((proxy == 1) & (icd == 1)).sum())
        proxy_pos_icd_neg = int(((proxy == 1) & (icd == 0)).sum())
        proxy_neg_icd_pos = int(((proxy == 0) & (icd == 1)).sum())
        both_neg = int(((proxy == 0) & (icd == 0)).sum())
        proxy_pos = both_pos + proxy_pos_icd_neg
        icd_pos = both_pos + proxy_neg_icd_pos
        n = len(group)

        rows.append(
            {
                "group": group_value,
                "proxy_label": proxy_col,
                "icd_comparator": icd_col,
                "n_episodes": n,
                "proxy_positive": proxy_pos,
                "icd_positive": icd_pos,
                "both_positive": both_pos,
                "proxy_positive_icd_negative": proxy_pos_icd_neg,
                "proxy_negative_icd_positive": proxy_neg_icd_pos,
                "both_negative": both_neg,
                "proxy_prevalence": proxy_pos / n if n else np.nan,
                "icd_prevalence": icd_pos / n if n else np.nan,
                "agreement_rate": (both_pos + both_neg) / n if n else np.nan,
                "jaccard_positive": both_pos / (proxy_pos + proxy_neg_icd_pos) if (proxy_pos + proxy_neg_icd_pos) else np.nan,
                "proxy_ppv_vs_icd": both_pos / proxy_pos if proxy_pos else np.nan,
                "proxy_sensitivity_vs_icd": both_pos / icd_pos if icd_pos else np.nan,
                "icd_ppv_vs_proxy": both_pos / icd_pos if icd_pos else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_discordance(episodes: pd.DataFrame) -> pd.DataFrame:
    work = episodes.copy()
    proxy = work["cvc_bsi_strict_primary_or_uncertain_proxy"].fillna(0).astype(int)
    icd = work["icd_cvc_bsi_specific"].fillna(0).astype(int)
    work["proxy_icd_group"] = np.select(
        [
            (proxy == 1) & (icd == 1),
            (proxy == 1) & (icd == 0),
            (proxy == 0) & (icd == 1),
        ],
        ["proxy_pos_icd_pos", "proxy_pos_icd_neg", "proxy_neg_icd_pos"],
        default="proxy_neg_icd_neg",
    )

    summary = (
        work.groupby("proxy_icd_group")
        .agg(
            n_episodes=("episode_id", "count"),
            strict_proxy_positive=("cvc_bsi_strict_primary_or_uncertain_proxy", "sum"),
            broad_proxy_positive=("cvc_bsi_broad_proxy", "sum"),
            icd_specific_positive=("icd_cvc_bsi_specific", "sum"),
            icd_any_cvc_positive=("icd_cvc_infection_any", "sum"),
            icd_broad_positive=("icd_cvc_or_vascular_infection_broad", "sum"),
            median_source_culture_count=("nearby_nonblood_source_culture_count", "median"),
            median_concordant_source_culture_count=("concordant_nonblood_source_culture_count", "median"),
            median_hadm_source_icd_count=("hadm_source_icd_count", "median"),
            eligible_48h_rate=("eligible_48h_observed_exposure", "mean"),
        )
        .reset_index()
    )

    class_counts = (
        work.groupby(["proxy_icd_group", "source_screen_class"])
        .size()
        .reset_index(name="n")
        .sort_values(["proxy_icd_group", "n"], ascending=[True, False])
    )
    class_counts.to_csv(OUT_DIR / "v0_5_run27_discordance_source_class_counts.csv", index=False)

    return summary


def summarize_lockbox_intersection(episodes: pd.DataFrame) -> pd.DataFrame:
    lock = episodes[episodes["anchor_year_group"].eq("2020 - 2022")].copy()
    if "positive_episode" not in lock.columns:
        lock["positive_episode"] = lock["cvc_bsi_strict_primary_or_uncertain_proxy"]

    rows = []
    flags = [
        "icd_cvc_bsi_specific",
        "icd_cvc_infection_any",
        "icd_cvc_or_vascular_infection_broad",
    ]
    policies = [
        ("top_10_percent_rows", "Top 10% row policy"),
        ("top_150_episodes", "Top 150 episode review list"),
        ("top_250_episodes", "Top 250 episode review list"),
    ]
    for flag in flags:
        for policy_col, policy_name in policies:
            if policy_col not in lock.columns:
                continue
            selected = lock[policy_col].fillna(0).astype(int) == 1
            icd_pos = lock[flag].fillna(0).astype(int) == 1
            proxy_pos = lock["cvc_bsi_strict_primary_or_uncertain_proxy"].fillna(0).astype(int) == 1
            rows.append(
                {
                    "icd_comparator": flag,
                    "policy": policy_name,
                    "n_lockbox_episodes": len(lock),
                    "selected_episodes": int(selected.sum()),
                    "proxy_positive_episodes": int(proxy_pos.sum()),
                    "icd_positive_episodes": int(icd_pos.sum()),
                    "selected_proxy_positive": int((selected & proxy_pos).sum()),
                    "selected_icd_positive": int((selected & icd_pos).sum()),
                    "selected_proxy_or_icd_positive": int((selected & (proxy_pos | icd_pos)).sum()),
                    "selected_ppv_vs_proxy": (selected & proxy_pos).sum() / selected.sum() if selected.sum() else np.nan,
                    "selected_ppv_vs_icd": (selected & icd_pos).sum() / selected.sum() if selected.sum() else np.nan,
                    "selected_ppv_vs_proxy_or_icd": (selected & (proxy_pos | icd_pos)).sum() / selected.sum() if selected.sum() else np.nan,
                    "proxy_recall": (selected & proxy_pos).sum() / proxy_pos.sum() if proxy_pos.sum() else np.nan,
                    "icd_recall": (selected & icd_pos).sum() / icd_pos.sum() if icd_pos.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def source_class_icd_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    return (
        episodes.groupby("source_screen_class", dropna=False)
        .agg(
            n_episodes=("episode_id", "count"),
            strict_primary_or_uncertain_positive=("cvc_bsi_strict_primary_or_uncertain_proxy", "sum"),
            icd_cvc_bsi_specific=("icd_cvc_bsi_specific", "sum"),
            icd_cvc_infection_any=("icd_cvc_infection_any", "sum"),
            icd_cvc_or_vascular_infection_broad=("icd_cvc_or_vascular_infection_broad", "sum"),
        )
        .reset_index()
        .sort_values("n_episodes", ascending=False)
    )


def organism_icd_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    positives = episodes[episodes["cvc_bsi_strict_primary_or_uncertain_proxy"].fillna(0).astype(int) == 1].copy()
    records = []
    for _, row in positives.iterrows():
        org_text = row.get("strict_proxy_positive_orgs", "")
        if pd.isna(org_text) or not str(org_text).strip():
            records.append(
                {
                    "organism": "unknown",
                    "icd_cvc_bsi_specific": row.get("icd_cvc_bsi_specific", 0),
                    "icd_cvc_infection_any": row.get("icd_cvc_infection_any", 0),
                    "icd_cvc_or_vascular_infection_broad": row.get("icd_cvc_or_vascular_infection_broad", 0),
                }
            )
            continue
        for org in str(org_text).split(";"):
            org = org.strip()
            if org:
                records.append(
                    {
                        "organism": org,
                        "icd_cvc_bsi_specific": row.get("icd_cvc_bsi_specific", 0),
                        "icd_cvc_infection_any": row.get("icd_cvc_infection_any", 0),
                        "icd_cvc_or_vascular_infection_broad": row.get("icd_cvc_or_vascular_infection_broad", 0),
                    }
                )
    if not records:
        return pd.DataFrame()
    orgs = pd.DataFrame(records)
    out = (
        orgs.groupby("organism")
        .agg(
            n_proxy_positive_episodes=("organism", "count"),
            icd_specific_overlap=("icd_cvc_bsi_specific", "sum"),
            icd_any_cvc_overlap=("icd_cvc_infection_any", "sum"),
            icd_broad_overlap=("icd_cvc_or_vascular_infection_broad", "sum"),
        )
        .reset_index()
    )
    out["icd_specific_overlap_rate"] = out["icd_specific_overlap"] / out["n_proxy_positive_episodes"]
    out = out.sort_values(["n_proxy_positive_episodes", "icd_specific_overlap_rate"], ascending=[False, False])
    return out


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(30, 30, 30)) -> None:
    draw.text(xy, text, font=font, fill=fill)


def plot_overlap(agreement: pd.DataFrame) -> None:
    row = agreement[
        (agreement["group"] == "all")
        & (agreement["proxy_label"] == "cvc_bsi_strict_primary_or_uncertain_proxy")
        & (agreement["icd_comparator"] == "icd_cvc_bsi_specific")
    ].iloc[0]

    width, height = 1200, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42, bold=True)
    label_font = get_font(28, bold=True)
    text_font = get_font(24)
    small_font = get_font(20)

    draw_text(draw, (70, 45), "Run 27 Proxy vs ICD-Coded CLABSI Agreement", title_font)
    draw_text(draw, (70, 105), "Comparator: primary-or-uncertain strict CVC-BSI proxy vs ICD specific CVC bloodstream/infection codes", small_font, (80, 80, 80))

    proxy_only = int(row["proxy_positive_icd_negative"])
    both = int(row["both_positive"])
    icd_only = int(row["proxy_negative_icd_positive"])
    both_neg = int(row["both_negative"])
    n = int(row["n_episodes"])

    # Venn-like circles.
    draw.ellipse((160, 210, 610, 660), fill=(74, 144, 226), outline=(35, 90, 160), width=4)
    draw.ellipse((440, 210, 890, 660), fill=(246, 178, 107), outline=(180, 105, 30), width=4)
    # Blend the overlap.
    draw.ellipse((380, 210, 670, 660), fill=(150, 126, 178), outline=None)

    draw_text(draw, (220, 250), "Proxy+", label_font, "white")
    draw_text(draw, (665, 250), "ICD+", label_font, (60, 60, 60))
    draw_text(draw, (245, 415), f"{proxy_only:,}", get_font(44, True), "white")
    draw_text(draw, (490, 415), f"{both:,}", get_font(44, True), "white")
    draw_text(draw, (720, 415), f"{icd_only:,}", get_font(44, True), (60, 60, 60))

    metric_lines = [
        f"Total episodes: {n:,}",
        f"Both negative: {both_neg:,}",
        f"Proxy PPV vs ICD: {row['proxy_ppv_vs_icd']:.1%}" if not pd.isna(row["proxy_ppv_vs_icd"]) else "Proxy PPV vs ICD: n/a",
        f"Proxy sensitivity vs ICD: {row['proxy_sensitivity_vs_icd']:.1%}" if not pd.isna(row["proxy_sensitivity_vs_icd"]) else "Proxy sensitivity vs ICD: n/a",
        f"Positive-set Jaccard: {row['jaccard_positive']:.1%}" if not pd.isna(row["jaccard_positive"]) else "Positive-set Jaccard: n/a",
    ]
    y = 230
    for line in metric_lines:
        draw_text(draw, (930, y), line, text_font)
        y += 46

    draw_text(draw, (70, 700), "Interpretation: ICD codes are an administrative agreement check, not adjudicated CLABSI ground truth.", small_font, (90, 90, 90))
    img.save(PLOT_DIR / "v0_5_run27_proxy_icd_overlap.png")


def plot_source_class(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df = plot_df[plot_df["strict_primary_or_uncertain_positive"] > 0].head(10)
    if len(plot_df) == 0:
        return

    width, height = 1400, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(40, True)
    text_font = get_font(22)
    small_font = get_font(18)

    draw_text(draw, (60, 35), "Run 27 ICD Agreement by Source-Screen Class", title_font)
    x0, y0 = 520, 130
    bar_w, row_h = 650, 52
    max_n = max(plot_df["strict_primary_or_uncertain_positive"].max(), 1)

    for i, row in enumerate(plot_df.itertuples(index=False)):
        y = y0 + i * row_h
        label = str(row.source_screen_class)
        if len(label) > 45:
            label = label[:42] + "..."
        draw_text(draw, (60, y + 8), label, small_font)
        total = int(row.strict_primary_or_uncertain_positive)
        overlap = int(row.icd_cvc_bsi_specific)
        total_len = int(bar_w * total / max_n)
        overlap_len = int(bar_w * overlap / max_n)
        draw.rectangle((x0, y + 8, x0 + total_len, y + 36), fill=(74, 144, 226))
        draw.rectangle((x0, y + 8, x0 + overlap_len, y + 36), fill=(246, 178, 107))
        draw_text(draw, (x0 + bar_w + 20, y + 4), f"{overlap}/{total} ICD-specific", text_font)

    draw.rectangle((60, 670, 90, 695), fill=(74, 144, 226))
    draw_text(draw, (100, 666), "Proxy-positive episodes", small_font)
    draw.rectangle((360, 670, 390, 695), fill=(246, 178, 107))
    draw_text(draw, (400, 666), "ICD-specific overlap", small_font)
    img.save(PLOT_DIR / "v0_5_run27_source_class_icd_agreement.png")


def write_notes(
    episode_labels: pd.DataFrame,
    agreement: pd.DataFrame,
    lockbox_intersection: pd.DataFrame,
    discordance_summary: pd.DataFrame,
) -> None:
    primary = agreement[
        (agreement["group"] == "all")
        & (agreement["proxy_label"] == "cvc_bsi_strict_primary_or_uncertain_proxy")
        & (agreement["icd_comparator"] == "icd_cvc_bsi_specific")
    ].iloc[0]
    broad = agreement[
        (agreement["group"] == "all")
        & (agreement["proxy_label"] == "cvc_bsi_strict_primary_or_uncertain_proxy")
        & (agreement["icd_comparator"] == "icd_cvc_or_vascular_infection_broad")
    ].iloc[0]

    lines = [
        "# Run 27 - ICD Agreement Label Validation",
        "",
        "## Purpose",
        "",
        "Run 27 evaluates whether the v0.5 microbiology/timing/source-screened CVC-BSI proxy aligns with administrative ICD-coded CLABSI or central-line infection diagnoses. This is a label-validity and framing run, not a model tuning run.",
        "",
        "## ICD Comparator Definitions",
        "",
        "- ICD-specific CVC bloodstream comparator: ICD-9 99931/99932 and ICD-10 T80.211* bloodstream infection due to central venous catheter.",
        "- ICD-any CVC infection comparator: ICD-specific plus local, other, or unspecified central venous catheter infection codes such as ICD-9 99933 and ICD-10 T80.212*/T80.218*/T80.219*.",
        "- ICD-broad comparator: ICD-any CVC infection plus broader vascular-device infection codes such as ICD-9 99662 and ICD-10 T82.7*.",
        "",
        "## Main Agreement Results",
        "",
        f"- Episodes evaluated: {int(primary['n_episodes']):,}.",
        f"- Primary-or-uncertain proxy positives: {int(primary['proxy_positive']):,} ({primary['proxy_prevalence']:.1%}).",
        f"- ICD-specific positives: {int(primary['icd_positive']):,} ({primary['icd_prevalence']:.1%}).",
        f"- Overlap between proxy-positive and ICD-specific positives: {int(primary['both_positive']):,}.",
        f"- Proxy-positive but ICD-specific negative: {int(primary['proxy_positive_icd_negative']):,}.",
        f"- ICD-specific positive but proxy-negative: {int(primary['proxy_negative_icd_positive']):,}.",
        f"- Positive-set Jaccard vs ICD-specific: {primary['jaccard_positive']:.1%}.",
        f"- Against the broader ICD comparator, overlap is {int(broad['both_positive']):,} with positive-set Jaccard {broad['jaccard_positive']:.1%}.",
        "",
        "## Interpretation",
        "",
        "ICD-coded CLABSI is useful as an external administrative agreement check, but it should not replace the structured proxy as ground truth. Proxy-positive/ICD-negative episodes likely include clinically plausible culture/timing events that were not coded as central-line infection. ICD-positive/proxy-negative episodes may reflect coding without a qualifying reconstructed line episode, culture timing mismatch, non-bloodstream/local line infection coding, or incomplete procedureevents line documentation.",
        "",
        "## Operational Implication",
        "",
        "The honest language remains: strict CVC-associated BSI proxy, with ICD-coded CLABSI agreement as a secondary validation check. Strong model claims should be framed around prospective risk stratification of the proxy rather than adjudicated NHSN CLABSI.",
        "",
    ]

    if len(lockbox_intersection):
        top150 = lockbox_intersection[
            (lockbox_intersection["icd_comparator"] == "icd_cvc_bsi_specific")
            & (lockbox_intersection["policy"] == "Top 150 episode review list")
        ]
        if len(top150):
            row = top150.iloc[0]
            lines.extend(
                [
                    "## Lockbox Model/ICD Intersection",
                    "",
                    f"- In the lockbox, the top 150 episode review list selected {int(row['selected_episodes']):,} episodes.",
                    f"- It captured {int(row['selected_proxy_positive']):,} proxy-positive episodes and {int(row['selected_icd_positive']):,} ICD-specific positive episodes.",
                    f"- Selected PPV vs proxy: {row['selected_ppv_vs_proxy']:.1%}. Selected PPV vs ICD-specific: {row['selected_ppv_vs_icd']:.1%}.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Key Output Files",
            "",
            "- `v0_5_run27_icd_clabsi_code_candidates.csv`",
            "- `v0_5_run27_hadm_icd_clabsi_flags.csv`",
            "- `v0_5_run27_episode_icd_agreement.csv`",
            "- `v0_5_run27_proxy_icd_agreement_table.csv`",
            "- `v0_5_run27_lockbox_icd_model_intersection.csv`",
            "- `plots/v0_5_run27_proxy_icd_overlap.png`",
        ]
    )
    (OUT_DIR / "run_27_v0_5_icd_agreement_label_validation_notes.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    print("Run 27: ICD agreement label validation")

    print("Loading v0.5 source-screened episode labels...")
    episodes = pd.read_csv(EPISODE_LABELS)
    episodes["hadm_id"] = episodes["hadm_id"].astype("Int64")
    hadm_ids = set(episodes["hadm_id"].dropna().astype(int).tolist())
    print(f"  Episodes: {len(episodes):,}")
    print(f"  Unique admissions: {len(hadm_ids):,}")

    print("Writing ICD code candidate dictionary...")
    candidates = load_icd_candidate_dictionary()
    candidates.to_csv(OUT_DIR / "v0_5_run27_icd_clabsi_code_candidates.csv", index=False)

    diagnoses = load_relevant_diagnoses(hadm_ids)
    diagnoses.to_csv(OUT_DIR / "v0_5_run27_relevant_diagnoses_with_icd_flags.csv", index=False)

    hadm_flags = aggregate_hadm_flags(diagnoses)
    hadm_flags.to_csv(OUT_DIR / "v0_5_run27_hadm_icd_clabsi_flags.csv", index=False)

    episodes = episodes.merge(hadm_flags, on="hadm_id", how="left")
    flag_cols = [
        "icd_cvc_bsi_specific",
        "icd_cvc_infection_any",
        "icd_vascular_device_infection_broad",
        "icd_cvc_or_vascular_infection_broad",
        "icd_catheter_infection_text_candidate",
    ]
    for col in flag_cols:
        episodes[col] = episodes[col].fillna(0).astype(int)

    # Bring in the frozen lockbox model-review outputs for operational overlap.
    if RUN26_EPISODES.exists():
        run26_cols = [
            "episode_id",
            "positive_episode",
            "max_score",
            "mean_score",
            "landmark_rows",
            "top_5_percent_rows",
            "top_10_percent_rows",
            "top_100_episodes",
            "top_150_episodes",
            "top_250_episodes",
            "capture_group_top_10_rows",
        ]
        run26 = pd.read_csv(RUN26_EPISODES, usecols=lambda c: c in run26_cols)
        episodes = episodes.merge(run26, on="episode_id", how="left")

    episodes.to_csv(OUT_DIR / "v0_5_run27_episode_icd_agreement.csv", index=False)

    proxy_cols = [
        "cvc_bsi_broad_proxy",
        "cvc_bsi_strict_proxy",
        "cvc_bsi_strict_primary_likely_proxy",
        "cvc_bsi_strict_primary_or_uncertain_proxy",
        "cvc_bsi_strict_secondary_possible_proxy",
    ]
    icd_cols = [
        "icd_cvc_bsi_specific",
        "icd_cvc_infection_any",
        "icd_cvc_or_vascular_infection_broad",
    ]

    agreement_frames = []
    for proxy_col in proxy_cols:
        for icd_col in icd_cols:
            agreement_frames.append(agreement_counts(episodes, proxy_col, icd_col))
            agreement_frames.append(agreement_counts(episodes, proxy_col, icd_col, "anchor_year_group"))
    agreement = pd.concat(agreement_frames, ignore_index=True)
    agreement.to_csv(OUT_DIR / "v0_5_run27_proxy_icd_agreement_table.csv", index=False)

    discordance = summarize_discordance(episodes)
    discordance.to_csv(OUT_DIR / "v0_5_run27_discordant_case_summary.csv", index=False)

    source_summary = source_class_icd_summary(episodes)
    source_summary.to_csv(OUT_DIR / "v0_5_run27_source_class_icd_agreement.csv", index=False)

    organism_summary = organism_icd_summary(episodes)
    organism_summary.to_csv(OUT_DIR / "v0_5_run27_organism_icd_agreement.csv", index=False)

    lockbox_intersection = summarize_lockbox_intersection(episodes)
    lockbox_intersection.to_csv(OUT_DIR / "v0_5_run27_lockbox_icd_model_intersection.csv", index=False)

    plot_overlap(agreement)
    plot_source_class(source_summary)
    write_notes(episodes, agreement, lockbox_intersection, discordance)

    primary = agreement[
        (agreement["group"] == "all")
        & (agreement["proxy_label"] == "cvc_bsi_strict_primary_or_uncertain_proxy")
        & (agreement["icd_comparator"] == "icd_cvc_bsi_specific")
    ].iloc[0]
    print("")
    print("Primary proxy vs ICD-specific CVC BSI agreement:")
    print(f"  Episodes: {int(primary['n_episodes']):,}")
    print(f"  Proxy positives: {int(primary['proxy_positive']):,} ({primary['proxy_prevalence']:.1%})")
    print(f"  ICD positives: {int(primary['icd_positive']):,} ({primary['icd_prevalence']:.1%})")
    print(f"  Both positive: {int(primary['both_positive']):,}")
    print(f"  Proxy+/ICD-: {int(primary['proxy_positive_icd_negative']):,}")
    print(f"  Proxy-/ICD+: {int(primary['proxy_negative_icd_positive']):,}")
    print(f"  Positive-set Jaccard: {primary['jaccard_positive']:.1%}")
    print("")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

