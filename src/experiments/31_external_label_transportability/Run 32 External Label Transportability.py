"""Run 32: publication-oriented external label transportability characterization."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


COMMENSAL_PATTERNS = [
    r"COAGULASE NEGATIVE", r"COAGULASE-NEGATIVE",
    r"STAPHYLOCOCCUS EPIDERMIDIS", r"STAPHYLOCOCCUS HOMINIS",
    r"STAPHYLOCOCCUS HAEMOLYTICUS", r"STAPHYLOCOCCUS CAPITIS",
    r"STAPHYLOCOCCUS WARNERI", r"STAPHYLOCOCCUS LUGDUNENSIS",
    r"CORYNEBACTERIUM", r"\bBACILLUS\b", r"LACTOBACILLUS",
    r"MICROCOCCUS", r"CUTIBACTERIUM", r"PROPIONIBACTERIUM",
    r"DIPHTHEROID", r"VIRIDANS", r"AEROCOCCUS",
]
COMMENSAL_RE = re.compile("|".join(COMMENSAL_PATTERNS), re.IGNORECASE)


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def canonical_organism(name: str) -> str:
    name = (name or "").strip().upper()
    aliases = {
        "STAPH AUREUS COAG +": "STAPHYLOCOCCUS AUREUS",
        "STAPHYLOCOCCUS, COAGULASE NEGATIVE": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
        "STAPHYLOCOCCUS COAGULASE NEGATIVE": "COAGULASE NEGATIVE STAPHYLOCOCCUS",
        "STAPHYLOCOCCUS, COAGULASE NEGATIVE, PRESUMPTIVELY NOT S. SAPROPHYTICUS":
            "COAGULASE NEGATIVE STAPHYLOCOCCUS",
        "STREPTOCOCCUS VIRIDANS": "STREPTOCOCCUS VIRIDANS GROUP (OTHER)",
    }
    return aliases.get(name, name)


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return center - half, center + half


def load_events(path: Path) -> pd.DataFrame:
    cols = ["anon_id", "order_proc_id_coded", "culture_description", "organism",
            "order_time_jittered_utc_shifted"]
    frames = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=250_000, low_memory=False):
        chunk["source"] = norm(chunk["culture_description"])
        chunk["organism_norm"] = norm(chunk["organism"])
        chunk["culture_date"] = pd.to_datetime(
            chunk["order_time_jittered_utc_shifted"], errors="coerce"
        ).dt.normalize()
        chunk = chunk[["anon_id", "order_proc_id_coded", "source", "organism_norm",
                       "culture_date"]].drop_duplicates()
        frames.append(chunk)
    events = pd.concat(frames, ignore_index=True).drop_duplicates()
    events["accession"] = events["order_proc_id_coded"].astype(str)
    events["positive"] = events["organism_norm"].ne("") & ~events["organism_norm"].isin(
        {"NO GROWTH", "NEGATIVE"}
    )
    events["commensal"] = events["organism_norm"].map(
        lambda x: bool(COMMENSAL_RE.search(x))
    )
    return events


def repeat_commensal_accessions(positive_blood: pd.DataFrame) -> set[str]:
    result = set()
    comm = positive_blood[positive_blood["commensal"]]
    for (_, organism), group in comm.groupby(["anon_id", "organism_norm"]):
        group = group.dropna(subset=["culture_date"]).sort_values("culture_date")
        for date in group["culture_date"].drop_duplicates():
            nearby = group[group["culture_date"].between(
                date - pd.Timedelta(days=1), date + pd.Timedelta(days=1)
            )]
            if nearby["accession"].nunique() >= 2:
                result.update(nearby["accession"])
    return result


def source_window_analysis(events: pd.DataFrame, strict_accessions: set[str]) -> pd.DataFrame:
    blood = events[(events["source"].eq("BLOOD")) & events["positive"]]
    strict = blood[blood["accession"].isin(strict_accessions)]
    nonblood = events[events["positive"] & events["source"].isin(
        ["URINE", "RESPIRATORY_TRACT"]
    )]
    index = defaultdict(list)
    for row in nonblood.itertuples(index=False):
        index[(str(row.anon_id), row.organism_norm)].append((row.culture_date, row.source))

    accession_info = []
    for accession, group in strict.groupby("accession"):
        patient = str(group["anon_id"].iloc[0])
        date = group["culture_date"].min()
        organisms = set(group["organism_norm"]) - {""}
        row = {"accession": accession}
        for window in [1, 2, 3]:
            sources = set()
            if pd.notna(date):
                for organism in organisms:
                    for other_date, source in index.get((patient, organism), []):
                        if pd.notna(other_date) and abs((other_date - date).days) <= window:
                            sources.add(source)
            row[f"source_pm{window}d"] = int(bool(sources))
            row[f"sources_pm{window}d"] = "; ".join(sorted(sources))
        accession_info.append(row)
    detail = pd.DataFrame(accession_info)
    rows = []
    for window in [1, 2, 3]:
        n = int(detail[f"source_pm{window}d"].sum())
        total = len(detail)
        low, high = wilson(n, total)
        rows.append({"window_days": window, "strict_accessions": total,
                     "matching_source_accessions": n, "proportion": n / total,
                     "ci_low": low, "ci_high": high})
    return detail, pd.DataFrame(rows)


def load_accession_covariates(root: Path) -> pd.DataFrame:
    demo = pd.read_csv(root / "demographics_deid_tj.csv",
                       usecols=["order_proc_id_coded", "age", "gender"])
    ward = pd.read_csv(root / "ward_type_deid_tj.csv",
                       usecols=["order_proc_id_coded", "hosp_ward_IP", "hosp_ward_OP",
                                "hosp_ward_ER", "hosp_ward_UC", "hosp_ward_day_surg"])
    demo = demo.drop_duplicates("order_proc_id_coded")
    ward = ward.drop_duplicates("order_proc_id_coded")
    cov = demo.merge(ward, on="order_proc_id_coded", how="outer")
    cov["accession"] = cov["order_proc_id_coded"].astype(str)
    ward_cols = ["hosp_ward_IP", "hosp_ward_ER", "hosp_ward_OP", "hosp_ward_UC",
                 "hosp_ward_day_surg"]
    names = ["inpatient", "emergency", "outpatient", "urgent_care", "day_surgery"]
    values = cov[ward_cols].fillna(0).to_numpy()
    cov["setting"] = [names[int(np.argmax(row))] if row.max() > 0 else "unknown"
                      for row in values]
    cov["age"] = cov["age"].fillna("Unknown")
    cov["gender"] = cov["gender"].fillna("Unknown")
    return cov[["accession", "age", "gender", "setting"]]


def subgroup_table(accessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["age", "gender", "setting"]:
        for level, group in accessions.groupby(variable, dropna=False):
            total = len(group)
            strict_n = int(group["strict"].sum())
            source_n = int(group.loc[group["strict"].eq(1), "source_pm3d"].sum())
            strict_low, strict_high = wilson(strict_n, total)
            source_low, source_high = wilson(source_n, strict_n)
            rows.append({"variable": variable, "level": level,
                         "positive_blood_accessions": total,
                         "strict_accessions": strict_n,
                         "strict_rate": strict_n / total if total else np.nan,
                         "strict_ci_low": strict_low, "strict_ci_high": strict_high,
                         "strict_with_source_pm3d": source_n,
                         "source_rate_among_strict": source_n / strict_n if strict_n else np.nan,
                         "source_ci_low": source_low, "source_ci_high": source_high})
    return pd.DataFrame(rows)


def organism_comparison(events: pd.DataFrame, strict_accessions: set[str],
                        mimic_path: Path) -> tuple[pd.DataFrame, dict]:
    armd = events[events["positive"] & events["source"].eq("BLOOD") &
                  events["accession"].isin(strict_accessions)].copy()
    armd["organism_canonical"] = armd["organism_norm"].map(canonical_organism)
    armd_counts = armd.groupby("organism_canonical")["accession"].nunique().rename("armd_accessions")
    mimic = pd.read_csv(mimic_path)
    mimic["organism_canonical"] = norm(mimic["org_name"]).map(canonical_organism)
    mimic_counts = mimic.groupby("organism_canonical")["episodes"].sum().rename("mimic_episodes")
    compare = pd.concat([armd_counts, mimic_counts], axis=1).fillna(0).reset_index()
    compare = compare.rename(columns={"organism_canonical": "organism_norm"})
    compare["armd_share"] = compare["armd_accessions"] / compare["armd_accessions"].sum()
    compare["mimic_share"] = compare["mimic_episodes"] / compare["mimic_episodes"].sum()
    compare["shared_organism"] = (compare["armd_accessions"] > 0) & (compare["mimic_episodes"] > 0)
    compare = compare.sort_values("mimic_episodes", ascending=False)
    p = compare["armd_share"].to_numpy() + 1e-12
    q = compare["mimic_share"].to_numpy() + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    js = float(
        0.5 * np.sum(p * np.log2(p / midpoint))
        + 0.5 * np.sum(q * np.log2(q / midpoint))
    )
    shared = compare[compare["shared_organism"]]
    armd_rank = shared["armd_accessions"].rank(method="average").to_numpy()
    mimic_rank = shared["mimic_episodes"].rank(method="average").to_numpy()
    rho = float(np.corrcoef(armd_rank, mimic_rank)[0, 1]) if len(shared) > 1 else np.nan
    metrics = {"shared_organisms": int(len(shared)), "jensen_shannon_divergence": js,
               "shared_organism_spearman_rho": float(rho),
               "shared_organism_spearman_p": None}
    return compare, metrics


def make_plots(output: Path, windows: pd.DataFrame, subgroups: pd.DataFrame,
               organisms: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(windows["window_days"], windows["proportion"],
                yerr=[windows["proportion"] - windows["ci_low"],
                      windows["ci_high"] - windows["proportion"]],
                marker="o", linewidth=2, capsize=4, color="#176B87")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Secondary-source search window (+/- days)")
    ax.set_ylabel("Strict events with same-organism nonblood culture")
    ax.set_title("ARMD-MGB Secondary-Source Window Sensitivity")
    ax.set_ylim(0, max(windows["ci_high"].max() * 1.2, 0.05))
    fig.tight_layout()
    fig.savefig(plot_dir / "run32_secondary_source_window_sensitivity.png", dpi=220)
    plt.close(fig)

    top_names = organisms.nlargest(12, "mimic_episodes")["organism_norm"]
    top = organisms[organisms["organism_norm"].isin(top_names)].copy()
    top = top.sort_values("mimic_share")
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y - 0.2, top["mimic_share"], height=0.38, label="MIMIC-IV episodes",
            color="#3B82B9")
    ax.barh(y + 0.2, top["armd_share"], height=0.38, label="ARMD-MGB accessions",
            color="#D9772B")
    ax.set_yticks(y, top["organism_norm"])
    ax.set_xlabel("Share of qualifying organism events")
    ax.set_title("Strict-Rule Organism Profile: MIMIC-IV vs ARMD-MGB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "run32_mimic_armd_organism_profile.png", dpi=220)
    plt.close(fig)

    display = subgroups[(subgroups["variable"].isin(["gender", "setting"])) &
                        (subgroups["positive_blood_accessions"] >= 100)].copy()
    display["label"] = display["variable"] + ": " + display["level"].astype(str)
    display = display.sort_values("strict_rate")
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(display))))
    ax.errorbar(display["strict_rate"], np.arange(len(display)),
                xerr=[display["strict_rate"] - display["strict_ci_low"],
                      display["strict_ci_high"] - display["strict_rate"]],
                fmt="o", capsize=3, color="#2A7F62")
    ax.set_yticks(np.arange(len(display)), display["label"])
    ax.set_xlabel("Strict organism-rule positivity among positive blood cultures")
    ax.set_title("ARMD-MGB Label Stability by Setting and Sex")
    fig.tight_layout()
    fig.savefig(plot_dir / "run32_subgroup_strict_rule_stability.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armd-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run31-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading ARMD-MGB culture events...", flush=True)
    events = load_events(args.armd_root / "microbiology_cohort_deid_tj_updated.csv")
    positive_blood = events[events["positive"] & events["source"].eq("BLOOD")]
    pathogen = set(positive_blood.loc[~positive_blood["commensal"], "accession"])
    commensal = set(positive_blood.loc[positive_blood["commensal"], "accession"])
    repeated = repeat_commensal_accessions(positive_blood)
    strict = pathogen | repeated

    source_detail, windows = source_window_analysis(events, strict)
    windows.to_csv(args.output / "run32_secondary_source_window_sensitivity.csv", index=False)

    accession = positive_blood.groupby("accession", as_index=False).agg(
        anon_id=("anon_id", "first"), organisms=("organism_norm", lambda x: "; ".join(sorted(set(x))))
    )
    accession["strict"] = accession["accession"].isin(strict).astype(int)
    accession["recognized_pathogen"] = accession["accession"].isin(pathogen).astype(int)
    accession["commensal_positive"] = accession["accession"].isin(commensal).astype(int)
    accession["repeated_commensal_48h"] = accession["accession"].isin(repeated).astype(int)
    accession = accession.merge(source_detail, on="accession", how="left")
    for col in ["source_pm1d", "source_pm2d", "source_pm3d"]:
        accession[col] = accession[col].fillna(0).astype(int)
    accession = accession.merge(load_accession_covariates(args.armd_root), on="accession", how="left")
    accession.to_csv(args.output / "run32_armd_positive_blood_accession_characterization.csv",
                     index=False)

    subgroups = subgroup_table(accession)
    subgroups.to_csv(args.output / "run32_armd_subgroup_stability.csv", index=False)

    mimic_path = args.project_root / "data" / "v0_5" / "v0_5_qualifying_organism_counts.csv"
    organisms, comparison_metrics = organism_comparison(events, strict, mimic_path)
    organisms.to_csv(args.output / "run32_mimic_armd_organism_comparison.csv", index=False)

    eicu = pd.read_csv(args.run31_output / "run31b_eicu_feasibility_summary.csv").iloc[0]
    eligibility = pd.DataFrame([
        {"dataset": "ARMD-MGB", "full_model_validation": False,
         "organism_rule_validation": True, "secondary_source_validation": "partial",
         "eligible_positive_events": len(strict),
         "decision": "Use for external component-level label validation"},
        {"dataset": "eICU-CRD", "full_model_validation": False,
         "organism_rule_validation": False, "secondary_source_validation": "no",
         "eligible_positive_events": int(eicu["positive_blood_stays_ge48h_after_explicit_placement"]),
         "decision": "Fail feasibility gate; do not estimate model performance"},
    ])
    eligibility.to_csv(args.output / "run32_external_validation_scope.csv", index=False)

    if not args.skip_plots:
        make_plots(args.output, windows, subgroups, organisms)

    total = len(accession)
    summary = {
        "run": 32,
        "title": "External Label Transportability Characterization",
        "armd_positive_blood_accessions": total,
        "armd_strict_accessions": len(strict),
        "armd_strict_rate": len(strict) / total,
        "armd_recognized_pathogen_accessions": len(pathogen),
        "armd_commensal_positive_accessions": len(commensal),
        "armd_repeated_commensal_48h_accessions": len(repeated),
        "secondary_source_window_results": windows.to_dict(orient="records"),
        **comparison_metrics,
        "eicu_eligible_positive_events": int(
            eicu["positive_blood_stays_ge48h_after_explicit_placement"]
        ),
        "claim": "External component-level label transportability, not external model validation",
    }
    (args.output / "run32_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    notes = f"""# Run 32 - External Label Transportability Characterization

## Question

Do the organism and partial secondary-source components of the MIMIC-IV strict CVC-associated
BSI proxy behave coherently in ARMD-MGB, and can eICU support any honest model validation?

## Results

- ARMD-MGB positive blood-culture accessions: {total:,}
- Strict organism-rule accessions: {len(strict):,} ({len(strict)/total:.1%})
- Recognized-pathogen accessions: {len(pathogen):,}
- Common-commensal-positive accessions: {len(commensal):,}
- Repeated-commensal accessions qualifying within 48 hours: {len(repeated):,}
- Same-organism urine/respiratory evidence within +/-1 day: {windows.iloc[0].proportion:.1%}
- Same-organism urine/respiratory evidence within +/-2 days: {windows.iloc[1].proportion:.1%}
- Same-organism urine/respiratory evidence within +/-3 days: {windows.iloc[2].proportion:.1%}
- Shared-organism rank correlation with MIMIC-IV: rho={comparison_metrics['shared_organism_spearman_rho']:.3f}
- Organism-distribution Jensen-Shannon divergence: {comparison_metrics['jensen_shannon_divergence']:.3f}
- eICU positive events meeting the explicit-placement feasibility criterion: {int(eicu['positive_blood_stays_ge48h_after_explicit_placement'])}

## Interpretation

ARMD-MGB provides external evidence that the organism-rule framework can be applied at another
health system and quantifies how partial secondary-source evidence changes with the search
window. Differences in organism distributions are expected because ARMD-MGB includes cultures
from multiple care settings and is not restricted to central-line episodes.

Same-organism urine or respiratory cultures are markers of possible secondary-source evidence,
not proof that a bloodstream infection was secondary. Wound, abdominal, symptom, MBI-LCBI,
and infection-prevention adjudication data remain unavailable.

eICU does not contain enough microbiology-linked, explicitly documented central-line episodes
to estimate discrimination, calibration, or review burden. No AUROC or PR-AUC is reported.

## Manuscript-safe claim

The study has temporal full-model evaluation within MIMIC-IV and external component-level
validation of organism and partial source-attribution logic in ARMD-MGB. It does not yet have
external institutional validation of the frozen prediction model.
"""
    (args.output / "run32_external_label_transportability_notes.md").write_text(
        notes, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

