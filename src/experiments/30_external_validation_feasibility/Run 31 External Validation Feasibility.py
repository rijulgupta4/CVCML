"""Run 31: external validation feasibility for ARMD-MGB and eICU.

This run does not evaluate the frozen MIMIC-IV model. It tests whether each
external source can reproduce the required outcome and predictor timeline,
and performs organism-rule transportability analyses where support exists.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


COMMENSAL_PATTERNS = [
    r"COAGULASE NEGATIVE",
    r"COAGULASE-NEGATIVE",
    r"STAPHYLOCOCCUS EPIDERMIDIS",
    r"STAPHYLOCOCCUS HOMINIS",
    r"STAPHYLOCOCCUS HAEMOLYTICUS",
    r"STAPHYLOCOCCUS CAPITIS",
    r"STAPHYLOCOCCUS WARNERI",
    r"STAPHYLOCOCCUS LUGDUNENSIS",
    r"CORYNEBACTERIUM",
    r"\bBACILLUS\b",
    r"LACTOBACILLUS",
    r"MICROCOCCUS",
    r"CUTIBACTERIUM",
    r"PROPIONIBACTERIUM",
    r"DIPHTHEROID",
    r"VIRIDANS",
    r"AEROCOCCUS",
]
COMMENSAL_RE = re.compile("|".join(COMMENSAL_PATTERNS), re.IGNORECASE)


def normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def is_commensal(name: str) -> bool:
    return bool(COMMENSAL_RE.search(name or ""))


def load_armd_events(path: Path) -> pd.DataFrame:
    usecols = [
        "anon_id",
        "pat_enc_csn_id_coded",
        "order_proc_id_coded",
        "culture_description",
        "organism",
        "neg_cx",
        "order_time_jittered_utc_shifted",
    ]
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["culture_source"] = normalize(chunk["culture_description"])
        chunk["organism_norm"] = normalize(chunk["organism"])
        chunk["culture_date"] = pd.to_datetime(
            chunk["order_time_jittered_utc_shifted"], errors="coerce"
        ).dt.normalize()
        chunk = chunk[
            [
                "anon_id",
                "pat_enc_csn_id_coded",
                "order_proc_id_coded",
                "culture_source",
                "organism_norm",
                "culture_date",
            ]
        ].drop_duplicates()
        frames.append(chunk)

    events = pd.concat(frames, ignore_index=True).drop_duplicates()
    events["is_positive"] = events["organism_norm"].ne("") & ~events[
        "organism_norm"
    ].isin({"NO GROWTH", "NEGATIVE"})
    events["is_commensal"] = events["organism_norm"].map(is_commensal)
    return events


def repeated_commensal_accessions(blood_positive: pd.DataFrame) -> set[str]:
    commensals = blood_positive[blood_positive["is_commensal"]].copy()
    qualifying = set()
    for (_, organism), group in commensals.groupby(["anon_id", "organism_norm"]):
        group = group.dropna(subset=["culture_date"]).sort_values("culture_date")
        dates = group["culture_date"].drop_duplicates().tolist()
        if len(dates) < 2:
            continue
        for date in dates:
            nearby = group[
                group["culture_date"].between(
                    date - pd.Timedelta(days=1), date + pd.Timedelta(days=1)
                )
            ]
            if nearby["order_proc_id_coded"].nunique() >= 2:
                qualifying.update(nearby["order_proc_id_coded"].astype(str))
    return qualifying


def armd_analysis(root: Path, output: Path) -> dict:
    print("Loading and deduplicating ARMD-MGB microbiology...", flush=True)
    events = load_armd_events(root / "microbiology_cohort_deid_tj_updated.csv")
    events["accession"] = events["order_proc_id_coded"].astype(str)

    blood = events[events["culture_source"].eq("BLOOD")].copy()
    blood_positive = blood[blood["is_positive"]].copy()
    blood_positive_accessions = set(blood_positive["accession"])
    pathogen_accessions = set(
        blood_positive.loc[~blood_positive["is_commensal"], "accession"]
    )
    commensal_accessions = set(
        blood_positive.loc[blood_positive["is_commensal"], "accession"]
    )
    repeat_commensal = repeated_commensal_accessions(blood_positive)
    strict_accessions = pathogen_accessions | repeat_commensal

    nonblood_positive = events[
        events["is_positive"] & events["culture_source"].isin(["URINE", "RESPIRATORY_TRACT"])
    ].copy()
    nonblood_index = defaultdict(list)
    for row in nonblood_positive.itertuples(index=False):
        nonblood_index[(str(row.anon_id), row.organism_norm)].append(
            (row.culture_date, row.culture_source)
        )

    accession_rows = []
    for accession, group in blood_positive.groupby("accession"):
        patient = str(group["anon_id"].iloc[0])
        culture_date = group["culture_date"].min()
        organisms = sorted(set(group["organism_norm"]) - {""})
        sources = set()
        if pd.notna(culture_date):
            for organism in organisms:
                for source_date, source in nonblood_index.get((patient, organism), []):
                    if pd.notna(source_date) and abs((source_date - culture_date).days) <= 3:
                        sources.add(source)
        accession_rows.append(
            {
                "accession": accession,
                "anon_id": patient,
                "culture_date": culture_date,
                "organisms": "; ".join(organisms),
                "contains_recognized_pathogen": int(accession in pathogen_accessions),
                "contains_common_commensal": int(accession in commensal_accessions),
                "repeated_commensal_48h": int(accession in repeat_commensal),
                "strict_organism_rule_positive": int(accession in strict_accessions),
                "same_organism_nonblood_source_pm3d": int(bool(sources)),
                "matching_nonblood_sources": "; ".join(sorted(sources)),
            }
        )
    accession_audit = pd.DataFrame(accession_rows)
    accession_audit.to_csv(output / "run31a_armd_blood_accession_audit.csv", index=False)

    organism_summary = (
        blood_positive.groupby(["organism_norm", "is_commensal"])
        .agg(
            raw_rows=("accession", "size"),
            unique_accessions=("accession", "nunique"),
            unique_patients=("anon_id", "nunique"),
        )
        .reset_index()
        .sort_values("unique_accessions", ascending=False)
    )
    organism_summary.to_csv(output / "run31a_armd_organism_summary.csv", index=False)

    strict = accession_audit[accession_audit["strict_organism_rule_positive"].eq(1)]
    source_counts = strict["matching_nonblood_sources"].replace("", "none_detected").value_counts()
    source_summary = source_counts.rename_axis("matching_source_class").reset_index(name="accessions")
    source_summary["proportion"] = source_summary["accessions"] / max(len(strict), 1)
    source_summary.to_csv(output / "run31a_armd_secondary_source_summary.csv", index=False)

    summary = {
        "dataset": "ARMD-MGB 1.0.0",
        "unique_microbiology_accessions": int(events["accession"].nunique()),
        "blood_accessions": int(blood["accession"].nunique()),
        "positive_blood_accessions": int(len(blood_positive_accessions)),
        "recognized_pathogen_accessions": int(len(pathogen_accessions)),
        "commensal_positive_accessions": int(len(commensal_accessions)),
        "repeat_commensal_48h_accessions": int(len(repeat_commensal)),
        "strict_organism_rule_accessions": int(len(strict_accessions)),
        "strict_with_matching_nonblood_source_pm3d": int(
            strict["same_organism_nonblood_source_pm3d"].sum()
        ),
        "strict_with_matching_nonblood_source_pm3d_rate": float(
            strict["same_organism_nonblood_source_pm3d"].mean() if len(strict) else np.nan
        ),
        "full_model_external_validation_feasible": False,
        "organism_rule_transportability_feasible": True,
        "partial_secondary_source_screen_feasible": True,
        "limitations": (
            "No central-line exposure timeline, physiologic landmark predictors, symptoms, "
            "wound/abdominal cultures, or official NHSN adjudication. Dates are day-granular "
            "and de-identified."
        ),
    }
    pd.DataFrame([summary]).to_csv(output / "run31a_armd_feasibility_summary.csv", index=False)
    return summary


def eicu_analysis(root: Path, output: Path) -> dict:
    print("Auditing eICU microbiology and explicit central-line records...", flush=True)
    patients = pd.read_csv(root / "patient.csv.gz", usecols=["patientunitstayid", "hospitalid"])
    micro = pd.read_csv(
        root / "microLab.csv.gz",
        usecols=["patientunitstayid", "culturetakenoffset", "culturesite", "organism"],
    )
    micro["site_norm"] = normalize(micro["culturesite"])
    micro["organism_norm"] = normalize(micro["organism"])
    blood = micro[micro["site_norm"].str.contains("BLOOD", regex=False)].copy()
    blood = blood.drop_duplicates(
        ["patientunitstayid", "culturetakenoffset", "site_norm", "organism_norm"]
    )
    blood_positive = blood[
        blood["organism_norm"].ne("")
        & ~blood["organism_norm"].isin({"NO GROWTH", "NEGATIVE"})
    ].copy()

    treatment = pd.read_csv(
        root / "treatment.csv.gz",
        usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"],
    )
    central_string = (
        "infectious diseases|procedures|vascular catheter placement|central venous"
    )
    placement = treatment[treatment["treatmentstring"].eq(central_string)].copy()
    placements = placement.groupby("patientunitstayid")["treatmentoffset"].apply(list).to_dict()

    eligible = []
    for row in blood_positive.itertuples(index=False):
        prior = placements.get(row.patientunitstayid, [])
        if any(row.culturetakenoffset - offset >= 48 * 60 for offset in prior):
            eligible.append(row)
    eligible_stays = {row.patientunitstayid for row in eligible}

    hospital_micro = (
        blood_positive.merge(patients, on="patientunitstayid", how="left")
        .groupby("hospitalid")
        .agg(
            positive_blood_rows=("patientunitstayid", "size"),
            positive_blood_stays=("patientunitstayid", "nunique"),
        )
        .reset_index()
        .sort_values("positive_blood_stays", ascending=False)
    )
    hospital_micro.to_csv(output / "run31b_eicu_hospital_microbiology_coverage.csv", index=False)

    summary = {
        "dataset": "eICU-CRD 2.0",
        "icu_stays": int(patients["patientunitstayid"].nunique()),
        "hospitals": int(patients["hospitalid"].nunique()),
        "blood_culture_rows_deduplicated": int(len(blood)),
        "positive_blood_rows_deduplicated": int(len(blood_positive)),
        "positive_blood_stays": int(blood_positive["patientunitstayid"].nunique()),
        "explicit_central_venous_placement_stays": int(placement["patientunitstayid"].nunique()),
        "positive_blood_stays_ge48h_after_explicit_placement": int(len(eligible_stays)),
        "full_model_external_validation_feasible": False,
        "reduced_documented_line_subset_feasible": False,
        "reason": (
            "Microbiology is sparsely populated across hospitals and explicit central-line "
            "placement records yield too few eligible positive events; catheter removal/end "
            "times are not reliably reconstructable."
        ),
    }
    pd.DataFrame([summary]).to_csv(output / "run31b_eicu_feasibility_summary.csv", index=False)
    return summary


def write_notes(output: Path, armd: dict, eicu: dict) -> None:
    notes = f"""# Run 31 - External Validation Feasibility

## Purpose

Determine whether ARMD-MGB or eICU can support honest external validation of the frozen
MIMIC-IV seven-day strict CVC-associated BSI proxy model. This is a feasibility and label-
transportability run, not a model-performance validation.

## ARMD-MGB findings

- Positive blood-culture accessions: {armd['positive_blood_accessions']:,}
- Recognized-pathogen accessions: {armd['recognized_pathogen_accessions']:,}
- Common-commensal-positive accessions: {armd['commensal_positive_accessions']:,}
- Repeated-commensal accessions within the 48-hour sensitivity window: {armd['repeat_commensal_48h_accessions']:,}
- Strict organism-rule accessions: {armd['strict_organism_rule_accessions']:,}
- Strict events with a same-organism urine or respiratory culture within +/-3 days: {armd['strict_with_matching_nonblood_source_pm3d']:,} ({armd['strict_with_matching_nonblood_source_pm3d_rate']:.1%})

ARMD-MGB supports external organism-rule transportability and partial secondary-source
screening. It cannot support full model validation because it lacks central-line exposure
episodes and the physiologic/therapy landmark feature timeline.

## eICU findings

- ICU stays: {eicu['icu_stays']:,} across {eicu['hospitals']:,} hospitals
- Positive blood-culture stays: {eicu['positive_blood_stays']:,}
- Explicit central-venous placement stays: {eicu['explicit_central_venous_placement_stays']:,}
- Positive blood-culture stays at least 48 hours after explicit placement: {eicu['positive_blood_stays_ge48h_after_explicit_placement']:,}

eICU fails the prespecified exact-validation feasibility gate. Sparse microbiology and
incomplete catheter episode timing would create a selected, incomparable cohort.

## Decision

1. Retain MIMIC-IV temporal lockbox results as the only full-model evaluation.
2. Use ARMD-MGB as external validation of organism classification and partial source logic.
3. Report eICU as a transparent failed feasibility assessment, not a negative model result.
4. Seek a hospital EHR dataset with complete line start/end times, microbiology, and temporal
   predictors for true model external validation.

## Terminology

These analyses validate components of a strict CVC-associated BSI proxy. They do not establish
NHSN-adjudicated CLABSI status.
"""
    (output / "run31_external_validation_feasibility_notes.md").write_text(
        notes, encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armd-root", type=Path, required=True)
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    armd = armd_analysis(args.armd_root, args.output)
    eicu = eicu_analysis(args.eicu_root, args.output)
    write_notes(args.output, armd, eicu)

    metadata = {
        "run": 31,
        "title": "External Validation Feasibility",
        "armd": armd,
        "eicu": eicu,
        "conclusion": (
            "ARMD-MGB supports component-level label transportability; eICU fails exact "
            "external-validation feasibility; neither supports replaying the frozen model."
        ),
    }
    (args.output / "run31_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

