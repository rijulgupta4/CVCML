import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def normalized_counts(path, column, chunksize=250_000):
    counts = Counter()
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize, low_memory=False):
        values = chunk[column].fillna("").astype(str).str.strip().str.upper()
        counts.update(values.value_counts().to_dict())
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    micro_path = args.root / "microbiology_cohort_deid_tj_updated.csv"
    procedure_path = args.root / "prior_procedures_deid_tj.csv"

    culture_rows = 0
    accessions = set()
    patients = set()
    encounters = set()
    blood_accessions = set()
    positive_blood_accessions = set()
    blood_patients = set()
    positive_blood_patients = set()
    culture_counts = Counter()
    blood_organisms = Counter()

    columns = [
        "anon_id",
        "pat_enc_csn_id_coded",
        "order_proc_id_coded",
        "culture_description",
        "organism",
        "neg_cx",
    ]
    for chunk in pd.read_csv(
        micro_path, usecols=columns, chunksize=250_000, low_memory=False
    ):
        culture_rows += len(chunk)
        chunk["culture"] = (
            chunk["culture_description"].fillna("").astype(str).str.strip().str.upper()
        )
        chunk["organism_norm"] = (
            chunk["organism"].fillna("").astype(str).str.strip().str.upper()
        )
        accessions.update(chunk["order_proc_id_coded"].dropna().astype(str))
        patients.update(chunk["anon_id"].dropna().astype(str))
        encounters.update(chunk["pat_enc_csn_id_coded"].dropna().astype(str))
        culture_counts.update(chunk["culture"].value_counts().to_dict())

        blood = chunk[chunk["culture"].str.contains("BLOOD", regex=False)].copy()
        blood_accessions.update(blood["order_proc_id_coded"].dropna().astype(str))
        blood_patients.update(blood["anon_id"].dropna().astype(str))
        positive = blood[
            blood["organism_norm"].ne("")
            & ~blood["organism_norm"].isin({"NO GROWTH", "NEGATIVE"})
        ]
        positive_blood_accessions.update(
            positive["order_proc_id_coded"].dropna().astype(str)
        )
        positive_blood_patients.update(positive["anon_id"].dropna().astype(str))
        blood_organisms.update(positive["organism_norm"].value_counts().to_dict())

    procedure_counts = normalized_counts(procedure_path, "procedure_description")
    line_procedures = Counter(
        {
            name: count
            for name, count in procedure_counts.items()
            if any(term in name.lower() for term in ("central", "catheter", "picc", "line"))
        }
    )

    print(f"Microbiology rows: {culture_rows:,}")
    print(f"Unique accessions: {len(accessions):,}")
    print(f"Unique encounters: {len(encounters):,}")
    print(f"Unique patients: {len(patients):,}")
    print(f"Blood accessions: {len(blood_accessions):,}")
    print(f"Blood-culture patients: {len(blood_patients):,}")
    print(f"Positive blood accessions: {len(positive_blood_accessions):,}")
    print(f"Positive-blood patients: {len(positive_blood_patients):,}")

    print("\nTop culture descriptions")
    for name, count in culture_counts.most_common(25):
        print(f"{count:>10,} | {name}")

    print("\nTop positive blood organisms")
    for name, count in blood_organisms.most_common(30):
        print(f"{count:>10,} | {name}")

    print("\nCatheter/line procedure-history categories")
    if line_procedures:
        for name, count in line_procedures.most_common(30):
            print(f"{count:>10,} | {name}")
    else:
        print("None")


if __name__ == "__main__":
    main()

