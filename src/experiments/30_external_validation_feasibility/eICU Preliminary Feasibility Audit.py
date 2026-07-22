import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path


LINE_PATTERN = re.compile(
    r"central.{0,40}(line|catheter)|"
    r"(line|catheter).{0,40}central|"
    r"picc|swan.?ganz|pulmonary artery catheter|"
    r"vascular catheter|line care",
    re.IGNORECASE,
)


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def scan_structured_line_documentation(root, filename, columns, offset_column):
    value_counts = Counter()
    stay_offsets = defaultdict(list)

    for row in read_rows(root / filename):
        text = " | ".join(row.get(column, "") for column in columns)
        if not LINE_PATTERN.search(text):
            continue

        normalized = " | ".join(row.get(column, "")[:120] for column in columns)
        value_counts[normalized] += 1
        try:
            offset = int(row[offset_column])
        except (KeyError, TypeError, ValueError):
            continue
        stay_offsets[row["patientunitstayid"]].append(offset)

    return value_counts, stay_offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eicu_root", type=Path)
    args = parser.parse_args()
    root = args.eicu_root

    stay_hospital = {}
    hospital_stays = Counter()
    for row in read_rows(root / "patient.csv.gz"):
        stay_id = row["patientunitstayid"]
        hospital_id = row["hospitalid"]
        stay_hospital[stay_id] = hospital_id
        hospital_stays[hospital_id] += 1

    print(f"Patient stays: {len(stay_hospital):,}", flush=True)
    print(f"Hospitals: {len(hospital_stays):,}", flush=True)

    blood_events = set()
    positive_blood_events = set()
    culture_sites = Counter()
    organisms = Counter()
    for row in read_rows(root / "microLab.csv.gz"):
        culture_site = row["culturesite"].strip()
        organism = row["organism"].strip().lower()
        culture_sites[culture_site] += 1
        organisms[organism] += 1

        if "blood" not in culture_site.lower():
            continue

        event = (
            row["patientunitstayid"],
            int(row["culturetakenoffset"]),
            culture_site,
            organism,
        )
        blood_events.add(event)
        if organism not in {"", "no growth"}:
            positive_blood_events.add(event)

    print(f"Deduplicated blood-culture events: {len(blood_events):,}", flush=True)
    print(
        f"Positive-organism blood events: {len(positive_blood_events):,}",
        flush=True,
    )

    scans = [
        (
            "note.csv.gz",
            ["notetype", "notepath", "notevalue", "notetext"],
            "noteoffset",
        ),
        (
            "nurseCare.csv.gz",
            ["celllabel", "cellattributepath", "cellattribute", "cellattributevalue"],
            "nursecareoffset",
        ),
        (
            "nurseAssessment.csv.gz",
            ["cellattributepath", "celllabel", "cellattribute", "cellattributevalue"],
            "nurseassessoffset",
        ),
    ]

    documented_offsets = defaultdict(list)
    scan_results = {}
    for filename, columns, offset_column in scans:
        counts, offsets = scan_structured_line_documentation(
            root, filename, columns, offset_column
        )
        scan_results[filename] = counts
        for stay_id, values in offsets.items():
            documented_offsets[stay_id].extend(values)
        print(
            f"Scanned {filename}: {sum(counts.values()):,} matching rows, "
            f"{len(offsets):,} stays",
            flush=True,
        )

    explicit_placements = defaultdict(list)
    treatment_counts = Counter()
    for row in read_rows(root / "treatment.csv.gz"):
        text = row["treatmentstring"]
        if not LINE_PATTERN.search(text):
            continue
        treatment_counts[text] += 1
        if text.endswith("vascular catheter placement|central venous"):
            explicit_placements[row["patientunitstayid"]].append(
                int(row["treatmentoffset"])
            )

    def eligible_events(offsets):
        return [
            event
            for event in positive_blood_events
            if event[0] in offsets
            and any(event[1] - offset >= 48 * 60 for offset in offsets[event[0]])
        ]

    explicit_eligible = eligible_events(explicit_placements)
    broad_eligible = eligible_events(documented_offsets)

    print(f"Explicit central-venous placement stays: {len(explicit_placements):,}")
    print(f"Broad structured line-documentation stays: {len(documented_offsets):,}")
    print(
        "Positive events >=48h after explicit placement: "
        f"{len(explicit_eligible):,} events / "
        f"{len({event[0] for event in explicit_eligible}):,} stays"
    )
    print(
        "Positive events >=48h after broad line documentation: "
        f"{len(broad_eligible):,} events / "
        f"{len({event[0] for event in broad_eligible}):,} stays"
    )

    print("\nTop treatment line strings")
    for text, count in treatment_counts.most_common(30):
        print(f"{count:>8,} | {text}")

    for filename, counts in scan_results.items():
        print(f"\n{filename}: {sum(counts.values()):,} rows, {len(counts):,} strings")
        for text, count in counts.most_common(30):
            print(f"{count:>8,} | {text[:500]}")


if __name__ == "__main__":
    main()

