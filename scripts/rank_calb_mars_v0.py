from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WT = (
    "LPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLVPGTGTTGPQSFDSNWIPLSTQLGYTPCWISPPPFMLNDTQVNT"
    "EYMVNAITALYAGSGNNKLPVLTWSQGGLVAQWGLTFFPSIRSKVDRLMAFAPDYKGTVLAGPLDALAVSAPSVWQQTTG"
    "SALTTALRNAGGLTQIVPTTNLYSATDEIVQPQVSNSPLDSSYLFNGKNVQAQAVCGPLFVIDHAGSLTSQFSYVVGRSA"
    "LRSTTGQARSADYGITDCNPLPANDLTPEQKVAAAALLAPAAAAIVAGPKQNCEPDLMPYARPFAVGKRTCSGIVTP"
)


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    entries = []
    header = None
    seq = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq)))
    return entries


def extract_metric(header: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}=([0-9.]+)", header)
    return float(m.group(1)) if m else None


def mutations(seq: str) -> list[str]:
    muts = []
    for i, (wt, aa) in enumerate(zip(WT, seq), start=1):
        if wt != aa:
            muts.append(f"{wt}{i}{aa}")
    return muts


def mars_score(seq: str) -> tuple[float, list[str]]:
    notes = []
    score = 0.0
    aa249 = seq[248]
    aa251 = seq[250]
    aa298 = seq[297]

    if aa298 == "L":
        score += 3.0
        notes.append("best_antioxidation_298")
    elif aa298 == "I":
        score += 2.5
        notes.append("backup_antioxidation_298")
    elif aa298 == "V":
        score += 1.5
        notes.append("hydrophobe_298")
    elif aa298 == "M":
        score -= 2.0
        notes.append("keeps_surface_met_298")
    else:
        score += 0.3
        notes.append("neutral_298")

    if aa251 in {"E", "D"}:
        score += 1.5
        notes.append("surface_charge_251")
    elif aa251 in {"Q", "N", "S", "T"}:
        score += 0.5
        notes.append("hydration_friendly_251")
    elif aa251 in {"F", "W", "Y", "C", "M"}:
        score -= 1.0
        notes.append("bad_surface_251")

    if aa249 in {"L", "I", "V", "Q", "N"}:
        score += 0.6
        notes.append("acceptable_249")
    elif aa249 in {"F", "W", "Y", "C", "M"}:
        score -= 0.8
        notes.append("risky_249")

    mut_count = len(mutations(seq))
    score -= 0.15 * max(0, mut_count - 1)
    if mut_count == 1:
        notes.append("low_burden")
    return score, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path(r"D:\Codex\Work\4-12 Mars protein\mars_stack\outputs\calb_mpnn_seed\run_01\seqs\1LBT.fa"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(r"D:\Codex\Work\4-12 Mars protein\mars_stack\outputs\calb_mpnn_seed\run_01\calb_mars_ranked.csv"),
    )
    args = parser.parse_args()

    rows = []
    for header, seq in parse_fasta(args.fasta):
        if "sample=" not in header:
            continue
        mpnn_score = extract_metric(header, "score")
        global_score = extract_metric(header, "global_score")
        mut_list = mutations(seq)
        score, notes = mars_score(seq)
        rows.append(
            {
                "header": header,
                "mutations": ";".join(mut_list) if mut_list else "WT",
                "mpnn_score": mpnn_score,
                "global_score": global_score,
                "mars_score_v0": round(score, 3),
                "notes": ";".join(notes),
                "sequence": seq,
            }
        )

    rows.sort(key=lambda r: (-r["mars_score_v0"], r["mpnn_score"] if r["mpnn_score"] is not None else 999))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["mutations", "mars_score_v0", "mpnn_score", "global_score", "notes", "sequence", "header"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ranked candidates to {args.out_csv}")
    for row in rows[:10]:
        print(row["mutations"], row["mars_score_v0"], row["mpnn_score"], row["notes"])


if __name__ == "__main__":
    main()
