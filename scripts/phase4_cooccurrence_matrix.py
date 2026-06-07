import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List


PHASE4_HISTORY_PATH = Path("phase4_history") / "phase4_history.jsonl"
OUTPUT_DIR = Path("phase4_outputs")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_phase4_history(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 history file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def get_constraint_set(record: dict, include_unresolved: bool = True) -> List[str]:
    violated = record.get("violated_constraints", []) or []
    unresolved = record.get("unresolved_constraints", []) or []

    constraint_set = set(violated)

    if include_unresolved:
        constraint_set.update(unresolved)

    return sorted(constraint_set)


def build_cooccurrence_data(
    history: List[dict],
    include_unresolved: bool = True,
) -> Dict[str, object]:
    """
    Builds:
    - individual constraint frequency
    - symmetric co-occurrence counts
    - pair counts
    - per-constraint partner counts
    """

    constraint_frequency = Counter()
    pair_frequency = Counter()
    matrix = defaultdict(lambda: defaultdict(int))

    for record in history:
        constraints = get_constraint_set(record, include_unresolved=include_unresolved)

        # Count individual appearances
        for constraint in constraints:
            constraint_frequency[constraint] += 1

        # Count pairwise co-occurrence
        for a, b in combinations(constraints, 2):
            pair_frequency[(a, b)] += 1
            matrix[a][b] += 1
            matrix[b][a] += 1

        # Diagonal = total appearances
        for constraint in constraints:
            matrix[constraint][constraint] += 1

    all_constraints = sorted(constraint_frequency.keys())

    partner_counts = {}
    for constraint in all_constraints:
        partners = {
            other: matrix[constraint][other]
            for other in all_constraints
            if other != constraint and matrix[constraint][other] > 0
        }
        partner_counts[constraint] = dict(sorted(partners.items(), key=lambda x: (-x[1], x[0])))

    return {
        "all_constraints": all_constraints,
        "constraint_frequency": dict(sorted(constraint_frequency.items())),
        "pair_frequency": {
            f"{a}__{b}": count
            for (a, b), count in sorted(pair_frequency.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        },
        "matrix": {
            row: {col: matrix[row][col] for col in all_constraints}
            for row in all_constraints
        },
        "partner_counts": partner_counts,
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_matrix_csv(path: Path, all_constraints: List[str], matrix: Dict[str, Dict[str, int]]) -> None:
    lines = []

    header = ["constraint"] + all_constraints
    lines.append(",".join(header))

    for row in all_constraints:
        values = [row] + [str(matrix[row].get(col, 0)) for col in all_constraints]
        lines.append(",".join(values))

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))


def write_summary_txt(path: Path, data: dict, include_unresolved: bool) -> None:
    lines = []

    lines.append("PHASE 4 CONSTRAINT CO-OCCURRENCE SUMMARY")
    lines.append("")
    lines.append(f"Include unresolved constraints: {include_unresolved}")
    lines.append("")

    lines.append("Individual constraint frequencies:")
    for constraint, count in data["constraint_frequency"].items():
        lines.append(f"  {constraint}: {count}")

    lines.append("")
    lines.append("Top co-occurring pairs:")
    if data["pair_frequency"]:
        for pair_key, count in data["pair_frequency"].items():
            a, b = pair_key.split("__")
            lines.append(f"  {a} + {b}: {count}")
    else:
        lines.append("  None")

    lines.append("")
    lines.append("Per-constraint co-fire partners:")
    for constraint, partners in data["partner_counts"].items():
        lines.append(f"  {constraint}:")
        if partners:
            for partner, count in partners.items():
                lines.append(f"    {partner}: {count}")
        else:
            lines.append("    None")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    history = load_phase4_history(PHASE4_HISTORY_PATH)

    # Version 1: include both violated + unresolved
    include_unresolved = False
    data = build_cooccurrence_data(history, include_unresolved=include_unresolved)

    json_path = OUTPUT_DIR / "phase4_constraint_cooccurrence.json"
    csv_path = OUTPUT_DIR / "phase4_constraint_cooccurrence_matrix.csv"
    txt_path = OUTPUT_DIR / "phase4_constraint_cooccurrence_summary.txt"

    write_json(json_path, data)
    write_matrix_csv(csv_path, data["all_constraints"], data["matrix"])
    write_summary_txt(txt_path, data, include_unresolved=include_unresolved)

    print("Constraint co-occurrence analysis complete.")
    print(f"History source: {PHASE4_HISTORY_PATH}")
    print(f"JSON saved to: {json_path}")
    print(f"CSV matrix saved to: {csv_path}")
    print(f"Text summary saved to: {txt_path}")


if __name__ == "__main__":
    main()