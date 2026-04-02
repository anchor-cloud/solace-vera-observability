import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set


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


def normalized_set(values) -> Set[str]:
    if not values:
        return set()
    return {str(v).strip().upper() for v in values if str(v).strip()}


def get_all_constraints(record: dict, include_unresolved: bool = True) -> Set[str]:
    violated = normalized_set(record.get("violated_constraints", []))
    unresolved = normalized_set(record.get("unresolved_constraints", []))

    if include_unresolved:
        return violated | unresolved
    return violated


def compute_constraint_role_audit(history: List[dict]) -> Dict[str, object]:
    """
    Computes:
    - total appearances
    - solo fires
    - co-fires
    - fail appearances
    - ambiguity appearances
    - dominance-style counts:
        * solo_dominant
        * joint_dominant
        * ambiguity_support
        * mixed_support
    - per-constraint scenario lists
    """

    total_appearances = Counter()
    violated_appearances = Counter()
    unresolved_appearances = Counter()

    solo_fires = Counter()
    co_fires = Counter()

    solo_dominant = Counter()
    joint_dominant = Counter()
    ambiguity_support = Counter()
    mixed_support = Counter()

    scenario_index = defaultdict(list)

    for record in history:
        scenario_id = record.get("scenario_id", "")
        phase3_output = str(record.get("phase3_output", "")).strip().upper()

        violated = normalized_set(record.get("violated_constraints", []))
        unresolved = normalized_set(record.get("unresolved_constraints", []))
        all_constraints = violated | unresolved

        if not all_constraints:
            continue

        # Base appearances
        for c in all_constraints:
            total_appearances[c] += 1
            scenario_index[c].append(scenario_id)

        for c in violated:
            violated_appearances[c] += 1

        for c in unresolved:
            unresolved_appearances[c] += 1

        # Solo vs co-fire
        if len(all_constraints) == 1:
            only_constraint = next(iter(all_constraints))
            solo_fires[only_constraint] += 1
        else:
            for c in all_constraints:
                co_fires[c] += 1

        # Dominance-style role assignment
        if phase3_output == "ETHICAL_FAIL_CONSTRAINT_VIOLATION":
            if len(violated) == 1 and len(all_constraints) == 1:
                # Single violated constraint, no unresolved companions
                only_constraint = next(iter(violated))
                solo_dominant[only_constraint] += 1
            else:
                # Any violated constraint in a fail case gets joint_dominant credit
                for c in violated:
                    joint_dominant[c] += 1

                # Any unresolved constraint in a fail case is supporting mixed uncertainty
                for c in unresolved:
                    mixed_support[c] += 1

                # If violated constraints are present alongside others, they are also mixed-supporting
                if len(all_constraints) > 1:
                    for c in violated:
                        mixed_support[c] += 1

        elif phase3_output == "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED":
            if violated:
                # This should be rare or impossible in your current schema, but handle honestly
                for c in violated:
                    joint_dominant[c] += 1
                for c in unresolved:
                    mixed_support[c] += 1
            else:
                # Pure ambiguity case
                if len(unresolved) == 1:
                    only_constraint = next(iter(unresolved))
                    ambiguity_support[only_constraint] += 1
                else:
                    for c in unresolved:
                        ambiguity_support[c] += 1
                        mixed_support[c] += 1

        # ETHICAL_PASS produces no role counts because there are no triggered constraints

    all_constraints_sorted = sorted(total_appearances.keys())

    per_constraint = {}
    for c in all_constraints_sorted:
        total = total_appearances[c]
        per_constraint[c] = {
            "total_appearances": total,
            "violated_appearances": violated_appearances[c],
            "unresolved_appearances": unresolved_appearances[c],
            "solo_fires": solo_fires[c],
            "co_fires": co_fires[c],
            "solo_dominant": solo_dominant[c],
            "joint_dominant": joint_dominant[c],
            "ambiguity_support": ambiguity_support[c],
            "mixed_support": mixed_support[c],
            "solo_fire_rate_pct": round((solo_fires[c] / total) * 100.0, 2) if total else 0.0,
            "co_fire_rate_pct": round((co_fires[c] / total) * 100.0, 2) if total else 0.0,
            "scenario_ids": scenario_index[c],
        }

    return {
        "total_records_analyzed": len(history),
        "constraints_analyzed": all_constraints_sorted,
        "per_constraint": per_constraint,
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, audit: Dict[str, object]) -> None:
    constraints = audit["constraints_analyzed"]
    per_constraint = audit["per_constraint"]

    header = [
        "constraint_id",
        "total_appearances",
        "violated_appearances",
        "unresolved_appearances",
        "solo_fires",
        "co_fires",
        "solo_dominant",
        "joint_dominant",
        "ambiguity_support",
        "mixed_support",
        "solo_fire_rate_pct",
        "co_fire_rate_pct",
    ]

    lines = [",".join(header)]

    for c in constraints:
        row = per_constraint[c]
        values = [
            c,
            str(row["total_appearances"]),
            str(row["violated_appearances"]),
            str(row["unresolved_appearances"]),
            str(row["solo_fires"]),
            str(row["co_fires"]),
            str(row["solo_dominant"]),
            str(row["joint_dominant"]),
            str(row["ambiguity_support"]),
            str(row["mixed_support"]),
            str(row["solo_fire_rate_pct"]),
            str(row["co_fire_rate_pct"]),
        ]
        lines.append(",".join(values))

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))


def write_summary_txt(path: Path, audit: Dict[str, object]) -> None:
    lines = []

    lines.append("PHASE 4 CONSTRAINT ROLE AUDIT")
    lines.append("")
    lines.append(f"Total records analyzed: {audit['total_records_analyzed']}")
    lines.append("")

    for c in audit["constraints_analyzed"]:
        row = audit["per_constraint"][c]
        lines.append(f"{c}")
        lines.append(f"  Total appearances: {row['total_appearances']}")
        lines.append(f"  Violated appearances: {row['violated_appearances']}")
        lines.append(f"  Unresolved appearances: {row['unresolved_appearances']}")
        lines.append(f"  Solo fires: {row['solo_fires']}")
        lines.append(f"  Co-fires: {row['co_fires']}")
        lines.append(f"  Solo dominant: {row['solo_dominant']}")
        lines.append(f"  Joint dominant: {row['joint_dominant']}")
        lines.append(f"  Ambiguity support: {row['ambiguity_support']}")
        lines.append(f"  Mixed support: {row['mixed_support']}")
        lines.append(f"  Solo fire rate: {row['solo_fire_rate_pct']}%")
        lines.append(f"  Co-fire rate: {row['co_fire_rate_pct']}%")
        lines.append(f"  Scenario IDs: {', '.join(row['scenario_ids']) if row['scenario_ids'] else 'None'}")
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    history = load_phase4_history(PHASE4_HISTORY_PATH)
    audit = compute_constraint_role_audit(history)

    json_path = OUTPUT_DIR / "phase4_constraint_role_audit.json"
    csv_path = OUTPUT_DIR / "phase4_constraint_role_audit.csv"
    txt_path = OUTPUT_DIR / "phase4_constraint_role_audit_summary.txt"

    write_json(json_path, audit)
    write_csv(csv_path, audit)
    write_summary_txt(txt_path, audit)

    print("Constraint role audit complete.")
    print(f"History source: {PHASE4_HISTORY_PATH}")
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Text summary saved to: {txt_path}")


if __name__ == "__main__":
    main()