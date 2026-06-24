"""Analyze a completed MOC (Midline Output Collapse) evidence run.

Usage:
    python analyze_moc_evidence.py [--run-dir DIR] [--out-csv FILE]

Defaults to the most recent moc_evidence run under pipeline_outputs/ if
--run-dir is omitted. Reads every MOC-*.json in the run dir, extracts:
    - scenario_id
    - csv_intended    (author's designed risk fields)
    - raw_risk_fields (live model's output with calibration disabled)
and computes:

    1. Overall collapse distribution of the model's raw risk output
       across all scenarios and all four risk dimensions.
    2. Per-dimension distribution (uncertainty / potential_harm /
       irreversibility / time_pressure separately).
    3. Distribution conditioned on each intended risk class (the author's
       classification of the scenario as LOW / MEDIUM / HIGH).
    4. Per-scenario collapse pattern (all-MEDIUM, all-LOW, all-HIGH, mixed)
       and agreement with intended class.

Outputs:
    - A formatted summary table on stdout.
    - moc_evidence_analysis.csv with one row per scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


RISK_FIELDS = ("uncertainty", "potential_harm", "irreversibility", "time_pressure")
LEVELS = ("LOW", "MEDIUM", "HIGH")


# ---------------------------------------------------------------------------
# Intended-class classifier (matches how moc_evidence_pack.csv was authored)
# ---------------------------------------------------------------------------
def classify_intended(intended: dict) -> str:
    """Classify a scenario as LOW / MEDIUM / HIGH based on csv_intended.

    - HIGH if potential_harm == HIGH
    - LOW  if uncertainty, potential_harm, irreversibility are all LOW
    - MEDIUM otherwise
    """
    ph = (intended.get("potential_harm") or "").upper()
    if ph == "HIGH":
        return "HIGH"
    u = (intended.get("uncertainty") or "").upper()
    i = (intended.get("irreversibility") or "").upper()
    if ph == "LOW" and u == "LOW" and i == "LOW":
        return "LOW"
    return "MEDIUM"


def collapse_pattern(raw: dict) -> str:
    """Classify the model's raw output pattern across the 4 risk dimensions."""
    vals = [(raw.get(f) or "").upper() for f in RISK_FIELDS]
    if not all(vals):
        return "incomplete"
    if all(v == "MEDIUM" for v in vals):
        return "all_MEDIUM"
    if all(v == "LOW" for v in vals):
        return "all_LOW"
    if all(v == "HIGH" for v in vals):
        return "all_HIGH"
    return "mixed"


def three_core_pattern(raw: dict) -> str:
    """Same as collapse_pattern but ignoring time_pressure.

    Useful for the canonical MOC signature: the three risk dimensions
    (uncertainty / potential_harm / irreversibility) collapsing to MEDIUM
    while the model correctly tags time_pressure separately.
    """
    core = [(raw.get(f) or "").upper() for f in (
        "uncertainty", "potential_harm", "irreversibility"
    )]
    if not all(core):
        return "incomplete"
    if all(v == "MEDIUM" for v in core):
        return "core3_all_MEDIUM"
    if all(v == "LOW" for v in core):
        return "core3_all_LOW"
    if all(v == "HIGH" for v in core):
        return "core3_all_HIGH"
    return "core3_mixed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class ScenarioRow:
    scenario_id: str
    status: str
    intended: dict
    raw: dict
    intended_class: str
    pattern_all4: str
    pattern_core3: str
    raw_matches_intended_class: bool
    notes: str


def _load_one(path: Path) -> ScenarioRow | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARN: could not parse {path.name}: {exc}", file=sys.stderr)
        return None

    scenario_id = data.get("scenario_id") or path.stem
    status = data.get("status", "unknown")
    intended = data.get("csv_intended") or {}
    raw = data.get("raw_risk_fields") or {}
    notes = data.get("csv_notes") or ""

    intended_class = classify_intended(intended)
    pat4 = collapse_pattern(raw)
    pat3 = three_core_pattern(raw)

    # Does the raw output reflect the intended class?
    # - LOW-intended  : raw core3 all LOW is a match
    # - HIGH-intended : raw has at least one HIGH on core3
    # - MEDIUM-intended : anything that isn't collapsed LOW or HIGH is a match
    raw_core = [(raw.get(f) or "").upper() for f in (
        "uncertainty", "potential_harm", "irreversibility"
    )]
    if intended_class == "LOW":
        matches = all(v == "LOW" for v in raw_core)
    elif intended_class == "HIGH":
        matches = any(v == "HIGH" for v in raw_core)
    else:  # MEDIUM
        matches = not (all(v == "LOW" for v in raw_core) or all(v == "HIGH" for v in raw_core))

    return ScenarioRow(
        scenario_id=scenario_id,
        status=status,
        intended=intended,
        raw=raw,
        intended_class=intended_class,
        pattern_all4=pat4,
        pattern_core3=pat3,
        raw_matches_intended_class=matches,
        notes=notes,
    )


def load_run(run_dir: Path) -> tuple[list[ScenarioRow], list[Path]]:
    """Return (parsed_rows, skipped_paths).

    Skipped = files that either couldn't parse or don't have the raw fields
    we need (e.g. pipeline_failed or model_call_failed where the model call
    errored before returning risk fields).
    """
    paths = sorted(run_dir.glob("MOC-*.json"))
    rows: list[ScenarioRow] = []
    skipped: list[Path] = []
    for p in paths:
        row = _load_one(p)
        if row is None:
            skipped.append(p)
            continue
        # Scenario must have at least a plausible raw_risk_fields block.
        if not any((row.raw.get(f) or "").strip() for f in RISK_FIELDS):
            print(f"  NOTE: {row.scenario_id} has no raw_risk_fields "
                  f"(status={row.status}) - excluding from distributions",
                  file=sys.stderr)
            skipped.append(p)
            continue
        rows.append(row)
    return rows, skipped


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def level_distribution(values: list[str]) -> dict[str, int]:
    c = Counter(v.upper() for v in values if v)
    return {lvl: c.get(lvl, 0) for lvl in LEVELS}


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def counts_row(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    return (
        f"LOW={counts['LOW']:>3} ({pct(counts['LOW'], total):>6})  "
        f"MEDIUM={counts['MEDIUM']:>3} ({pct(counts['MEDIUM'], total):>6})  "
        f"HIGH={counts['HIGH']:>3} ({pct(counts['HIGH'], total):>6})"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(rows: list[ScenarioRow], skipped: list[Path]) -> None:
    total_scenarios = len(rows)
    total_datapoints = total_scenarios * len(RISK_FIELDS)

    # 1. Overall raw-output distribution (all 4 dimensions pooled)
    pooled = [(r.raw.get(f) or "") for r in rows for f in RISK_FIELDS]
    overall = level_distribution(pooled)

    # 2. Per-dimension distribution
    per_dim: dict[str, dict[str, int]] = {}
    for f in RISK_FIELDS:
        per_dim[f] = level_distribution([(r.raw.get(f) or "") for r in rows])

    # 3. Distribution by intended class (pooled across 4 dims)
    classes = ("LOW", "MEDIUM", "HIGH")
    by_class_pooled: dict[str, dict[str, int]] = {}
    by_class_counts: dict[str, int] = {c: 0 for c in classes}
    for c in classes:
        rows_c = [r for r in rows if r.intended_class == c]
        by_class_counts[c] = len(rows_c)
        by_class_pooled[c] = level_distribution(
            [(r.raw.get(f) or "") for r in rows_c for f in RISK_FIELDS]
        )

    # 3b. Per-dimension distribution inside each intended class
    by_class_dim: dict[str, dict[str, dict[str, int]]] = {}
    for c in classes:
        rows_c = [r for r in rows if r.intended_class == c]
        by_class_dim[c] = {
            f: level_distribution([(r.raw.get(f) or "") for r in rows_c])
            for f in RISK_FIELDS
        }

    # 4. Per-scenario collapse patterns
    patt_all4 = Counter(r.pattern_all4 for r in rows)
    patt_core3 = Counter(r.pattern_core3 for r in rows)

    # 5. Agreement with intended class
    agree = sum(1 for r in rows if r.raw_matches_intended_class)

    # ---- output ----
    print()
    print("=" * 78)
    print("MOC EVIDENCE ANALYSIS")
    print("=" * 78)
    print(f"Scenarios analyzed       : {total_scenarios}")
    print(f"Raw data points          : {total_datapoints} "
          f"({total_scenarios} scenarios x 4 risk dimensions)")
    print(f"Skipped (no raw fields)  : {len(skipped)}")
    print()

    # Overall
    print("-" * 78)
    print("1. OVERALL RAW-OUTPUT DISTRIBUTION (pooled across 4 dimensions)")
    print("-" * 78)
    print("   " + counts_row(overall))
    print()

    # Per dimension
    print("-" * 78)
    print("2. DISTRIBUTION PER RISK DIMENSION")
    print("-" * 78)
    for f in RISK_FIELDS:
        print(f"   {f:<18}: " + counts_row(per_dim[f]))
    print()

    # By intended class
    print("-" * 78)
    print("3. RAW DISTRIBUTION CONDITIONED ON INTENDED RISK CLASS")
    print("-" * 78)
    print("   Intended class = author's LOW / MEDIUM / HIGH label for the scenario.")
    print("   The pooled row counts all four risk dimensions for scenarios in that class.")
    print()
    for c in classes:
        n = by_class_counts[c]
        print(f"   Intended {c:<6} (N={n:>2} scenarios, {n * 4:>3} data points):")
        print("       POOLED            : " + counts_row(by_class_pooled[c]))
        for f in RISK_FIELDS:
            print(f"       {f:<18}: " + counts_row(by_class_dim[c][f]))
        print()

    # Per-scenario collapse patterns
    print("-" * 78)
    print("4. PER-SCENARIO COLLAPSE PATTERNS")
    print("-" * 78)
    print("   Across all 4 risk dimensions (unc/harm/irrev/tp):")
    for pat in ("all_MEDIUM", "all_LOW", "all_HIGH", "mixed", "incomplete"):
        n = patt_all4.get(pat, 0)
        if n:
            print(f"     {pat:<12}: {n:>3} ({pct(n, total_scenarios)})")
    print()
    print("   Across the 3 CORE risk dimensions (unc/harm/irrev, ignoring time_pressure):")
    for pat in ("core3_all_MEDIUM", "core3_all_LOW", "core3_all_HIGH", "core3_mixed", "incomplete"):
        n = patt_core3.get(pat, 0)
        if n:
            print(f"     {pat:<18}: {n:>3} ({pct(n, total_scenarios)})")
    print()

    # Agreement
    print("-" * 78)
    print("5. AGREEMENT WITH AUTHOR-INTENDED CLASS")
    print("-" * 78)
    print(f"   Raw-output class matches intended class : {agree} / {total_scenarios} "
          f"({pct(agree, total_scenarios)})")
    print(f"   Disagreement                             : "
          f"{total_scenarios - agree} / {total_scenarios} "
          f"({pct(total_scenarios - agree, total_scenarios)})")
    print()

    # Breakdown of disagreement by intended class
    print("   Disagreement breakdown by intended class:")
    for c in classes:
        rows_c = [r for r in rows if r.intended_class == c]
        n = len(rows_c)
        dis = sum(1 for r in rows_c if not r.raw_matches_intended_class)
        print(f"     Intended {c:<6}: {dis:>3} / {n:>3} mislabeled by model "
              f"({pct(dis, n)})")
    print()

    # Headline number
    n_moc = patt_core3.get("core3_all_MEDIUM", 0)
    print("=" * 78)
    print("HEADLINE: MOC (Midline Output Collapse) rate")
    print("=" * 78)
    print(f"  Scenarios where the live model returned MEDIUM on all three")
    print(f"  core risk dimensions (uncertainty, potential_harm, irreversibility):")
    print(f"    {n_moc} / {total_scenarios}  ({pct(n_moc, total_scenarios)})")
    print()

    # Also: per intended class, MOC rate
    print("  MOC rate by intended class (fraction of class that collapsed to core3 MEDIUM):")
    for c in classes:
        rows_c = [r for r in rows if r.intended_class == c]
        n = len(rows_c)
        moc_c = sum(1 for r in rows_c if r.pattern_core3 == "core3_all_MEDIUM")
        print(f"    Intended {c:<6} (N={n:>2}): {moc_c:>3} collapsed to MEDIUM "
              f"({pct(moc_c, n)})")
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_csv(rows: list[ScenarioRow], out_path: Path) -> None:
    cols = [
        "scenario_id",
        "intended_class",
        "intended_uncertainty",
        "intended_potential_harm",
        "intended_irreversibility",
        "intended_time_pressure",
        "raw_uncertainty",
        "raw_potential_harm",
        "raw_irreversibility",
        "raw_time_pressure",
        "collapse_pattern_all4",
        "collapse_pattern_core3",
        "raw_matches_intended_class",
        "status",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "scenario_id": r.scenario_id,
                "intended_class": r.intended_class,
                "intended_uncertainty":     r.intended.get("uncertainty", ""),
                "intended_potential_harm":  r.intended.get("potential_harm", ""),
                "intended_irreversibility": r.intended.get("irreversibility", ""),
                "intended_time_pressure":   r.intended.get("time_pressure", ""),
                "raw_uncertainty":     r.raw.get("uncertainty", ""),
                "raw_potential_harm":  r.raw.get("potential_harm", ""),
                "raw_irreversibility": r.raw.get("irreversibility", ""),
                "raw_time_pressure":   r.raw.get("time_pressure", ""),
                "collapse_pattern_all4":  r.pattern_all4,
                "collapse_pattern_core3": r.pattern_core3,
                "raw_matches_intended_class": r.raw_matches_intended_class,
                "status": r.status,
                "notes": r.notes,
            })


# ---------------------------------------------------------------------------
# Auto-detect latest run dir
# ---------------------------------------------------------------------------
def _latest_moc_run_dir() -> Path | None:
    base = Path("pipeline_outputs")
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("moc_evidence_*"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="analyze_moc_evidence",
        description=(
            "Analyze a MOC (Midline Output Collapse) evidence run produced "
            "by run_moc_evidence.py. Reads MOC-*.json files, prints a "
            "publishable summary, and writes a per-scenario CSV."
        ),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory (default: most recent pipeline_outputs/moc_evidence_*).",
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("moc_evidence_analysis.csv"),
        help="Per-scenario CSV output path (default: moc_evidence_analysis.csv).",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir or _latest_moc_run_dir()
    if run_dir is None or not run_dir.is_dir():
        print(f"ERROR: run dir not found: {args.run_dir}", file=sys.stderr)
        return 1

    print(f"[analyze_moc_evidence] Run directory : {run_dir}")
    print(f"[analyze_moc_evidence] CSV output    : {args.out_csv}")

    rows, skipped = load_run(run_dir)
    if not rows:
        print("ERROR: no analyzable MOC-*.json files found.", file=sys.stderr)
        return 1

    print_report(rows, skipped)
    write_csv(rows, args.out_csv)
    print(f"Per-scenario CSV written: {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
