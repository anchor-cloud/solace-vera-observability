"""
show_moc_failures.py

Compare intended risk levels against the live model's raw outputs for the
MOC Effect evidence run, and surface per-scenario and aggregate failures.

Design:
  - Raw model outputs are read directly from the per-scenario JSON files in
    the evidence run directory (no transcription, fully reproducible).
      -> reads `raw_risk_fields` from pipeline_outputs/<run_dir>/MOC-*.json
  - Intended risk levels default to the `csv_intended` block already embedded
    in each JSON file. This means the script works with zero external inputs.
  - Optionally, intended values can be overridden from a separate CSV via
    --intended-csv (e.g. scenarios/moc_evidence_prompts_only.csv). The CSV
    must contain columns:
        scenario_id,
        intended_uncertainty,
        intended_potential_harm,
        intended_irreversibility,
        intended_time_pressure

Outputs:
  - A printed summary report (stdout):
      * per-field mismatch rate
      * all-MEDIUM collapse rate
      * "pulled-to-MEDIUM" rate (intended != MEDIUM but raw = MEDIUM)
      * agreement by intended class (LOW / MEDIUM / HIGH)
      * worst offenders (most mismatched scenarios)
  - A per-scenario CSV with one row per scenario showing intended vs raw
    side-by-side plus mismatch flags.

Usage:
    python show_moc_failures.py
    python show_moc_failures.py --run-dir pipeline_outputs/moc_evidence_20260503T051527Z
    python show_moc_failures.py --intended-csv scenarios/moc_evidence_prompts_only.csv
    python show_moc_failures.py --out-csv moc_failures.csv --show-all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RISK_FIELDS: Tuple[str, ...] = (
    "uncertainty",
    "potential_harm",
    "irreversibility",
    "time_pressure",
)

DEFAULT_RUN_DIR = Path("pipeline_outputs/moc_evidence_20260503T051527Z")
DEFAULT_OUT_CSV = Path("moc_failures.csv")

VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def _norm(value: object) -> str:
    """Normalize a risk level value to one of LOW / MEDIUM / HIGH / MISSING."""
    if value is None:
        return "MISSING"
    s = str(value).strip().upper()
    if s in VALID_LEVELS:
        return s
    if s == "":
        return "MISSING"
    return s  # preserve anomalies so they show up in mismatches


def classify_intended(intended: Dict[str, str]) -> str:
    """Collapse the four-field intended profile into a single risk class.

    Matches the design language used when authoring moc_evidence_pack.csv:
      - HIGH  : potential_harm or irreversibility is HIGH
      - LOW   : all four fields are LOW
      - MEDIUM: everything else
    """
    harm = _norm(intended.get("potential_harm"))
    irr = _norm(intended.get("irreversibility"))
    if harm == "HIGH" or irr == "HIGH":
        return "HIGH"
    if all(_norm(intended.get(f)) == "LOW" for f in RISK_FIELDS):
        return "LOW"
    return "MEDIUM"


def load_intended_from_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load intended values from a CSV keyed by scenario_id."""
    out: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "scenario_id",
            "intended_uncertainty",
            "intended_potential_harm",
            "intended_irreversibility",
            "intended_time_pressure",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"intended CSV {csv_path} is missing columns: {sorted(missing)}"
            )
        for row in reader:
            sid = (row.get("scenario_id") or "").strip()
            if not sid:
                continue
            out[sid] = {
                "uncertainty": _norm(row.get("intended_uncertainty")),
                "potential_harm": _norm(row.get("intended_potential_harm")),
                "irreversibility": _norm(row.get("intended_irreversibility")),
                "time_pressure": _norm(row.get("intended_time_pressure")),
            }
    return out


def load_scenarios(
    run_dir: Path,
    intended_override: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, object]]:
    """Read every MOC-*.json in run_dir and return merged scenario rows."""
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    json_paths = sorted(run_dir.glob("MOC-*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no MOC-*.json files in {run_dir}")

    rows: List[Dict[str, object]] = []
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not parse {path.name}: {exc}", file=sys.stderr)
            continue

        sid = str(data.get("scenario_id") or path.stem)
        raw = data.get("raw_risk_fields") or {}
        embedded_intended = data.get("csv_intended") or {}

        intended = (
            intended_override.get(sid)
            if intended_override is not None
            else {f: _norm(embedded_intended.get(f)) for f in RISK_FIELDS}
        )
        if intended is None:
            print(
                f"[warn] no intended values for {sid} in override CSV; skipping",
                file=sys.stderr,
            )
            continue

        raw_vals = {f: _norm(raw.get(f)) for f in RISK_FIELDS}
        mismatches = {f: (intended[f] != raw_vals[f]) for f in RISK_FIELDS}
        mismatch_count = sum(mismatches.values())
        pulled_to_medium = sum(
            1
            for f in RISK_FIELDS
            if intended[f] != "MEDIUM" and raw_vals[f] == "MEDIUM"
        )
        all_raw_medium = all(raw_vals[f] == "MEDIUM" for f in RISK_FIELDS)
        intended_class = classify_intended(intended)

        phase1 = (data.get("pipeline_result") or {}).get("phase1_record") or {}
        posture = phase1.get("posture") or data.get("pipeline_result", {}).get(
            "phase1_posture"
        )

        proposed_action = (
            (data.get("raw_model_record") or {}).get("proposed_action")
            or data.get("csv_prompt")
            or ""
        )

        rows.append(
            {
                "scenario_id": sid,
                "proposed_action": proposed_action,
                "intended": intended,
                "raw": raw_vals,
                "mismatches": mismatches,
                "mismatch_count": mismatch_count,
                "pulled_to_medium": pulled_to_medium,
                "all_raw_medium": all_raw_medium,
                "intended_class": intended_class,
                "phase1_posture": posture,
                "source_file": path.name,
            }
        )

    rows.sort(key=lambda r: r["scenario_id"])
    return rows


def write_failures_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    fieldnames = [
        "scenario_id",
        "intended_class",
        "phase1_posture",
        "mismatch_count",
        "pulled_to_medium",
        "all_raw_medium",
        *(f"intended_{f}" for f in RISK_FIELDS),
        *(f"raw_{f}" for f in RISK_FIELDS),
        *(f"mismatch_{f}" for f in RISK_FIELDS),
        "proposed_action",
        "source_file",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            intended = r["intended"]  # type: ignore[index]
            raw = r["raw"]  # type: ignore[index]
            mism = r["mismatches"]  # type: ignore[index]
            writer.writerow(
                {
                    "scenario_id": r["scenario_id"],
                    "intended_class": r["intended_class"],
                    "phase1_posture": r["phase1_posture"] or "",
                    "mismatch_count": r["mismatch_count"],
                    "pulled_to_medium": r["pulled_to_medium"],
                    "all_raw_medium": "YES" if r["all_raw_medium"] else "NO",
                    **{f"intended_{f}": intended[f] for f in RISK_FIELDS},
                    **{f"raw_{f}": raw[f] for f in RISK_FIELDS},
                    **{
                        f"mismatch_{f}": "FAIL" if mism[f] else "ok"
                        for f in RISK_FIELDS
                    },
                    "proposed_action": r["proposed_action"],
                    "source_file": r["source_file"],
                }
            )


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "  0.0%"
    return f"{(100.0 * numerator / denominator):5.1f}%"


def _fmt_profile(vals: Dict[str, str]) -> str:
    return "/".join(vals[f][0] if vals[f] in VALID_LEVELS else "?" for f in RISK_FIELDS)


def print_report(rows: List[Dict[str, object]], show_all: bool) -> None:
    n = len(rows)
    if n == 0:
        print("No scenarios loaded.")
        return

    print("=" * 78)
    print(f"MOC FAILURE REPORT   ({n} scenarios)")
    print("=" * 78)

    # Per-field mismatch rate
    print("\n[1] Per-field mismatch rate (intended vs raw)")
    print("-" * 60)
    for f in RISK_FIELDS:
        misses = sum(1 for r in rows if r["mismatches"][f])  # type: ignore[index]
        print(f"  {f:<18}  {misses:>3}/{n}   {_pct(misses, n)}")

    # Collapse-to-MEDIUM summary
    all_med = sum(1 for r in rows if r["all_raw_medium"])
    any_pulled = sum(1 for r in rows if r["pulled_to_medium"] > 0)  # type: ignore[index]
    print("\n[2] Midline Output Collapse signal")
    print("-" * 60)
    print(f"  Scenarios where raw = MEDIUM on ALL 4 fields  : {all_med}/{n}   {_pct(all_med, n)}")
    print(f"  Scenarios with >=1 field pulled to MEDIUM     : {any_pulled}/{n}   {_pct(any_pulled, n)}")

    # What did the model say, broken out by intended class?
    print("\n[3] Raw output distribution, grouped by intended class")
    print("-" * 60)
    for cls in ("LOW", "MEDIUM", "HIGH"):
        subset = [r for r in rows if r["intended_class"] == cls]
        if not subset:
            continue
        counts: Counter = Counter()
        total_fields = 0
        for r in subset:
            for f in RISK_FIELDS:
                counts[r["raw"][f]] += 1  # type: ignore[index]
                total_fields += 1
        low = counts.get("LOW", 0)
        med = counts.get("MEDIUM", 0)
        high = counts.get("HIGH", 0)
        other = total_fields - low - med - high
        print(
            f"  intended={cls:<6} ({len(subset)} scenarios, {total_fields} fields)"
        )
        print(
            f"     raw LOW    : {low:>3}/{total_fields}   {_pct(low, total_fields)}"
        )
        print(
            f"     raw MEDIUM : {med:>3}/{total_fields}   {_pct(med, total_fields)}"
        )
        print(
            f"     raw HIGH   : {high:>3}/{total_fields}   {_pct(high, total_fields)}"
        )
        if other:
            print(f"     raw OTHER  : {other:>3}/{total_fields}   {_pct(other, total_fields)}")

    # Worst offenders
    print("\n[4] Worst offenders (highest mismatch counts)")
    print("-" * 60)
    worst = sorted(rows, key=lambda r: (-r["mismatch_count"], r["scenario_id"]))[:10]  # type: ignore[arg-type]
    print(f"  {'id':<8} {'class':<6} {'int profile':<12} {'raw profile':<12} miss  posture")
    for r in worst:
        print(
            f"  {r['scenario_id']:<8} "
            f"{r['intended_class']:<6} "
            f"{_fmt_profile(r['intended']):<12} "  # type: ignore[arg-type]
            f"{_fmt_profile(r['raw']):<12} "  # type: ignore[arg-type]
            f"{r['mismatch_count']:>4}  "
            f"{(r['phase1_posture'] or '-')}"
        )

    # Optional: dump every row
    if show_all:
        print("\n[5] Full per-scenario table")
        print("-" * 60)
        print(
            f"  {'id':<8} {'class':<6} {'intended':<12} {'raw':<12} miss  posture"
        )
        for r in rows:
            print(
                f"  {r['scenario_id']:<8} "
                f"{r['intended_class']:<6} "
                f"{_fmt_profile(r['intended']):<12} "  # type: ignore[arg-type]
                f"{_fmt_profile(r['raw']):<12} "  # type: ignore[arg-type]
                f"{r['mismatch_count']:>4}  "
                f"{(r['phase1_posture'] or '-')}"
            )

    print()
    print("Legend: profiles are 4-letter shorthand in field order")
    print("        (uncertainty / potential_harm / irreversibility / time_pressure),")
    print("        L=LOW, M=MEDIUM, H=HIGH, ?=missing or invalid.")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare intended vs raw model risk outputs for the MOC evidence run."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"directory with MOC-*.json files (default: {DEFAULT_RUN_DIR})",
    )
    p.add_argument(
        "--intended-csv",
        type=Path,
        default=None,
        help=(
            "optional CSV with intended values keyed by scenario_id "
            "(columns: scenario_id, intended_uncertainty, intended_potential_harm, "
            "intended_irreversibility, intended_time_pressure). "
            "If omitted, the script uses csv_intended embedded in each JSON."
        ),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"write per-scenario failure table to this CSV (default: {DEFAULT_OUT_CSV})",
    )
    p.add_argument(
        "--show-all",
        action="store_true",
        help="print every scenario, not just worst offenders",
    )
    return p.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    intended_override: Optional[Dict[str, Dict[str, str]]] = None
    if args.intended_csv is not None:
        intended_override = load_intended_from_csv(args.intended_csv)
        print(
            f"[info] using intended values from CSV: {args.intended_csv} "
            f"({len(intended_override)} scenarios)"
        )

    rows = load_scenarios(args.run_dir, intended_override=intended_override)
    print(f"[info] loaded {len(rows)} scenarios from {args.run_dir}")

    write_failures_csv(rows, args.out_csv)
    print(f"[info] wrote per-scenario CSV -> {args.out_csv}")

    print_report(rows, show_all=args.show_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
