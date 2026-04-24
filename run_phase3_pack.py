"""Run a Phase 3 scenario pack CSV through evaluate_phase3() and verify outcomes.

Scope
-----
This runner is for **Phase 3 semantic testing only**. It is NOT a full-pipeline
runner. It invokes phase3_gate.evaluate_phase3() directly on the record built
from each CSV row, bypassing Phase 1, Phase 2, and the final execution gate.

Why this runner exists separately from run_full_pipeline.py:
  * It preserves every field in the CSV row (including the atomic/consent
    fields required by EC-04 / EC-06 / EC-09), rather than projecting the
    record down to the canonical 10-field schema used by the full pipeline.
  * It honors the `drop_fields` column, deleting listed keys from the record
    before evaluation, so absent non-critical fields can exercise EC-META and
    the per-evaluator KeyError guards (EC-02 / EC-05 / EC-07 / EC-08 / EC-11 /
    EC-12).
  * It never rewrites `posture`, `rationale`, or `context_tag`, so explicit
    Phase 3 FAIL conditions (e.g. EC-10 under PROCEED) can be exercised
    end-to-end inside evaluate_phase3.

For full-pipeline canonical testing (Phase 1 + Phase 2 + Phase 3 + final
gate + Phase 4 history), use run_full_pipeline.py with a canonical-compatible
scenario pack. The atomic-field pack routed through that runner will collapse
to uniform Phase 3 ambiguity because the canonical projection does not carry
atomic fields; see scenarios/README.md for the separation.

Usage:
    python run_phase3_pack.py [path/to/pack.csv]

If no path is supplied, defaults to
scenarios/phase3_ambiguity_and_hardening_v1.csv.

For each row the script:
  1. Reads the CSV row into a dict.
  2. Removes any fields listed in the `drop_fields` cell (comma-separated).
  3. Invokes phase3_gate.evaluate_phase3(record).
  4. Compares result["phase3_output"] to the row's `expected_phase3` value.

It prints a per-scenario PASS/FAIL line and a summary at the end.
Exit code equals the number of failed scenarios (0 = all passed).
"""

import csv
import sys
import traceback
from collections import Counter
from pathlib import Path

from phase3_gate import evaluate_phase3


DEFAULT_PACK = Path("scenarios/phase3_ambiguity_and_hardening_v1.csv")

# Metadata columns that are NOT part of the Phase 3 record proper.
NON_RECORD_COLUMNS = {
    "drop_fields",
    "expected_phase3",
    "expected_trace_hints",
    "notes",
}


def build_record(row: dict) -> dict:
    record = {
        key: value
        for key, value in row.items()
        if key not in NON_RECORD_COLUMNS
    }
    drop_spec = (row.get("drop_fields") or "").strip()
    if drop_spec:
        for field in (f.strip() for f in drop_spec.split(",")):
            if field:
                record.pop(field, None)
    return record


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_pack(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"ERROR: pack not found at {csv_path}")
        return 1

    rows = load_rows(csv_path)
    print(f"Running pack: {csv_path} ({len(rows)} rows)")
    print("-" * 80)

    passed = 0
    failed = 0
    errored = 0
    expected_counts = Counter()
    actual_counts = Counter()
    mismatches: list[str] = []

    for row in rows:
        scenario_id = row.get("scenario_id", "<missing>")
        expected = (row.get("expected_phase3") or "").strip()
        expected_counts[expected] += 1

        try:
            record = build_record(row)
            result = evaluate_phase3(record)
        except Exception as exc:
            errored += 1
            failed += 1
            print(f"[FAIL] {scenario_id}: EXCEPTION {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        actual = result["phase3_output"]
        actual_counts[actual] += 1

        if actual == expected:
            passed += 1
            print(f"[PASS] {scenario_id}: {actual}")
        else:
            failed += 1
            violated = result.get("violated_constraints", [])
            unresolved = result.get("unresolved_constraints", [])
            detail = (
                f"expected={expected} actual={actual} "
                f"violated={violated} unresolved={unresolved}"
            )
            mismatches.append(f"{scenario_id}: {detail}")
            print(f"[FAIL] {scenario_id}: {detail}")

    print("-" * 80)
    print("SUMMARY")
    print(f"  total    : {len(rows)}")
    print(f"  passed   : {passed}")
    print(f"  failed   : {failed}")
    print(f"  exceptions: {errored}")
    print()
    print("  expected phase3_output counts:")
    for key, value in sorted(expected_counts.items()):
        print(f"    {key or '<blank>'}: {value}")
    print("  actual phase3_output counts:")
    for key, value in sorted(actual_counts.items()):
        print(f"    {key or '<blank>'}: {value}")

    if mismatches:
        print()
        print("MISMATCHES")
        for line in mismatches:
            print(f"  - {line}")

    return failed


def main() -> int:
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = DEFAULT_PACK
    return run_pack(csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
