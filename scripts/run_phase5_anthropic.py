"""run_phase5_anthropic.py

Standalone Phase 5 (Anthropic/Claude) backfill runner.

For each scenario in the given CSV (default: scenarios/moc_evidence_pack.csv),
this script locates the MOST RECENT existing Phase 3 output JSON in
pipeline_outputs/, loads it, runs the Anthropic Phase 5 reflection over the
stored phase3_input_record + phase3_result, and writes the result back into the
SAME JSON file under a NEW key ``phase5_reflection_claude``.

It never overwrites the existing ``phase5_reflection`` key (OpenAI results).

Every scenario is wrapped in try/except: a single failure logs a warning and the
run continues. This script does not modify any other files.

Usage (from project root):
    python scripts/run_phase5_anthropic.py
    python scripts/run_phase5_anthropic.py scenarios/moc_evidence_pack.csv

Requires: ANTHROPIC_API_KEY in the environment (and: pip install anthropic).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase5_reflection_anthropic import run_phase5_reflection

# Project root = parent of this scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_OUTPUTS = ROOT / "pipeline_outputs"
DEFAULT_CSV = ROOT / "scenarios" / "moc_evidence_pack.csv"

# Key we write into the JSON. Distinct from the OpenAI key "phase5_reflection".
OUTPUT_KEY = "phase5_reflection_claude"


def load_scenario_ids(csv_path: Path) -> List[str]:
    """Read scenario_id values (in order, de-duplicated) from the CSV."""
    ids: List[str] = []
    seen = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("scenario_id") or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def find_latest_phase3_json(scenario_id: str) -> Optional[Path]:
    """Find the most recent existing Phase 3 output JSON for a scenario ID.

    Searches every run folder under pipeline_outputs/ for
    ``<scenario_id>_phase3.json`` and returns the newest by modification time.
    """
    if not PIPELINE_OUTPUTS.exists():
        return None
    matches = list(
        PIPELINE_OUTPUTS.glob(f"**/phase3_results/{scenario_id}_phase3.json")
    )
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def extract_inputs(doc: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Pull phase3_input_record and phase3_result out of a loaded JSON doc."""
    record = doc.get("phase3_input_record") or {}
    phase3_result = doc.get("phase3_result") or {}
    return record, phase3_result


def main(argv: List[str]) -> None:
    csv_arg = argv[0] if argv else None
    csv_path = Path(csv_arg) if csv_arg else DEFAULT_CSV

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    scenario_ids = load_scenario_ids(csv_path)
    print(f"Loaded {len(scenario_ids)} scenario IDs from {csv_path}")
    print(f"Searching for Phase 3 outputs under: {PIPELINE_OUTPUTS}\n")

    processed = 0
    skipped = 0
    errored = 0

    for idx, scenario_id in enumerate(scenario_ids, start=1):
        prefix = f"[{idx}/{len(scenario_ids)}] {scenario_id}"
        try:
            json_path = find_latest_phase3_json(scenario_id)
            if json_path is None:
                print(f"{prefix}: SKIP — no Phase 3 output found in pipeline_outputs/")
                skipped += 1
                continue

            with json_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)

            record, phase3_result = extract_inputs(doc)

            # Defensive: the Anthropic module caches under "phase5_reflection".
            # Pass a copy without that key so it always runs Claude fresh and
            # never returns a cached OpenAI result.
            record_for_run = dict(record)
            record_for_run.pop("phase5_reflection", None)

            reflection = run_phase5_reflection(record_for_run, phase3_result)

            # Write into the NEW key only; leave existing phase5_reflection intact.
            doc[OUTPUT_KEY] = reflection
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)

            err = reflection.get("reflection_error")
            status = "ERROR_IN_REFLECTION" if err else "OK"
            answered = reflection.get("questions_answered")
            consent = reflection.get("q4_consent_to_share")
            print(
                f"{prefix}: {status} — wrote '{OUTPUT_KEY}' to "
                f"{json_path.relative_to(ROOT)} "
                f"(answered={answered}, consent={consent}"
                + (f", error={err}" if err else "")
                + ")"
            )
            processed += 1

        except Exception as exc:  # noqa: BLE001 - one scenario must never stop the run
            errored += 1
            print(
                f"{prefix}: FAILED — {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print("\n=== Phase 5 (Anthropic) backfill complete ===")
    print(f"  Processed: {processed}")
    print(f"  Skipped (no Phase 3 output): {skipped}")
    print(f"  Failed: {errored}")


if __name__ == "__main__":
    main(sys.argv[1:])
