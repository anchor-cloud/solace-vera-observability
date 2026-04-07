import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, UTC
from pathlib import Path 
import sys
from typing import Any, Dict, List
from collections import Counter

from phase1_rebuild import evaluate_phase1
from phase2_gate import validate_record
from phase3_gate import evaluate_phase3, evaluate_phase3_independent


PIPELINE_HISTORY_PATH = Path("phase4_history") / "phase4_history.jsonl"
PIPELINE_SUMMARY_DIR = Path("phase4_outputs")


def normalize_artifact(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError("Artifact must be a dict or dataclass instance.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_upper(value: str) -> str:
    return (value or "").strip().upper()


CANONICAL_FIELDS = [
    "scenario_id",
    "proposed_action",
    "uncertainty",
    "potential_harm",
    "irreversibility",
    "time_pressure",
    "posture",
    "rationale",
    "context_tag",
    "use_domain",
]


def canonical_phase2_validated_view(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: record.get(field, "") for field in CANONICAL_FIELDS}


def fingerprint_record_view(view: Dict[str, Any]) -> str:
    payload = json.dumps(
        view,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_final_execution_gate(
    phase1_posture: str,
    phase2_outcome: str,
    phase3_output: str,
) -> Dict[str, Any]:
    """
    Autonomous execution gate from actual Phase 1 posture (untampered record),
    Phase 2 outcome from validate_record (normalized), and Phase 3 output
    from phase3_result["phase3_output"]. Pure function; no side effects.
    """
    p1 = safe_upper(phase1_posture)
    p2 = safe_upper(phase2_outcome)
    p3 = safe_upper(phase3_output)

    if p1 != "PROCEED":
        return {
            "execution_allowed": False,
            "final_disposition": "BLOCKED_BY_PHASE1_POSTURE",
            "stop_reason": f"Phase 1 posture is {p1}; autonomous execution is not allowed.",
        }
    if p2 == "ESCALATE":
        return {
            "execution_allowed": False,
            "final_disposition": "BLOCKED_BY_PHASE2_ESCALATION",
            "stop_reason": "Phase 2 requires escalation before execution.",
        }
    if p2 in {"REJECT", "REJECT_NEW_POSTURE_REQUIRED"}:
        return {
            "execution_allowed": False,
            "final_disposition": "BLOCKED_BY_PHASE2_REJECTION",
            "stop_reason": f"Phase 2 outcome is {p2}; autonomous execution is not allowed.",
        }
    if p3 == "ETHICAL_FAIL_CONSTRAINT_VIOLATION":
        return {
            "execution_allowed": False,
            "final_disposition": "BLOCKED_BY_PHASE3_FAIL",
            "stop_reason": "Phase 3 ethical evaluation reported a constraint violation.",
        }
    if p3 == "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED":
        return {
            "execution_allowed": False,
            "final_disposition": "BLOCKED_BY_PHASE3_AMBIGUITY",
            "stop_reason": "Phase 3 requires human review for unresolved ambiguity.",
        }
    if p1 == "PROCEED" and p2 == "PROCEED" and p3 == "ETHICAL_PASS":
        return {
            "execution_allowed": True,
            "final_disposition": "EXECUTION_ALLOWED",
            "stop_reason": "Phase 1 PROCEED, Phase 2 PROCEED, and Phase 3 ETHICAL_PASS.",
        }
    return {
        "execution_allowed": False,
        "final_disposition": "BLOCKED_UNKNOWN_STATE",
        "stop_reason": (
            f"Unexpected combination: Phase 1={p1}, Phase 2={p2}, Phase 3={p3}; "
            f"autonomous execution is not allowed."
        ),
    }


def load_scenarios(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_phase1_adapter(
    scenario_id: str,
    proposed_action: str,
    uncertainty: str,
    potential_harm: str,
    irreversibility: str,
    time_pressure: str,
    context_tag: str,
    use_domain: str,
) -> Dict[str, Any]:
    record = evaluate_phase1(
        scenario_id=scenario_id,
        proposed_action=proposed_action,
        uncertainty=uncertainty,
        potential_harm=potential_harm,
        irreversibility=irreversibility,
        time_pressure=time_pressure,
        context_tag=context_tag,
        use_domain=use_domain,
    )
    return normalize_artifact(record)


def apply_phase3_tamper_mode(record: Dict[str, Any], mode: str) -> Dict[str, Any]:
    tampered = dict(record)
    mode = (mode or "none").strip().lower()

    if mode == "none":
        return tampered

    elif mode == "force_posture_proceed":
        tampered["posture"] = "PROCEED"
        tampered["rationale"] = (
            f"PROCEED triggered because uncertainty={tampered['uncertainty']}, "
            f"potential_harm={tampered['potential_harm']}, "
            f"irreversibility={tampered['irreversibility']}, "
            f"time_pressure={tampered['time_pressure']}, "
            f"use_domain={tampered['use_domain']}"
        )

    elif mode == "bad_rationale_generic":
        tampered["rationale"] = "Action reviewed and seems appropriate"

    elif mode == "bad_rationale_missing_posture_reference":
        tampered["rationale"] = (
            f"Action justified because uncertainty={tampered['uncertainty']}, "
            f"potential_harm={tampered['potential_harm']}, "
            f"irreversibility={tampered['irreversibility']}, "
            f"time_pressure={tampered['time_pressure']}, "
            f"use_domain={tampered['use_domain']} "
            f"and the record state is stable for evaluation."
        )

    else:
        raise ValueError(f"Unknown Phase 3 tamper mode: {mode}")

    return tampered


def append_phase4_history(entry: Dict[str, Any]) -> None:
    ensure_dir(PIPELINE_HISTORY_PATH.parent)
    with PIPELINE_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_phase4_history() -> List[Dict[str, Any]]:
    if not PIPELINE_HISTORY_PATH.exists():
        return []

    records = []
    with PIPELINE_HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_phase4_summary(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_records = len(history)

    phase1_posture_counts = Counter()
    phase2_outcome_counts = Counter()
    phase3_output_counts = Counter()
    violated_constraint_counts = Counter()
    unresolved_constraint_counts = Counter()
    context_tag_counts = Counter()
    use_domain_counts = Counter()

    for record in history:
        phase1_posture_counts.update([record.get("phase1_posture", "")])
        phase2_outcome_counts.update([record.get("phase2_outcome", "")])
        phase3_output_counts.update([record.get("phase3_output", "")])

        for c in record.get("violated_constraints", []):
            violated_constraint_counts.update([c])

        for c in record.get("unresolved_constraints", []):
            unresolved_constraint_counts.update([c])

        context_tag_counts.update([record.get("context_tag", "")])
        use_domain_counts.update([record.get("use_domain", "")])

    def pct(count: int) -> float:
        if total_records == 0:
            return 0.0
        return round((count / total_records) * 100.0, 2)

    summary = {
        "total_records": total_records,
        "phase1_posture_counts": dict(phase1_posture_counts),
        "phase2_outcome_counts": dict(phase2_outcome_counts),
        "phase3_output_counts": dict(phase3_output_counts),
        "phase3_output_percentages": {
            key: pct(value) for key, value in phase3_output_counts.items()
        },
        "violated_constraint_counts": dict(violated_constraint_counts),
        "unresolved_constraint_counts": dict(unresolved_constraint_counts),
        "context_tag_counts": dict(context_tag_counts),
        "use_domain_counts": dict(use_domain_counts),
        "drift_heuristics": {
            "repeat_ec12_ambiguity_count": unresolved_constraint_counts.get("EC-12", 0),
            "repeat_ec10_ambiguity_count": unresolved_constraint_counts.get("EC-10", 0),
            "repeat_ec10_fail_count": violated_constraint_counts.get("EC-10", 0),
        },
    }

    return summary


def save_phase4_summary(summary: Dict[str, Any]) -> None:
    ensure_dir(PIPELINE_SUMMARY_DIR)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_json_path = PIPELINE_SUMMARY_DIR / f"phase4_summary_{timestamp}.json"
    summary_txt_path = PIPELINE_SUMMARY_DIR / f"phase4_summary_{timestamp}.txt"

    write_json(summary_json_path, summary)

    txt_lines = [
        f"Total records: {summary['total_records']}",
        "",
        "Phase 1 posture counts:",
    ]
    for k, v in summary["phase1_posture_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Phase 2 outcome counts:")
    for k, v in summary["phase2_outcome_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Phase 3 output counts:")
    for k, v in summary["phase3_output_counts"].items():
        pct_value = summary["phase3_output_percentages"].get(k, 0.0)
        txt_lines.append(f"  {k}: {v} ({pct_value}%)")

    txt_lines.append("")
    txt_lines.append("Violated constraint counts:")
    for k, v in summary["violated_constraint_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Unresolved constraint counts:")
    for k, v in summary["unresolved_constraint_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Context tag counts:")
    for k, v in summary["context_tag_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Use domain counts:")
    for k, v in summary["use_domain_counts"].items():
        txt_lines.append(f"  {k}: {v}")

    txt_lines.append("")
    txt_lines.append("Drift heuristics:")
    for k, v in summary["drift_heuristics"].items():
        txt_lines.append(f"  {k}: {v}")

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))


def run_post_run_evaluator(outdir: Path, csv_path: Path) -> None:
    """
    Run the external safety_net_evaluator on this run folder. Failures are non-fatal for
    pipeline artifacts (outputs remain valid).
    """
    evaluator_script = Path(__file__).resolve().parent / "safety_net_evaluator.py"
    json_out = outdir / "safety_net_eval.json"
    csv_out = outdir / "safety_net_eval.csv"
    try:
        subprocess.run(
            [
                sys.executable,
                str(evaluator_script),
                str(outdir.resolve()),
                str(csv_path.resolve()),
            ],
            check=True,
        )
        print(f"Safety net evaluator JSON: {json_out.resolve()}")
        print(f"Safety net evaluator CSV: {csv_out.resolve()}")
    except (subprocess.CalledProcessError, OSError) as exc:
        print(
            "WARNING: Pipeline run completed, but the post-run safety net evaluator failed "
            f"({type(exc).__name__}: {exc}). Pipeline outputs in "
            f"{outdir.resolve()} are unchanged.",
            file=sys.stderr,
        )


def main() -> None:
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path("scenarios/phase3_tests_v2.csv")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    outdir = Path("pipeline_outputs") / f"full_pipeline_{timestamp}"
    phase1_dir = outdir / "phase1_records"
    phase2_dir = outdir / "phase2_results"
    phase3_dir = outdir / "phase3_results"

    ensure_dir(outdir)
    ensure_dir(phase1_dir)
    ensure_dir(phase2_dir)
    ensure_dir(phase3_dir)

    rows = load_scenarios(csv_path)

    total = 0
    passed = 0
    failed = 0
    summary_lines = []

    for row in rows:
        total += 1

        scenario_id = row["scenario_id"].strip()
        proposed_action = row["proposed_action"].strip()
        uncertainty = row["uncertainty"].strip()
        potential_harm = row["potential_harm"].strip()
        irreversibility = row["irreversibility"].strip()
        time_pressure = row["time_pressure"].strip()
        context_tag = row["context_tag"].strip()
        use_domain = row["use_domain"].strip()

        expected_phase1 = safe_upper(row.get("expected_phase1", ""))
        expected_phase3 = safe_upper(row.get("expected_phase3", ""))
        tamper_mode = row.get("tamper_mode", "none").strip()
        notes = row.get("notes", "").strip()

        try:
            # -------------------------
            # Phase 1
            # -------------------------
            phase1_record = run_phase1_adapter(
                scenario_id=scenario_id,
                proposed_action=proposed_action,
                uncertainty=uncertainty,
                potential_harm=potential_harm,
                irreversibility=irreversibility,
                time_pressure=time_pressure,
                context_tag=context_tag,
                use_domain=use_domain,
            )

            actual_phase1 = safe_upper(phase1_record.get("posture", ""))

            write_json(phase1_dir / f"{scenario_id}.json", phase1_record)

            # -------------------------
            # Phase 2
            # -------------------------
            raw_phase2_outcome, phase2_reason = validate_record(phase1_record)
            phase2_outcome = safe_upper(raw_phase2_outcome)

            # -------------------------
            # Phase 3
            # -------------------------
            phase3_input_record = apply_phase3_tamper_mode(phase1_record, tamper_mode)

            phase2_validated_fingerprint = fingerprint_record_view(
                canonical_phase2_validated_view(phase1_record)
            )
            phase3_input_fingerprint = fingerprint_record_view(
                canonical_phase2_validated_view(phase3_input_record)
            )
            provenance_match = phase2_validated_fingerprint == phase3_input_fingerprint

            if not provenance_match:
                phase2_outcome = "REJECT_NEW_POSTURE_REQUIRED"
                phase2_reason = (
                    "Integrity failure: Phase 3 input record differs from Phase 2–validated record "
                    "(posture/rationale/fields mutated)."
                )
                phase3_result = {
                    "phase3_output": "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
                    "violated_constraints": ["EC-11"],
                    "unresolved_constraints": [],
                    "trace": [
                        {
                            "constraint_id": "EC-11",
                            "constraint_name": "System Integrity and Anti-Tamper Protection",
                            "status": "FAIL",
                            "message": "Integrity failure: Phase 3 input differs from Phase 2–validated record (provenance mismatch).",
                            "implemented": True,
                        }
                    ],
                }
                phase3_counterfactual = None
            else:
                phase3_result = evaluate_phase3(phase3_input_record)
                phase3_counterfactual = evaluate_phase3_independent(phase3_input_record)

            phase2_result = {
                "scenario_id": scenario_id,
                "phase2_outcome": phase2_outcome,
                "phase2_reason": phase2_reason,
            }
            write_json(phase2_dir / f"{scenario_id}_phase2.json", phase2_result)

            actual_phase3 = safe_upper(phase3_result.get("phase3_output", ""))

            final_execution_gate = compute_final_execution_gate(
                actual_phase1,
                phase2_outcome,
                actual_phase3,
            )

            write_json(
                phase3_dir / f"{scenario_id}_phase3.json",
                {
                    "scenario_id": scenario_id,
                    "tamper_mode": tamper_mode,
                    "phase3_input_record": phase3_input_record,
                    "phase3_result": phase3_result,
                    "provenance_check": {
                        "canonical_fields": CANONICAL_FIELDS,
                        "phase2_validated_fingerprint": phase2_validated_fingerprint,
                        "phase3_input_fingerprint": phase3_input_fingerprint,
                        "match": provenance_match,
                        "on_mismatch_phase2_outcome": "REJECT_NEW_POSTURE_REQUIRED",
                        "on_mismatch_phase2_reason": (
                            "Integrity failure: Phase 3 input record differs from Phase 2–validated record "
                            "(posture/rationale/fields mutated)."
                        ),
                    },
                    "phase3_counterfactual": phase3_counterfactual,
                    "final_execution_gate": final_execution_gate,
                },
            )

            phase1_match = actual_phase1 == expected_phase1 if expected_phase1 else True
            phase3_match = actual_phase3 == expected_phase3 if expected_phase3 else True
            test_pass = phase1_match and phase3_match

            if test_pass:
                passed += 1
            else:
                failed += 1

            # -------------------------
            # Phase 4 append-only history
            # -------------------------
            history_entry = {
                "timestamp_utc": timestamp,
                "scenario_id": scenario_id,
                "notes": notes,
                "tamper_mode": tamper_mode,
                "phase1_posture": actual_phase1,
                "phase2_outcome": phase2_outcome,
                "phase2_reason": phase2_reason,
                "phase3_output": actual_phase3,
                "violated_constraints": phase3_result.get("violated_constraints", []),
                "unresolved_constraints": phase3_result.get("unresolved_constraints", []),
                "context_tag": phase1_record.get("context_tag", ""),
                "use_domain": phase1_record.get("use_domain", ""),
                "pass": test_pass,
                "execution_allowed": final_execution_gate["execution_allowed"],
                "final_disposition": final_execution_gate["final_disposition"],
                "stop_reason": final_execution_gate["stop_reason"],
            }
            append_phase4_history(history_entry)

            summary_lines.append(
                "\n".join(
                    [
                        f"{scenario_id}",
                        f"  Notes: {notes}",
                        f"  Tamper mode: {tamper_mode}",
                        f"  Phase 1: {actual_phase1} (expected {expected_phase1})",
                        f"  Phase 2: {phase2_outcome}",
                        f"  Phase 3: {actual_phase3} (expected {expected_phase3})",
                        f"  Final disposition: {final_execution_gate['final_disposition']}",
                        f"  Execution allowed: {final_execution_gate['execution_allowed']}",
                        f"  Stop reason: {final_execution_gate['stop_reason']}",
                        f"  Violated constraints: {phase3_result.get('violated_constraints', [])}",
                        f"  Unresolved constraints: {phase3_result.get('unresolved_constraints', [])}",
                        f"  PASS: {test_pass}",
                    ]
                )
            )

        except Exception as exc:
            failed += 1
            summary_lines.append(
                "\n".join(
                    [
                        f"{scenario_id}",
                        f"  ERROR: {exc}",
                        "  PASS: False",
                    ]
                )
            )

    # refresh phase4 summary from canonical history
    history = load_phase4_history()
    phase4_summary = build_phase4_summary(history)
    save_phase4_summary(phase4_summary)

    summary_lines.append("")
    summary_lines.append("=== TOTALS ===")
    summary_lines.append(f"Total: {total}")
    summary_lines.append(f"Passed: {passed}")
    summary_lines.append(f"Failed: {failed}")

    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines))

    print(f"Finished. Output written to: {outdir}")
    print(f"Phase 4 canonical history: {PIPELINE_HISTORY_PATH}")
    print(f"Phase 4 summaries refreshed in: {PIPELINE_SUMMARY_DIR}")

    run_post_run_evaluator(outdir, csv_path)


if __name__ == "__main__":
    main()
