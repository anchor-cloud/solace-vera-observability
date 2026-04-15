from typing import Any, Callable, Dict

from phase2_gate import validate_record
from phase3_gate import evaluate_phase3
from run_full_pipeline import (
    apply_phase3_tamper_mode,
    canonical_phase2_validated_view,
    compute_final_execution_gate,
    fingerprint_record_view,
    run_phase1_adapter,
    safe_upper,
)


def enforce_and_execute(
    action_payload: dict,
    actuator_fn: Callable[..., Any],
    tamper_mode: str = "none",
) -> Dict[str, Any]:
    required = (
        "scenario_id",
        "proposed_action",
        "uncertainty",
        "potential_harm",
        "irreversibility",
        "time_pressure",
        "context_tag",
        "use_domain",
    )
    missing = [k for k in required if k not in action_payload]
    if missing:
        raise ValueError(
            "action_payload missing required fields: " + ", ".join(sorted(missing))
        )

    scenario_id = str(action_payload["scenario_id"]).strip()
    proposed_action = str(action_payload["proposed_action"]).strip()
    uncertainty = str(action_payload["uncertainty"]).strip()
    potential_harm = str(action_payload["potential_harm"]).strip()
    irreversibility = str(action_payload["irreversibility"]).strip()
    time_pressure = str(action_payload["time_pressure"]).strip()
    context_tag = str(action_payload["context_tag"]).strip()
    use_domain = str(action_payload["use_domain"]).strip()

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

    raw_phase2_outcome, phase2_reason = validate_record(phase1_record)
    phase2_outcome = safe_upper(raw_phase2_outcome)

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
    else:
        phase3_result = evaluate_phase3(phase3_input_record)

    actual_phase1 = safe_upper(phase1_record.get("posture", ""))
    actual_phase3 = safe_upper(phase3_result.get("phase3_output", ""))

    final_execution_gate = compute_final_execution_gate(
        actual_phase1,
        phase2_outcome,
        actual_phase3,
    )

    base: Dict[str, Any] = {
        "execution_allowed": final_execution_gate["execution_allowed"],
        "final_disposition": final_execution_gate["final_disposition"],
        "stop_reason": final_execution_gate["stop_reason"],
        "phase1_record": phase1_record,
        "phase2_outcome": phase2_outcome,
        "phase2_reason": phase2_reason,
        "phase3_input_record": phase3_input_record,
        "phase3_result": phase3_result,
        "provenance_match": provenance_match,
    }

    if not final_execution_gate["execution_allowed"]:
        base["actuator_called"] = False
        return base

    base["actuator_called"] = True
    base["actuator_result"] = actuator_fn(action_payload)
    return base
