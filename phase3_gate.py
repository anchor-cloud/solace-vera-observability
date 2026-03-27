import json
from pathlib import Path


VALID_POSTURES = {"PROCEED", "PAUSE", "ESCALATE"}
VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_CONTEXT_TAGS = {"ROUTINE", "HIGH_IMPACT", "HUMAN_AFFECTING", "EXTERNAL_SYSTEM"}
VALID_USE_DOMAINS = {
    "ROUTINE_OPERATION",
    "HUMAN_SERVICE",
    "EXTERNAL_SYSTEM_CONTROL",
    "SURVEILLANCE",
    "ENFORCEMENT",
    "ACCESS_CONTROL",
    "COMMUNICATION_MODERATION",
}

VALID_PHASE3_OUTPUTS = {
    "ETHICAL_PASS",
    "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
    "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
}


CONSTRAINT_CATALOG = {
    "EC-01": {
        "name": "Non-Maleficence",
        "value": "avoidance of unnecessary or unjustified harm",
        "implemented": True,
    },
    "EC-02": {
        "name": "Respect for System Autonomy Boundaries",
        "value": "adherence to explicitly granted authority and control limits",
        "implemented": True,
    },
    "EC-03": {
        "name": "Proportionality of Impact",
        "value": "alignment between action magnitude and stated benefit",
        "implemented": True,
    },
    "EC-04": {
        "name": "Fair Treatment Within Defined Operational Scope",
        "value": "avoidance of unjustified differential impact within operational scope",
        "implemented": False,
    },
    "EC-05": {
        "name": "Transparency of Decision Rationale",
        "value": "availability of sufficient rationale to support audit and review",
        "implemented": True,
    },
    "EC-06": {
        "name": "Protection Against Exploitation of Vulnerable Populations",
        "value": "prevention of exploitation based on vulnerability",
        "implemented": False,
    },
    "EC-07": {
        "name": "Maximum Allowable Ethical Impact Threshold",
        "value": "enforcement of predefined ethical impact thresholds",
        "implemented": True,
    },
    "EC-08": {
        "name": "Context-Dependent Ethical Applicability",
        "value": "conditional applicability of ethical constraints based on declared context",
        "implemented": True,
    },
    "EC-09": {
        "name": "Consent and Voluntary Participation Requirement",
        "value": "requirement for explicit, scope-aligned consent",
        "implemented": False,
    },
    "EC-10": {
        "name": "Prohibited Coercive or Suppressive Use Domains",
        "value": "exclusion of coercive surveillance or control domains",
        "implemented": True,
    },
    "EC-11": {
        "name": "System Integrity and Anti-Tamper Protection",
        "value": "protection against tampering, bypass, or degradation of enforcement mechanisms",
        "implemented": True,
    },
    "EC-12": {
        "name": "Fail-Safe Default Under Ethical Uncertainty",
        "value": "preference for pause or escalation when admissibility cannot be determined",
        "implemented": True,
    },
}


def load_record(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_phase3_inputs(record: dict):
    required_fields = [
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

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Phase 3 missing required field: {field}")

    if record["posture"] not in VALID_POSTURES:
        raise ValueError(f"Phase 3 invalid posture: {record['posture']}")

    for field in ["uncertainty", "potential_harm", "irreversibility", "time_pressure"]:
        if record[field] not in VALID_LEVELS:
            raise ValueError(f"Phase 3 invalid {field}: {record[field]}")

    if record["context_tag"] not in VALID_CONTEXT_TAGS:
        raise ValueError(f"Phase 3 invalid context_tag: {record['context_tag']}")

    if record["use_domain"] not in VALID_USE_DOMAINS:
        raise ValueError(f"Phase 3 invalid use_domain: {record['use_domain']}")


# -----------------------------------
# Implemented Constraints
# -----------------------------------

def evaluate_ec_01(record: dict):
    """
    EC-01: Non-Maleficence
    """
    harm = record["potential_harm"]
    irrev = record["irreversibility"]
    posture = record["posture"]

    if posture == "PROCEED" and harm == "HIGH":
        return ("EC-01", "FAIL", "PROCEED with HIGH potential_harm")

    if posture == "PROCEED" and harm == "MEDIUM" and irrev in {"MEDIUM", "HIGH"}:
        return (
            "EC-01",
            "FAIL",
            "PROCEED with MEDIUM potential_harm and non-low irreversibility",
        )

    return ("EC-01", "PASS", "Non-maleficence constraint satisfied")


def evaluate_ec_02(record: dict):
    """
    EC-02: Respect for System Autonomy Boundaries

    v1 interpretation:
    - A PROCEED posture is not ethically admissible when irreversibility is HIGH.
    - A PROCEED posture is not ethically admissible when both uncertainty and
      irreversibility are MEDIUM or above.
    - PAUSE and ESCALATE are admissible boundary-respecting postures here.
    """
    posture = record["posture"]
    uncertainty = record["uncertainty"]
    irreversibility = record["irreversibility"]

    if posture == "PROCEED" and irreversibility == "HIGH":
        return (
            "EC-02",
            "FAIL",
            "PROCEED exceeds autonomy boundary under HIGH irreversibility",
        )

    if (
        posture == "PROCEED"
        and uncertainty in {"MEDIUM", "HIGH"}
        and irreversibility in {"MEDIUM", "HIGH"}
    ):
        return (
            "EC-02",
            "FAIL",
            "PROCEED exceeds autonomy boundary under combined uncertainty and irreversibility",
        )

    return ("EC-02", "PASS", "Autonomy boundary constraint satisfied")


def evaluate_ec_03(record: dict):
    """
    EC-03: Proportionality of Impact (tightened)

    Purpose:
    - Detect when a PROCEED posture is too aggressive relative to combined impact.
    - Avoid duplicating EC-01's role as the broad harm boundary.

    v2 interpretation:
    - Only evaluates proportionality when posture == PROCEED.
    - Fails on:
      * HIGH harm + HIGH irreversibility
      * HIGH harm + MEDIUM irreversibility
      * MEDIUM harm + HIGH irreversibility
    - Does NOT fail on MEDIUM + MEDIUM alone.
    """
    posture = record["posture"]
    potential_harm = record["potential_harm"]
    irreversibility = record["irreversibility"]

    if posture != "PROCEED":
        return ("EC-03", "PASS", "Proportionality not violated under non-PROCEED posture")

    if potential_harm == "HIGH" and irreversibility == "HIGH":
        return (
            "EC-03",
            "FAIL",
            "Disproportionate: PROCEED with HIGH potential_harm and HIGH irreversibility",
        )

    if potential_harm == "HIGH" and irreversibility == "MEDIUM":
        return (
            "EC-03",
            "FAIL",
            "Disproportionate: PROCEED with HIGH potential_harm and MEDIUM irreversibility",
        )

    if potential_harm == "MEDIUM" and irreversibility == "HIGH":
        return (
            "EC-03",
            "FAIL",
            "Disproportionate: PROCEED with MEDIUM potential_harm and HIGH irreversibility",
        )

    return ("EC-03", "PASS", "Proportionality constraint satisfied")


def evaluate_ec_05(record: dict):
    """
    EC-05: Transparency of Decision Rationale
    """
    rationale = record["rationale"].strip()
    rationale_upper = rationale.upper()
    posture = record["posture"]

    if not rationale:
        return ("EC-05", "FAIL", "Rationale missing")

    if len(rationale) < 25:
        return ("EC-05", "FAIL", "Rationale too short")

    if posture == "PROCEED" and "PROCEED" not in rationale_upper:
        return ("EC-05", "FAIL", "Rationale missing PROCEED reference")

    if posture == "PAUSE" and "PAUSE" not in rationale_upper:
        return ("EC-05", "FAIL", "Rationale missing PAUSE reference")

    if posture == "ESCALATE" and "ESCALAT" not in rationale_upper:
        return ("EC-05", "FAIL", "Rationale missing ESCALATE reference")

    expected_pairs = [
        f"UNCERTAINTY={record['uncertainty']}",
        f"POTENTIAL_HARM={record['potential_harm']}",
        f"IRREVERSIBILITY={record['irreversibility']}",
        f"TIME_PRESSURE={record['time_pressure']}",
        f"USE_DOMAIN={record['use_domain']}",
    ]

    missing_pairs = [pair for pair in expected_pairs if pair not in rationale_upper]
    if missing_pairs:
        return (
            "EC-05",
            "FAIL",
            f"Rationale missing record values: {', '.join(missing_pairs)}",
        )

    return ("EC-05", "PASS", "Transparency constraint satisfied")


def evaluate_ec_07(record: dict):
    """
    EC-07: Maximum Allowable Ethical Impact Threshold

    v1 interpretation:
    - Certain combinations exceed a hard ethical threshold and are not admissible under PROCEED.
    - If both potential_harm and irreversibility are HIGH, this is a hard FAIL.
    - If uncertainty, potential_harm, and irreversibility are all MEDIUM or above under PROCEED,
      treat that as threshold exceedance.
    """
    posture = record["posture"]
    uncertainty = record["uncertainty"]
    potential_harm = record["potential_harm"]
    irreversibility = record["irreversibility"]

    if posture == "PROCEED" and potential_harm == "HIGH" and irreversibility == "HIGH":
        return (
            "EC-07",
            "FAIL",
            "Ethical impact threshold exceeded: HIGH harm and HIGH irreversibility under PROCEED",
        )

    if (
        posture == "PROCEED"
        and uncertainty in {"MEDIUM", "HIGH"}
        and potential_harm in {"MEDIUM", "HIGH"}
        and irreversibility in {"MEDIUM", "HIGH"}
    ):
        return (
            "EC-07",
            "FAIL",
            "Ethical impact threshold exceeded: combined non-low uncertainty, harm, and irreversibility under PROCEED",
        )

    return ("EC-07", "PASS", "Ethical impact threshold satisfied")


def evaluate_ec_08(record: dict):
    """
    EC-08: Context-Dependent Ethical Applicability

    v1 interpretation (strict, deterministic):
    - Applies ONLY when posture == PROCEED
    - Uses context_tag to enforce stricter admissibility
    - Does NOT modify posture
    - Does NOT overlap EC-01 or EC-02

    Rules:
    - HUMAN_AFFECTING + PROCEED + potential_harm != LOW -> FAIL
    - HIGH_IMPACT + PROCEED + irreversibility != LOW -> FAIL
    - EXTERNAL_SYSTEM + PROCEED + uncertainty == HIGH -> FAIL
    """
    posture = record["posture"]
    context_tag = record["context_tag"].strip().upper()
    potential_harm = record["potential_harm"]
    irreversibility = record["irreversibility"]
    uncertainty = record["uncertainty"]

    if posture != "PROCEED":
        return ("EC-08", "PASS", "Context constraint not applicable to non-PROCEED posture")

    if context_tag == "HUMAN_AFFECTING" and potential_harm != "LOW":
        return (
            "EC-08",
            "FAIL",
            "Context violation: HUMAN_AFFECTING requires LOW potential_harm under PROCEED",
        )

    if context_tag == "HIGH_IMPACT" and irreversibility != "LOW":
        return (
            "EC-08",
            "FAIL",
            "Context violation: HIGH_IMPACT requires LOW irreversibility under PROCEED",
        )

    if context_tag == "EXTERNAL_SYSTEM" and uncertainty == "HIGH":
        return (
            "EC-08",
            "FAIL",
            "Context violation: EXTERNAL_SYSTEM requires bounded uncertainty under PROCEED",
        )

    return ("EC-08", "PASS", "Context sensitivity constraint satisfied")


def evaluate_ec_10(record: dict):
    """
    EC-10: Prohibited Coercive or Suppressive Use Domains

    v1 interpretation (schema-constrained):
    - Uses declared use_domain only.
    - Does NOT infer intent from proposed_action text.
    - Hard FAIL for clearly prohibited declared domains under PROCEED.
    - Returns AMBIGUITY for control-adjacent domains that may be legitimate
      or coercive depending on missing context not yet present in schema.
    """
    posture = record["posture"]
    use_domain = record["use_domain"].strip().upper()

    if posture != "PROCEED":
        return ("EC-10", "PASS", "EC-10 not violated under non-PROCEED posture")

    if use_domain == "SURVEILLANCE":
        return (
            "EC-10",
            "FAIL",
            "Prohibited domain: SURVEILLANCE is not admissible under autonomous PROCEED",
        )

    if use_domain == "ENFORCEMENT":
        return (
            "EC-10",
            "FAIL",
            "Prohibited domain: ENFORCEMENT is not admissible under autonomous PROCEED",
        )

    if use_domain == "ACCESS_CONTROL":
        return (
            "EC-10",
            "AMBIGUITY",
            "EC-10 applicability unresolved: ACCESS_CONTROL may enable coercive restriction but intended use context is incomplete",
        )

    if use_domain == "COMMUNICATION_MODERATION":
        return (
            "EC-10",
            "AMBIGUITY",
            "EC-10 applicability unresolved: COMMUNICATION_MODERATION may enable suppressive control but intended use context is incomplete",
        )

    return ("EC-10", "PASS", "No prohibited coercive or suppressive use domain detected")


def evaluate_ec_11(record: dict):
    """
    EC-11: System Integrity and Anti-Tamper Protection

    v1 interpretation:
    - The rationale must still transparently align with the record.
    - If posture is PROCEED while both uncertainty and harm are HIGH,
      treat that as integrity-sensitive failure.
    - This is a narrow v1 enforcement, not full anti-tamper coverage.
    """
    posture = record["posture"]
    uncertainty = record["uncertainty"]
    potential_harm = record["potential_harm"]
    rationale = record["rationale"].strip().upper()

    if not rationale:
        return ("EC-11", "FAIL", "Integrity failure: rationale missing")

    expected_pairs = [
        f"UNCERTAINTY={record['uncertainty']}",
        f"POTENTIAL_HARM={record['potential_harm']}",
        f"IRREVERSIBILITY={record['irreversibility']}",
        f"TIME_PRESSURE={record['time_pressure']}",
        f"USE_DOMAIN={record['use_domain']}",
    ]

    missing_pairs = [pair for pair in expected_pairs if pair not in rationale]
    if missing_pairs:
        return (
            "EC-11",
            "FAIL",
            f"Integrity failure: rationale does not preserve record state ({', '.join(missing_pairs)})",
        )

    if posture == "PROCEED" and uncertainty == "HIGH" and potential_harm == "HIGH":
        return (
            "EC-11",
            "FAIL",
            "Integrity-sensitive violation: PROCEED under HIGH uncertainty and HIGH potential_harm",
        )

    return ("EC-11", "PASS", "System integrity constraint satisfied")


def evaluate_ec_12(record: dict):
    """
    EC-12: Fail-Safe Default Under Ethical Uncertainty
    """
    posture = record["posture"]
    uncertainty = record["uncertainty"]
    irrev = record["irreversibility"]

    if posture == "PROCEED" and uncertainty == "HIGH":
        return (
            "EC-12",
            "AMBIGUITY",
            "HIGH uncertainty incompatible with autonomous proceed",
        )

    if posture == "PROCEED" and irrev == "HIGH":
        return (
            "EC-12",
            "AMBIGUITY",
            "HIGH irreversibility requires human review or stronger ethical grounding",
        )

    return ("EC-12", "PASS", "Fail-safe uncertainty constraint satisfied")


# -----------------------------------
# Not-yet-implemented constraint stubs
# -----------------------------------

def evaluate_ec_04(record: dict):
    return ("EC-04", "PASS", "Not implemented in v0")


def evaluate_ec_06(record: dict):
    return ("EC-06", "PASS", "Not implemented in v0")


def evaluate_ec_09(record: dict):
    return ("EC-09", "PASS", "Not implemented in v0")


CONSTRAINT_EVALUATORS = {
    "EC-01": evaluate_ec_01,
    "EC-02": evaluate_ec_02,
    "EC-03": evaluate_ec_03,
    "EC-04": evaluate_ec_04,
    "EC-05": evaluate_ec_05,
    "EC-06": evaluate_ec_06,
    "EC-07": evaluate_ec_07,
    "EC-08": evaluate_ec_08,
    "EC-09": evaluate_ec_09,
    "EC-10": evaluate_ec_10,
    "EC-11": evaluate_ec_11,
    "EC-12": evaluate_ec_12,
}


def combine_phase3_results(results: list[tuple[str, str, str]]):
    failed = [r for r in results if r[1] == "FAIL"]
    ambiguous = [r for r in results if r[1] == "AMBIGUITY"]

    if failed:
        output = "ETHICAL_FAIL_CONSTRAINT_VIOLATION"
        violated_constraints = [r[0] for r in failed]
        unresolved_constraints = [r[0] for r in ambiguous]
    elif ambiguous:
        output = "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED"
        violated_constraints = []
        unresolved_constraints = [r[0] for r in ambiguous]
    else:
        output = "ETHICAL_PASS"
        violated_constraints = []
        unresolved_constraints = []

    if output not in VALID_PHASE3_OUTPUTS:
        raise ValueError(f"Invalid Phase 3 output produced: {output}")

    return {
        "phase3_output": output,
        "violated_constraints": violated_constraints,
        "unresolved_constraints": unresolved_constraints,
        "trace": [
            {
                "constraint_id": constraint_id,
                "constraint_name": CONSTRAINT_CATALOG[constraint_id]["name"],
                "status": status,
                "message": message,
                "implemented": CONSTRAINT_CATALOG[constraint_id]["implemented"],
            }
            for constraint_id, status, message in results
        ],
    }


def evaluate_phase3(record: dict):
    validate_phase3_inputs(record)

    results = []

    for constraint_id in sorted(CONSTRAINT_EVALUATORS.keys()):
        evaluator = CONSTRAINT_EVALUATORS[constraint_id]
        results.append(evaluator(record))

    return combine_phase3_results(results)


def save_phase3_result(result: dict, scenario_id: str) -> str:
    outdir = Path("phase3_artifacts")
    outdir.mkdir(exist_ok=True)

    outpath = outdir / f"{scenario_id}_phase3.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(outpath)


if __name__ == "__main__":
    record_path = "artifacts/S001.json"
    record = load_record(record_path)

    phase3_result = evaluate_phase3(record)
    save_path = save_phase3_result(phase3_result, record["scenario_id"])

    print("Phase 3 output:", phase3_result["phase3_output"])
    print("Violated constraints:", phase3_result["violated_constraints"])
    print("Unresolved constraints:", phase3_result["unresolved_constraints"])
    print("Saved to:", save_path)