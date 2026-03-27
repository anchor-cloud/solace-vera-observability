import json


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


def load_record(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_record(record: dict):
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
            return "REJECT_NEW_POSTURE_REQUIRED", f"Missing field: {field}"

    if record["posture"] not in VALID_POSTURES:
        return "REJECT", f"Invalid posture: {record['posture']}"

    for field in ["uncertainty", "potential_harm", "irreversibility", "time_pressure"]:
        if record[field] not in VALID_LEVELS:
            return "REJECT", f"Invalid {field}: {record[field]}"

    if record["context_tag"] not in VALID_CONTEXT_TAGS:
        return "REJECT", f"Invalid context_tag: {record['context_tag']}"

    if record["use_domain"] not in VALID_USE_DOMAINS:
        return "REJECT", f"Invalid use_domain: {record['use_domain']}"

    if not record["proposed_action"] or not record["proposed_action"].strip():
        return "REJECT", "Missing proposed action"

    if not record["rationale"] or not record["rationale"].strip():
        return "REJECT", "Missing rationale"

    rationale = record["rationale"].strip()
    rationale_upper = rationale.upper()

    if len(rationale) < 25:
        return "REJECT", "Rationale too short to justify decision"

    if not any(level in rationale_upper for level in ["LOW", "MEDIUM", "HIGH"]):
        return "REJECT", "Rationale does not reference input levels"

    if record["posture"] == "ESCALATE" and "ESCALAT" not in rationale_upper:
        return "REJECT", "Escalation rationale missing escalation reference"

    if record["posture"] == "PAUSE" and "PAUSE" not in rationale_upper:
        return "REJECT", "Pause rationale missing pause reference"

    if record["posture"] == "PROCEED" and "PROCEED" not in rationale_upper:
        return "REJECT", "Proceed rationale missing proceed reference"

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
            "REJECT",
            f"Rationale does not match record values: missing {', '.join(missing_pairs)}",
        )

    if record["uncertainty"] == "HIGH" and record["posture"] == "PROCEED":
        return "REJECT_NEW_POSTURE_REQUIRED", "High uncertainty cannot PROCEED"

    if record["potential_harm"] == "HIGH" and record["posture"] == "PROCEED":
        return "REJECT_NEW_POSTURE_REQUIRED", "High harm cannot PROCEED"

    if record["irreversibility"] == "HIGH" and record["posture"] == "PROCEED":
        return "REJECT_NEW_POSTURE_REQUIRED", "High irreversibility cannot PROCEED"

    if (
        record["uncertainty"] == "LOW"
        and record["potential_harm"] == "LOW"
        and record["irreversibility"] == "LOW"
        and record["posture"] == "ESCALATE"
    ):
        return "REJECT_NEW_POSTURE_REQUIRED", "Low-risk case should not ESCALATE"

    if record["posture"] == "ESCALATE":
        return "ESCALATE", "Escalation required before proceeding"

    return "PROCEED", "Record valid"


if __name__ == "__main__":
    record_path = "artifacts/S001.json"
    record = load_record(record_path)

    decision, reason = validate_record(record)

    print("Phase 2 decision:", decision)
    print("Reason:", reason)