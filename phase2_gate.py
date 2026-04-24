import json
import re


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
    "FINANCIAL",
}

_LEVEL_FIELDS = ["uncertainty", "potential_harm", "irreversibility"]

# Narrative risk words (standalone); not satisfied by KEY=value fields alone.
_RISK_TERMS = re.compile(
    r"\b(risk|danger|dangerous|threat|unsafe|hazard|concern)\b",
    re.IGNORECASE,
)
# Explicit mitigation vocabulary only (no posture words).
_MITIGATION_TERMS = re.compile(
    r"\b(?:mitigat(?:e|ion|ing)?|safeguard|monitor(?:ing)?|verif(?:y|ication)|"
    r"compensat(?:e|ion|ing)?|rollback|backup|redress|contingency)\b",
    re.IGNORECASE,
)


def _count_medium_dimensions(record: dict) -> int:
    return sum(1 for f in _LEVEL_FIELDS if record.get(f) == "MEDIUM")


def _proceed_risk_narrative_without_mitigation(rationale: str) -> bool:
    if not _RISK_TERMS.search(rationale):
        return False
    return _MITIGATION_TERMS.search(rationale) is None


def _has_objective_rationale_duplication(rationale: str) -> bool:
    """True only when duplication is exact (doubled body or repeated sentence)."""
    t = rationale.strip()
    if len(t) < 2:
        return False
    if len(t) % 2 == 0 and t[: len(t) // 2] == t[len(t) // 2 :]:
        return True
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sentences) < 2:
        return False
    norm = [re.sub(r"\s+", " ", s.casefold()) for s in sentences]
    return len(norm) != len(set(norm))


_ACTION_MARKERS = [
    "payroll", "account", "access", "login", "session", "privileges",
    "contractor", "employee", "badge", "vpn", "token", "credentials",
    "offboarding", "termination",
]

_CAUSAL_PHRASES = [
    "because", "to prevent", "to reduce", "in order to", "so that",
    "due to", "pending", "while",
]

_SCOPE_REVERSIBILITY_CUES = [
    "only", "temporary", "temporarily", "limited", "specific session",
    "session only", "not the full account", "reversible", "restore",
    "restored", "restorable", "pending verification",
]

_MITIGATION_CUES = [
    "verify", "verification", "monitor", "monitoring", "rollback",
    "safeguard", "preserve logs", "review", "investigation",
    "containment", "compensating control",
]

_SENSITIVE_PROCEED_DOMAINS = {"ACCESS_CONTROL", "COMMUNICATION_MODERATION", "FINANCIAL"}


def _extract_action_markers(proposed_action: str) -> list[str]:
    lower = proposed_action.lower()
    found = []
    for marker in _ACTION_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"
        if re.search(pattern, lower):
            found.append(marker)
    return found


def _rationale_mentions_action_target(rationale: str, markers: list[str]) -> bool:
    lower = rationale.lower()
    hit_count = 0

    for marker in markers:
        pattern = r"\b" + re.escape(marker) + r"\b"
        if re.search(pattern, lower):
            hit_count += 1

    if len(markers) <= 1:
        return hit_count >= 1

    return hit_count >= 2


def _has_causal_language(rationale: str) -> bool:
    lower = rationale.lower()

    phrase_hits = any(phrase in lower for phrase in _CAUSAL_PHRASES)

    intent_patterns = [
        r"\bto\s+(prevent|reduce|limit|restrict|avoid|mitigate|control)\b",
        r"\bfor\s+(containment|safety|security|verification|review|investigation)\b",
        r"\bneeded\s+to\b",
        r"\brequired\s+to\b",
        r"\bused\s+to\b",
    ]

    pattern_hits = any(re.search(pattern, lower) for pattern in intent_patterns)

    return phrase_hits or pattern_hits


def _has_scope_or_reversibility_cue(rationale: str) -> bool:
    lower = rationale.lower()
    return any(cue in lower for cue in _SCOPE_REVERSIBILITY_CUES)


def _has_mitigation_cue(rationale: str) -> bool:
    lower = rationale.lower()
    return any(cue in lower for cue in _MITIGATION_CUES)


def _is_metadata_echo_only(rationale: str, markers: list[str]) -> bool:
    upper = rationale.upper()

    required_pair_prefixes = [
        "UNCERTAINTY=",
        "POTENTIAL_HARM=",
        "IRREVERSIBILITY=",
        "TIME_PRESSURE=",
        "USE_DOMAIN=",
    ]

    has_required_field_pairs = all(prefix in upper for prefix in required_pair_prefixes)

    if not has_required_field_pairs:
        return False

    if _rationale_mentions_action_target(rationale, markers):
        return False

    if _has_causal_language(rationale):
        return False

    return True


_GENERIC_JUSTIFICATION_PHRASES = [
    "this action is necessary",
    "appropriate action",
    "ensure safety",
    "reduce risk",
    "limited in scope",
    "monitoring in place",
    "mitigation in place",
    "monitoring will continue",
    "review will occur",
    "verification is pending",
    "logs are preserved",
    "pending verification",
    "temporary measure",
    "control is in place",
]


def _is_overly_generic_justification(rationale: str, markers: list[str]) -> bool:
    lower = rationale.lower()

    generic_hits = sum(1 for phrase in _GENERIC_JUSTIFICATION_PHRASES if phrase in lower)
    marker_hits = sum(1 for marker in markers if re.search(r"\b" + re.escape(marker) + r"\b", lower))
    has_causal = _has_causal_language(rationale)

    return generic_hits >= 2 and has_causal and marker_hits <= 1


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

    if _has_objective_rationale_duplication(rationale):
        return "REJECT", "Duplicated rationale text detected"

    if record["posture"] == "PROCEED" and _proceed_risk_narrative_without_mitigation(
        rationale
    ):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "PROCEED rationale describes risk without mitigation vocabulary",
        )

    markers = _extract_action_markers(record["proposed_action"])

    if markers and not _rationale_mentions_action_target(rationale, markers):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "Rationale is not coherently tied to proposed action",
        )

    if not _has_causal_language(rationale):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "Rationale lacks causal explanation for proposed action",
        )

    if _is_metadata_echo_only(rationale, markers):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "Rationale appears to restate metadata without action-specific justification",
        )

    if _is_overly_generic_justification(rationale, markers):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "Rationale is generic and lacks action-specific justification detail",
        )

    if record["posture"] == "PROCEED":
        if record.get("use_domain") in _SENSITIVE_PROCEED_DOMAINS:
            if not (_has_scope_or_reversibility_cue(rationale) and _has_mitigation_cue(rationale)):
                return (
                    "REJECT_NEW_POSTURE_REQUIRED",
                    "Sensitive-domain PROCEED rationale lacks sufficient scope control and mitigation detail",
                )
        elif not (_has_scope_or_reversibility_cue(rationale) or _has_mitigation_cue(rationale)):
            return (
                "REJECT_NEW_POSTURE_REQUIRED",
                "PROCEED rationale lacks bounded-scope or mitigation explanation",
            )

    if _count_medium_dimensions(record) >= 2 and record["posture"] == "PROCEED":
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "Multiple MEDIUM risk dimensions cannot PROCEED",
        )

    if (
        record["time_pressure"] == "HIGH"
        and record["uncertainty"] == "HIGH"
        and record["posture"] == "PROCEED"
    ):
        return (
            "REJECT_NEW_POSTURE_REQUIRED",
            "High time pressure with high uncertainty cannot PROCEED",
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