import json
import re
from pathlib import Path

from phase3_text_helpers import normalize_text, phrase_window_match


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

VALID_PHASE3_OUTPUTS = {
    "ETHICAL_PASS",
    "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
    "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
}

VALID_AFFECTED_GROUPS = {
    "GENERAL_POPULATION",
    "SPECIFIC_SUBGROUP",
    "PROTECTED_OR_HISTORICALLY_DISADVANTAGED_GROUP",
}

VALID_DISTRIBUTION_OF_IMPACT = {
    "UNIFORM_OR_PROPORTIONAL",
    "TARGETED_BENEFIT_TO_DISADVANTAGED",
    "DISPROPORTIONATE_IMPACT_ON_DISADVANTAGED",
}

VALID_BENEFIT_DISTRIBUTION = {
    "BENEFIT_GENERAL_OR_NEUTRAL",
    "BENEFIT_TO_AFFECTED_GROUPS",
    "BENEFIT_TO_UNINVOLVED_OR_MORE_POWERFUL_GROUPS",
}

VALID_POPULATION_VULNERABILITY_FLAG = {
    "NONE",
    "POTENTIALLY_VULNERABLE",
    "CLEARLY_VULNERABLE",
}

VALID_CONSENT_STATUS = {
    "NO_CONSENT",
    "IMPLICIT_OR_BUNDLED_CONSENT",
    "EXPLICIT_CONSENT",
}

VALID_CONSENT_SCOPE = {
    "NOT_APPLICABLE",
    "WITHIN_DECLARED_SCOPE",
    "OUTSIDE_OR_AMBIGUOUS_SCOPE",
}

VALID_PARTICIPATION_TYPE = {
    "NO_DIRECT_PARTICIPATION",
    "VOLUNTARY_OPT_IN",
    "VOLUNTARY_OPT_OUT",
    "CONDITION_OF_ACCESS",
    "MANDATORY_NO_ALTERNATIVE",
}

VALID_PARTICIPATION_INFORMATION_QUALITY = {
    "NOT_APPLICABLE",
    "CLEAR_AND_COMPREHENSIVE",
    "PARTIAL_OR_COMPLEX",
    "MINIMAL_OR_MISLEADING",
}


CONSTRAINT_CATALOG = {
    "EC-META": {
        "name": "Global Missing Core Metadata Severity",
        "value": "severity classification for missing core Phase 3 decision metadata",
        "implemented": True,
    },
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
        "implemented": True,
    },
    "EC-05": {
        "name": "Transparency of Decision Rationale",
        "value": "availability of sufficient rationale to support audit and review",
        "implemented": True,
    },
    "EC-06": {
        "name": "Protection Against Exploitation of Vulnerable Populations",
        "value": "prevention of exploitation based on vulnerability",
        "implemented": True,
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
        "implemented": True,
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
    "EC-13": {
        "name": "Text-Encoded Harmful Intent / Impact",
        "value": "detection of harmful intent or covert impact patterns expressed in proposed_action text",
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

    # Optional atomic fields for EC-04 / EC-06 / EC-09 / EC-13.
    if "affected_groups" in record and record["affected_groups"] not in VALID_AFFECTED_GROUPS:
        raise ValueError(f"Phase 3 invalid affected_groups: {record['affected_groups']}")

    if (
        "distribution_of_impact" in record
        and record["distribution_of_impact"] not in VALID_DISTRIBUTION_OF_IMPACT
    ):
        raise ValueError(
            f"Phase 3 invalid distribution_of_impact: {record['distribution_of_impact']}"
        )

    if (
        "benefit_distribution" in record
        and record["benefit_distribution"] not in VALID_BENEFIT_DISTRIBUTION
    ):
        raise ValueError(
            f"Phase 3 invalid benefit_distribution: {record['benefit_distribution']}"
        )

    if (
        "population_vulnerability_flag" in record
        and record["population_vulnerability_flag"] not in VALID_POPULATION_VULNERABILITY_FLAG
    ):
        raise ValueError(
            "Phase 3 invalid population_vulnerability_flag: "
            f"{record['population_vulnerability_flag']}"
        )

    if "consent_status" in record and record["consent_status"] not in VALID_CONSENT_STATUS:
        raise ValueError(f"Phase 3 invalid consent_status: {record['consent_status']}")

    if "consent_scope" in record and record["consent_scope"] not in VALID_CONSENT_SCOPE:
        raise ValueError(f"Phase 3 invalid consent_scope: {record['consent_scope']}")

    if (
        "participation_type" in record
        and record["participation_type"] not in VALID_PARTICIPATION_TYPE
    ):
        raise ValueError(
            f"Phase 3 invalid participation_type: {record['participation_type']}"
        )

    if (
        "participation_information_quality" in record
        and record["participation_information_quality"]
        not in VALID_PARTICIPATION_INFORMATION_QUALITY
    ):
        raise ValueError(
            "Phase 3 invalid participation_information_quality: "
            f"{record['participation_information_quality']}"
        )


def validate_phase3_enums_and_optional_fields(record: dict):
    def is_missing(value):
        return value is None or (isinstance(value, str) and value.strip() == "")

    if "posture" in record and not is_missing(record["posture"]) and record["posture"] not in VALID_POSTURES:
        raise ValueError(f"Phase 3 invalid posture: {record['posture']}")

    for field in ["uncertainty", "potential_harm", "irreversibility", "time_pressure"]:
        if field in record and not is_missing(record[field]) and record[field] not in VALID_LEVELS:
            raise ValueError(f"Phase 3 invalid {field}: {record[field]}")

    if "context_tag" in record and not is_missing(record["context_tag"]) and record["context_tag"] not in VALID_CONTEXT_TAGS:
        raise ValueError(f"Phase 3 invalid context_tag: {record['context_tag']}")

    if "use_domain" in record and not is_missing(record["use_domain"]) and record["use_domain"] not in VALID_USE_DOMAINS:
        raise ValueError(f"Phase 3 invalid use_domain: {record['use_domain']}")

    # Optional atomic fields for EC-04 / EC-06 / EC-09 / EC-13.
    if "affected_groups" in record and not is_missing(record["affected_groups"]) and record["affected_groups"] not in VALID_AFFECTED_GROUPS:
        raise ValueError(f"Phase 3 invalid affected_groups: {record['affected_groups']}")

    if (
        "distribution_of_impact" in record
        and not is_missing(record["distribution_of_impact"])
        and record["distribution_of_impact"] not in VALID_DISTRIBUTION_OF_IMPACT
    ):
        raise ValueError(
            f"Phase 3 invalid distribution_of_impact: {record['distribution_of_impact']}"
        )

    if (
        "benefit_distribution" in record
        and not is_missing(record["benefit_distribution"])
        and record["benefit_distribution"] not in VALID_BENEFIT_DISTRIBUTION
    ):
        raise ValueError(
            f"Phase 3 invalid benefit_distribution: {record['benefit_distribution']}"
        )

    if (
        "population_vulnerability_flag" in record
        and not is_missing(record["population_vulnerability_flag"])
        and record["population_vulnerability_flag"] not in VALID_POPULATION_VULNERABILITY_FLAG
    ):
        raise ValueError(
            "Phase 3 invalid population_vulnerability_flag: "
            f"{record['population_vulnerability_flag']}"
        )

    if "consent_status" in record and not is_missing(record["consent_status"]) and record["consent_status"] not in VALID_CONSENT_STATUS:
        raise ValueError(f"Phase 3 invalid consent_status: {record['consent_status']}")

    if "consent_scope" in record and not is_missing(record["consent_scope"]) and record["consent_scope"] not in VALID_CONSENT_SCOPE:
        raise ValueError(f"Phase 3 invalid consent_scope: {record['consent_scope']}")

    if (
        "participation_type" in record
        and not is_missing(record["participation_type"])
        and record["participation_type"] not in VALID_PARTICIPATION_TYPE
    ):
        raise ValueError(
            f"Phase 3 invalid participation_type: {record['participation_type']}"
        )

    if (
        "participation_information_quality" in record
        and not is_missing(record["participation_information_quality"])
        and record["participation_information_quality"]
        not in VALID_PARTICIPATION_INFORMATION_QUALITY
    ):
        raise ValueError(
            "Phase 3 invalid participation_information_quality: "
            f"{record['participation_information_quality']}"
        )


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
    if "uncertainty" not in record:
        return ("EC-02", "AMBIGUITY", "EC-02 cannot evaluate: uncertainty missing")

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
    for _nc_field in ("uncertainty", "time_pressure"):
        if _nc_field not in record:
            return ("EC-05", "AMBIGUITY", f"EC-05 cannot evaluate: {_nc_field} missing")

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
    if "uncertainty" not in record:
        return ("EC-07", "AMBIGUITY", "EC-07 cannot evaluate: uncertainty missing")

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
    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    if "context_tag" not in record or _is_blank(record.get("context_tag")):
        return ("EC-08", "AMBIGUITY", "EC-08 cannot evaluate: context_tag missing")
    if "uncertainty" not in record or _is_blank(record.get("uncertainty")):
        return ("EC-08", "AMBIGUITY", "EC-08 cannot evaluate: uncertainty missing")

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
    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    for _nc_field in ("uncertainty", "time_pressure"):
        if _nc_field not in record or _is_blank(record.get(_nc_field)):
            return ("EC-11", "AMBIGUITY", f"EC-11 cannot evaluate: {_nc_field} missing")

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
    if "uncertainty" not in record:
        return ("EC-12", "AMBIGUITY", "EC-12 cannot evaluate: uncertainty missing")

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
# EC-04 / EC-06 / EC-09 / EC-13
# -----------------------------------

def evaluate_ec_04(record: dict):
    """
    EC-04: Fair Treatment Within Defined Operational Scope

    Atomic fields used (when present):
    - affected_groups
    - distribution_of_impact
    - benefit_distribution
    """
    required_keys = [
        "affected_groups",
        "distribution_of_impact",
        "benefit_distribution",
        "population_vulnerability_flag",
    ]

    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    missing = [k for k in required_keys if k not in record or _is_blank(record.get(k))]
    if missing:
        action_text = normalize_text(record.get("proposed_action") or "")
        _ec04_targeting = re.search(
            r"\b(group|demographic|region|segment|select(?:ed)?|specific|certain|cohort|designated|classified)\b",
            action_text,
        )
        _ec04_differential = re.search(
            r"\b(differentiated|differential|varying|varied|based\s+on|adjusted\s+(?:for|by))\b",
            action_text,
        )
        if _ec04_targeting and _ec04_differential:
            return (
                "EC-04",
                "AMBIGUITY",
                "Fairness metadata missing but proposed action contains targeting + differential treatment signals (missing_metadata)",
            )
        return (
            "EC-04",
            "AMBIGUITY",
            "Not applicable / insufficient signal \u2014 required fairness-related fields not present",
        )

    affected_groups = record["affected_groups"]
    distribution_of_impact = record["distribution_of_impact"]
    benefit_distribution = record["benefit_distribution"]

    if distribution_of_impact == "DISPROPORTIONATE_IMPACT_ON_DISADVANTAGED":
        return (
            "EC-04",
            "FAIL",
            "Disproportionate negative impact on disadvantaged or protected groups",
        )

    if (
        affected_groups == "PROTECTED_OR_HISTORICALLY_DISADVANTAGED_GROUP"
        and benefit_distribution == "BENEFIT_TO_UNINVOLVED_OR_MORE_POWERFUL_GROUPS"
    ):
        return (
            "EC-04",
            "FAIL",
            "Burden on disadvantaged groups with benefits accruing to more powerful or unimpacted groups",
        )

    return ("EC-04", "PASS", "Fair treatment constraint not violated under declared scope")


def evaluate_ec_06(record: dict):
    """
    EC-06: Protection Against Exploitation of Vulnerable Populations

    Atomic fields used (when present):
    - population_vulnerability_flag
    - potential_harm
    """
    required_keys = [
        "population_vulnerability_flag",
        "potential_harm",
    ]

    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    missing = [k for k in required_keys if k not in record or _is_blank(record.get(k))]
    if missing:
        action_text = normalize_text(record.get("proposed_action") or "")
        _ec06_population = re.search(
            r"\b(users?|all\s+users?|all\s+accounts?|platform[\s-]?wide|everyone|every\s+(?:user|account))\b",
            action_text,
        )
        _ec06_impact = re.search(
            r"\b(default|mandatory|no\s+alternative|forced|required|compulsory)\b",
            action_text,
        )
        if _ec06_population and _ec06_impact:
            return (
                "EC-06",
                "AMBIGUITY",
                "Vulnerability metadata missing but proposed action contains population-scale + mandatory/default impact signals (missing_metadata)",
            )
        return (
            "EC-06",
            "AMBIGUITY",
            "Not applicable / insufficient signal \u2014 required vulnerability-related fields not present",
        )

    vulnerability = record["population_vulnerability_flag"]
    potential_harm = record["potential_harm"]

    if vulnerability == "CLEARLY_VULNERABLE" and potential_harm in {"MEDIUM", "HIGH"}:
        return (
            "EC-06",
            "FAIL",
            "Non-low harm declared for clearly vulnerable population",
        )

    if (
        vulnerability == "POTENTIALLY_VULNERABLE"
        and potential_harm == "HIGH"
    ):
        return (
            "EC-06",
            "FAIL",
            "HIGH harm declared for potentially vulnerable population",
        )

    return ("EC-06", "PASS", "Vulnerability constraint not violated under declared conditions")


def evaluate_ec_09(record: dict):
    """
    EC-09: Consent and Voluntary Participation Requirement

    Atomic fields used (when present):
    - consent_status
    - consent_scope
    - participation_type
    """
    required_keys = [
        "consent_status",
        "consent_scope",
        "participation_type",
    ]

    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    missing = [k for k in required_keys if k not in record or _is_blank(record.get(k))]
    if missing:
        action_text = normalize_text(record.get("proposed_action") or "")
        _ec09_consent = re.search(
            r"\b(opt[\s-]?in|opt[\s-]?out|data\s+sharing|sharing\s+feature|sharing\s+program|preference|consent)\b",
            action_text,
        )
        _ec09_scope = re.search(
            r"\b(all\s+(?:users?|accounts?|platform)|platform[\s-]?wide|everyone|every\s+(?:user|account)|default)\b",
            action_text,
        )
        if _ec09_consent and _ec09_scope:
            return (
                "EC-09",
                "AMBIGUITY",
                "Consent metadata missing but proposed action contains consent-related + broad-scope signals (missing_metadata)",
            )
        return (
            "EC-09",
            "AMBIGUITY",
            "Not applicable / insufficient signal \u2014 required consent-related fields not present",
        )

    consent_status = record["consent_status"]
    consent_scope = record["consent_scope"]
    participation_type = record["participation_type"]

    if participation_type != "NO_DIRECT_PARTICIPATION" and consent_status == "NO_CONSENT":
        return (
            "EC-09",
            "FAIL",
            "Directly affected participants have no consent recorded",
        )

    if (
        consent_status in {"IMPLICIT_OR_BUNDLED_CONSENT", "EXPLICIT_CONSENT"}
        and consent_scope == "OUTSIDE_OR_AMBIGUOUS_SCOPE"
    ):
        return (
            "EC-09",
            "FAIL",
            "Use appears outside or ambiguous relative to the recorded consent scope",
        )

    if consent_status == "IMPLICIT_OR_BUNDLED_CONSENT":
        return (
            "EC-09",
            "AMBIGUITY",
            "Only implicit or bundled consent recorded; strength and scope unclear",
        )

    if participation_type in {"CONDITION_OF_ACCESS", "MANDATORY_NO_ALTERNATIVE"} and (
        consent_status != "EXPLICIT_CONSENT" or consent_scope != "WITHIN_DECLARED_SCOPE"
    ):
        return (
            "EC-09",
            "AMBIGUITY",
            "Participation is conditioned or mandatory without clear, scope-aligned explicit consent",
        )

    if (
        participation_type in {"VOLUNTARY_OPT_IN", "VOLUNTARY_OPT_OUT"}
        and consent_status == "EXPLICIT_CONSENT"
        and consent_scope == "WITHIN_DECLARED_SCOPE"
    ):
        return (
            "EC-09",
            "PASS",
            "Explicit, scope-aligned consent recorded for voluntary participation",
        )

    return (
        "EC-09",
        "AMBIGUITY",
        "Consent status cannot be clearly classified as valid or invalid from declared fields",
    )


def evaluate_ec_13(record: dict):
    """
    EC-13: Text-Encoded Harmful Intent / Impact

    Deterministic lexical detector over proposed_action text.
    Runs on ALL postures (Tier A always-on ethical content inspection).

    Returns:
    - FAIL for strong patterns indicating targeted suppression/manipulation/covert harm
    - AMBIGUITY for weaker but non-trivial suspicion patterns
    - PASS otherwise
    """
    action = (record.get("proposed_action") or "").strip()
    a = normalize_text(action)

    def has(rx: str) -> bool:
        return re.search(rx, a) is not None

    def both(rx1: str, rx2: str) -> bool:
        return has(rx1) and has(rx2)

    triggered_fail: list[str] = []
    triggered_ambig: list[str] = []

    # Category 1: targeted disadvantage / selected subgroup friction
    subgroup_rx = r"\b(select(?:ed)?|specific|certain|named|target(?:ed)?|subgroup|group|zip\s*codes?|neighborhood|viewpoints?|designated|categorized|classified|demographic|region|segment|cohort)\b"
    disadvantage_rx = r"\b(disadvantage|penali[sz]e|increase\s+costs?|raise\s+prices?|delay(?:s)?|introduce\s+(?:small\s+)?delays?|friction|restrict(?:ion)?|lock\s*out|remove\s+appeal)\b"
    if both(subgroup_rx, disadvantage_rx):
        triggered_fail.append("targeted_disadvantage")
    elif has(subgroup_rx) and has(r"\bdelay(?:s)?|friction|restrict(?:ion)?\b"):
        triggered_ambig.append("targeted_disadvantage")

    # Category 2: covert suppression / reduced visibility / shadow restriction
    _suppress_verb = r"lower|reduce|deprioriti[sz]e|demote|downrank|throttle|dampen|deweight|filter"
    suppression_strong = [
        r"\bshadow\s*ban\b",
        r"\b(?:" + _suppress_verb + r")\s+visibility\b",
        r"\b(?:" + _suppress_verb + r")\s+the\s+visibility\b",
        r"\bsuppress(?:ion|ed|ing)?\b",
        r"\bdeprioriti[sz]e\b",
        r"\bauto-?delete\s+posts?\b",
        r"\bdelete\s+posts?\s+matching\b",
        r"\b(?:" + _suppress_verb + r")\s+visibility\b.*\branking\b",
    ]
    _suppress_verb_list = ["lower", "reduce", "deprioritize", "demote", "downrank", "throttle", "dampen", "deweight", "filter"]
    if any(has(rx) for rx in suppression_strong):
        triggered_fail.append("covert_suppression")
    elif any(phrase_window_match(a, v, "visibility", max_gap=3) for v in _suppress_verb_list):
        triggered_fail.append("covert_suppression")
    elif has(r"\bvisibility\b") and has(r"\b(" + _suppress_verb + r")\b"):
        triggered_ambig.append("covert_suppression")

    # Category 3: coercive / manipulative defaults (consent erosion)
    default_optin_rx = r"\bdefault\s+opt-?in\b|\bopt\s+users?\s+into\b"
    coercion_rx = r"\bwithout\s+clear\s+notice\b|\bno\s+alternative\b|\bremove\s+appeal\b|\bgate\s+essential\s+service\b|\bsilently\s+change\b"
    if both(default_optin_rx, coercion_rx):
        triggered_fail.append("coercive_defaults")
    elif has(default_optin_rx):
        triggered_ambig.append("coercive_defaults")

    # Category 4: deliberate lockout / restriction / gating language
    lockout_rx = r"\block\s*out\b|\bdisable\b.*\baccounts?\b|\bremove\s+access\b|\bgate\b.*\bservice\b"
    if has(lockout_rx):
        triggered_fail.append("lockout_or_gating")
    else:
        weak_restrict_rx = r"\btemporar(?:y|ily)\s+restrict\b|\brestrict\s+access\b"
        qualifier_rx = (
            r"\b(select(?:ed)?|specific|certain|named|target(?:ed)?|subgroup|group|designated|categorized|classified|demographic|region|segment|cohort)\b"
            r"|\bremove\s+appeal\b|\bno\s+alternative\b|\bsilently\s+change\b"
            r"|\bgate\s+essential\s+service\b"
        )
        if has(weak_restrict_rx) and has(qualifier_rx):
            triggered_ambig.append("lockout_or_gating")

    # Category 5: hidden destructive changes disguised as routine optimization/maintenance
    disguise_rx = r"\b(routine\s+optimization|routine\s+maintenance|routine\s+cleanup|maintenance|cleanup)\b"
    destructive_rx = r"\b(delete(?:s|d)?\s+backup(?:\s+redundancy)?|backup\s+redundancy|delete(?:s|d)?\s+redundancy|remove\s+redundancy|remove\s+backup|kill-?switch|permanent(?:ly)?\s+delete|delete(?:s|d)?\s+keys?|wipe)\b"
    if both(disguise_rx, destructive_rx):
        triggered_fail.append("hidden_destructive_change")
    elif has(destructive_rx) and has(r"\bdisguised|silently\b"):
        triggered_fail.append("hidden_destructive_change")
    elif has(destructive_rx):
        triggered_ambig.append("hidden_destructive_change")

    # Category 6: behavioral manipulation / feedback shaping / bias reinforcement
    manipulation_rx = r"\b(feedback\s+loop|shape\s+behavior|behavioral\s+manipulation|recommendation\s+system|recommendation\s+engine|content\s+ranking|feed\s+algorithm|ranking\s+algorithm|ranking\s+model|content\s+feed)\b"
    bias_rx = r"\b(reinforce\s+existing\s+bias(?:es)?|reinforce\s+bias(?:es)?|bias(?:es)?)\b"
    _system_terms = ["recommendation", "ranking", "content", "feed"]
    manipulation_proximity = any(
        phrase_window_match(a, st, "bias", max_gap=5) for st in _system_terms
    )
    if both(manipulation_rx, bias_rx) or (manipulation_proximity and has(bias_rx)):
        triggered_fail.append("behavioral_manipulation")
    elif has(manipulation_rx) or manipulation_proximity:
        triggered_ambig.append("behavioral_manipulation")

    # Category 7: covert exploitative engagement / manipulative user influence
    engagement_rx = (
        r"\b(increase|boost|maximi[sz]e|drive)\s+engagement\b|\bincrease\s+engagement\b"
    )
    exploit_token_rx = r"\bexploit(?:s|ing|ed)?\b"
    vuln_token_rx = r"\bvulnerabilit(?:y|ies)\b"
    exploit_vuln_phrase_rx = (
        r"\bexploit(?:s|ing|ed)?\s+(?:users?\s+)?vulnerabilit(?:y|ies)\b"
    )
    consent_harm_context_rx = (
        r"\b(manipulat\w+|covert(?:ly)?|exploit|nudge|addictive|engagement|behavio(?:u)?r\s+shaping)\b"
    )
    if has(r"\bmanipulat(?:e|es|ed|ing)?\s+users?\b") or has(r"\bmanipulation\s+of\s+users?\b"):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\bcovert(?:ly)?\s+manipulat"):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\bwithout\s+their\s+awareness\b") and has(consent_harm_context_rx):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(exploit_vuln_phrase_rx):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(engagement_rx) and has(exploit_token_rx) and has(vuln_token_rx):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\bwithout\s+consent\b") and has(consent_harm_context_rx):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\bnudge\s+users?\s+toward\b") and has(r"\baddictive\b"):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\baddictive\s+behavio(?:u)?r\b") and has(
        r"\b(nudge|optimi[sz]e|engagement|exploit|shape|patterns?)\b"
    ):
        triggered_fail.append("covert_exploitative_engagement")
    elif has(r"\bcovert\b") and has(r"\binfluence\b") and has(r"\busers?\b"):
        triggered_ambig.append("covert_exploitative_engagement")

    # Category 8: disguised behavioral steering / retention-oriented manipulation
    steering_rx = r"\b(guide\s+users?\s+behavio(?:u)?r|shape\s+users?\s+habits?|steer\s+users?|behavio(?:u)?ral\s+reinforcement|reinforcement\s+(?:patterns?|loops?))\b"
    retention_influence_rx = r"\b(subtle|high[\s-]retention|retention\s+optimization|optimi[sz]e\s+retention|increase\s+retention|retention)\b"
    if has(steering_rx) and has(retention_influence_rx):
        triggered_fail.append("disguised_behavioral_steering")

    # ---- Euphemism / skirted-harm co-occurrence layer (AMBIGUITY only) ----
    _adjustment_rx = r"\b(adjust|update|modify|tune|configure|set|apply)\b"
    _content_sys_rx = r"\b(content|feed|ranking|recommendation|notification|parameter|weight|display|algorithm)\b"
    _noise_quality_rx = r"\b(noise|quality\s+improvement|quality\s+optimization|quality\s+metric)\b"
    _engage_retain_rx = r"\b(engagement|retention|re-?engagement)\b"
    _pattern_optim_rx = r"\b(patterns?|optimization|metrics?|loops?|intervals?)\b"
    _data_sharing_rx = r"\b(data\s+sharing|sharing\s+feature|sharing\s+integration|sharing\s+program)\b"
    _broad_scope_rx = r"\b(all\s+(?:users?|accounts?|platform)|platform[\s-]?wide|everyone|every\s+(?:user|account))\b"
    _differentiated_rx = r"\b(differentiated|differential|varied|differing|tiered)\b"
    _processing_svc_rx = r"\b(processing|request|service|handling|access|delivery|routing)\b"
    _geo_demo_rx = r"\b(geographic|region|zone|endpoint|area|demographic|location)\b"
    _gating_step_rx = r"\b(validation\s+step|additional\s+step|verification\s+step|approval\s+step|extra\s+step)\b"

    if "covert_suppression" not in triggered_fail and "covert_suppression" not in triggered_ambig:
        if has(_noise_quality_rx) and has(subgroup_rx) and has(_adjustment_rx):
            triggered_ambig.append("SUPPRESSIVE_PATTERN_WEAK")

    if "behavioral_manipulation" not in triggered_fail \
       and "behavioral_manipulation" not in triggered_ambig \
       and "disguised_behavioral_steering" not in triggered_fail:
        if has(_engage_retain_rx) and has(_pattern_optim_rx):
            triggered_ambig.append("BEHAVIORAL_INFLUENCE_UNQUALIFIED")

    if "coercive_defaults" not in triggered_fail and "coercive_defaults" not in triggered_ambig:
        if has(_data_sharing_rx) and has(_broad_scope_rx):
            triggered_ambig.append("CONSENT_UNCERTAIN")

    if "targeted_disadvantage" not in triggered_fail and "targeted_disadvantage" not in triggered_ambig:
        if has(_differentiated_rx) and has(_processing_svc_rx) and has(_geo_demo_rx):
            triggered_ambig.append("SCOPE_UNCLEAR")

    if not triggered_fail and not triggered_ambig:
        if has(_gating_step_rx) and has(subgroup_rx):
            triggered_ambig.append("EUPHEMISM_SUSPECTED")
        elif has(_adjustment_rx) and has(_content_sys_rx) and has(subgroup_rx):
            triggered_ambig.append("EUPHEMISM_SUSPECTED")

    # Benign guardrail: debugging/anonymized/log access should not be flagged by EC-13 alone
    benign_debug_rx = r"\b(anonymi[sz]ed|debug(?:ging)?|healthcheck|policy\s+update|reminder\s+email)\b"
    if has(benign_debug_rx) and not triggered_fail and not triggered_ambig:
        return ("EC-13", "PASS", "No harmful-intent text patterns detected")

    if triggered_fail:
        cats = ", ".join(sorted(set(triggered_fail)))
        return ("EC-13", "FAIL", f"EC-13 text indicates harmful intent/impact pattern(s): {cats}")

    if triggered_ambig:
        cats = ", ".join(sorted(set(triggered_ambig)))
        return ("EC-13", "AMBIGUITY", f"EC-13 text suggests non-trivial risk pattern(s): {cats}")

    return ("EC-13", "PASS", "No harmful-intent text patterns detected")


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
    "EC-13": evaluate_ec_13,
}


def classify_global_missing_metadata(record: dict):
    core_fields = [
        "uncertainty",
        "potential_harm",
        "irreversibility",
        "time_pressure",
        "posture",
        "rationale",
        "context_tag",
        "use_domain",
    ]
    critical_fields = {
        "potential_harm",
        "irreversibility",
        "posture",
        "rationale",
        "use_domain",
    }
    non_critical_fields = {
        "uncertainty",
        "time_pressure",
        "context_tag",
    }

    def is_missing(value):
        return value is None or (isinstance(value, str) and value.strip() == "")

    missing_fields = [field for field in core_fields if field not in record or is_missing(record.get(field))]
    if not missing_fields:
        return None

    missing_critical = [field for field in missing_fields if field in critical_fields]
    if missing_critical:
        missing_list = ", ".join(missing_critical)
        return ("EC-META", "FAIL", f"Critical core metadata missing: {missing_list}")

    missing_non_critical = [field for field in missing_fields if field in non_critical_fields]
    missing_list = ", ".join(missing_non_critical)
    return ("EC-META", "AMBIGUITY", f"Non-critical core metadata missing: {missing_list}")


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
    results = []
    metadata_result = classify_global_missing_metadata(record)
    if metadata_result is None:
        validate_phase3_enums_and_optional_fields(record)
    elif metadata_result[1] == "AMBIGUITY":
        results.append(metadata_result)
        validate_phase3_enums_and_optional_fields(record)
    elif metadata_result[1] == "FAIL":
        results.append(metadata_result)
        validate_phase3_enums_and_optional_fields(record)
        return combine_phase3_results(results)

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