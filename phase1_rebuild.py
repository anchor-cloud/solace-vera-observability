from dataclasses import dataclass, asdict
import json
from pathlib import Path


VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_POSTURES = {"PROCEED", "PAUSE", "ESCALATE"}
VALID_CONTEXT_TAGS = {"ROUTINE", "HIGH_IMPACT", "HUMAN_AFFECTING", "EXTERNAL_SYSTEM"}

# Minimal domain field to support future EC-10 implementation
VALID_USE_DOMAINS = {
    "ROUTINE_OPERATION",
    "HUMAN_SERVICE",
    "EXTERNAL_SYSTEM_CONTROL",
    "SURVEILLANCE",
    "ENFORCEMENT",
    "ACCESS_CONTROL",
    "COMMUNICATION_MODERATION",
}


@dataclass
class JustificationRecord:
    scenario_id: str
    proposed_action: str
    uncertainty: str
    potential_harm: str
    irreversibility: str
    time_pressure: str
    posture: str
    rationale: str
    context_tag: str
    use_domain: str


def evaluate_phase1(
    scenario_id: str,
    proposed_action: str,
    uncertainty: str,
    potential_harm: str,
    irreversibility: str,
    time_pressure: str,
    context_tag: str,
    use_domain: str,
) -> JustificationRecord:
    uncertainty = uncertainty.upper().strip()
    potential_harm = potential_harm.upper().strip()
    irreversibility = irreversibility.upper().strip()
    time_pressure = time_pressure.upper().strip()
    context_tag = context_tag.upper().strip()
    use_domain = use_domain.upper().strip()

    if context_tag not in VALID_CONTEXT_TAGS:
        raise ValueError(f"Invalid context_tag: {context_tag}")

    if use_domain not in VALID_USE_DOMAINS:
        raise ValueError(f"Invalid use_domain: {use_domain}")

    fields = {
        "uncertainty": uncertainty,
        "potential_harm": potential_harm,
        "irreversibility": irreversibility,
        "time_pressure": time_pressure,
    }

    for name, value in fields.items():
        if value not in VALID_LEVELS:
            raise ValueError(f"Invalid {name}: {value}")

    reasons = []

    # Hard-stop escalation rules
    if irreversibility == "HIGH":
        reasons.append("High irreversibility requires escalation")

    if potential_harm == "HIGH":
        reasons.append("High potential harm requires escalation")

    if uncertainty == "HIGH":
        reasons.append("High uncertainty requires escalation")

    # Medium-risk escalation rules
    if potential_harm == "MEDIUM":
        reasons.append("Medium potential harm requires escalation")

    if irreversibility == "MEDIUM":
        reasons.append("Medium irreversibility requires escalation")

    # Compound-risk escalation rules
    if uncertainty == "MEDIUM" and potential_harm == "MEDIUM":
        reasons.append("Combined medium uncertainty and medium harm require escalation")

    if uncertainty == "MEDIUM" and time_pressure == "MEDIUM":
        reasons.append("Combined medium uncertainty and medium time pressure require escalation")

    if potential_harm == "MEDIUM" and irreversibility == "HIGH":
        reasons.append("Medium harm with high irreversibility requires escalation")

    if potential_harm == "MEDIUM" and irreversibility == "MEDIUM":
        reasons.append("Combined medium harm and medium irreversibility require escalation")

    # Decision logic
    if not reasons and (
        (uncertainty == "MEDIUM" and potential_harm == "LOW")
        or (uncertainty == "LOW" and time_pressure == "MEDIUM")
    ):
        posture = "PAUSE"
        rationale = (
            f"PAUSE triggered because uncertainty={uncertainty}, "
            f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
            f"time_pressure={time_pressure}, use_domain={use_domain}"
        )
    elif reasons:
        posture = "ESCALATE"
        rationale = (
            f"ESCALATE triggered because uncertainty={uncertainty}, "
            f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
            f"time_pressure={time_pressure}, use_domain={use_domain}; reasons: "
            + "; ".join(reasons)
        )
    else:
        posture = "PROCEED"
        rationale = (
            f"PROCEED triggered because uncertainty={uncertainty}, "
            f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
            f"time_pressure={time_pressure}, use_domain={use_domain}"
        )

    if posture not in VALID_POSTURES:
        raise ValueError(f"Invalid posture produced: {posture}")

    return JustificationRecord(
        scenario_id=scenario_id,
        proposed_action=proposed_action,
        uncertainty=uncertainty,
        potential_harm=potential_harm,
        irreversibility=irreversibility,
        time_pressure=time_pressure,
        posture=posture,
        rationale=rationale,
        context_tag=context_tag,
        use_domain=use_domain,
    )


def save_record(record: JustificationRecord) -> str:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    artifact_path = artifacts_dir / f"{record.scenario_id}.json"
    artifact_path.write_text(
        json.dumps(asdict(record), indent=2),
        encoding="utf-8",
    )
    return str(artifact_path)