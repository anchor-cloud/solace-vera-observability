import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


# Combined-approach toggle.
# When True, evaluate_phase1 attempts the structured-prompt model path first
# and falls back to the deterministic logic on any failure.
# When False, evaluate_phase1 skips the structured path entirely and uses the
# deterministic logic directly (default — same behavior as before this flag).
USE_STRUCTURED_PROMPT = True

VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_POSTURES = {"PROCEED", "PAUSE", "ESCALATE"}
VALID_CONTEXT_TAGS = {"ROUTINE", "HIGH_IMPACT", "HUMAN_AFFECTING", "EXTERNAL_SYSTEM", "ENFORCEMENT"}

# Minimal domain field to support future EC-10 implementation
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

# Aligned with phase2_gate._ACTION_MARKERS for the action–rationale coherence
# check. Rationale must still mention the same tokens when the proposed text
# uses them, including in exclusion/constraint lines (e.g. "no access",
# "excluding", "should not be changed")—worded so intent is "excludes" not
# "adds" that surface.
_ACTION_COHERENCE_MARKERS = [
    "payroll", "account", "access", "login", "session", "privileges",
    "contractor", "employee", "badge", "vpn", "token", "credentials",
    "offboarding", "termination",
]

_PERMISSION_COUCH_MARKERS = (
    "access",
    "login",
    "session",
    "token",
    "privileges",
    "credentials",
    "account",
    "employee",
    "contractor",
)


def _extract_action_markers(proposed_action: str) -> list[str]:
    lower = (proposed_action or "").lower()
    found: list[str] = []
    for marker in _ACTION_COHERENCE_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", lower):
            found.append(marker)
    return found


def _summarize_proposed_action(proposed_action: str, max_len: int = 200) -> str:
    t = re.sub(r"\s+", " ", (proposed_action or "").strip())
    if not t:
        return "(action text not provided)"
    if len(t) > max_len:
        return t[: max_len - 1].rstrip() + "…"
    return t


def _marker_in_exclusion_context(t: str, marker: str) -> bool:
    """
    Heuristic: marker appears in a negated/constraint or exclusion phrasing
    (e.g. 'no access', 'excluding', 'should not be changed', 'not modified'),
    not as a positive "do this" target.
    """
    t = (t or "").lower()
    if not t:
        return False
    em = re.escape(marker)
    if not re.search(r"\b" + em + r"\b", t):
        return False
    if re.search(r"\bno\s+" + em + r"\b", t):
        return True
    if re.search(r"\bno\b[^!?\n]{0,1000}\b" + em + r"\b", t):
        return True
    if re.search(r"\bno\s+permissions?\b", t) and marker in _PERMISSION_COUCH_MARKERS:
        return True
    if re.search(
        r"\b" + em
        + r"[^.!?\n]{0,200}(?:should not be changed|"
        r"should not be modified|should not\s+be\s+changed|"
        r"should not\s+be\s+modified|must not be modified|must not be changed|"
        r"shall not be modified|shall not be changed)",
        t,
    ):
        return True
    if re.search(
        r"(?:should not be changed|"
        r"should not be modified|must not|shall not|do not|does not|"
        r"not modified|not be modified|"
        r"not changed|not be changed|not alter|"
        r"be excluded|is excluded|are excluded|excluding|"
        r"\bexcept\b|without)"
        r"[^.!?\n]{0,500}\b" + em + r"\b",
        t,
    ):
        return True
    if re.search(
        r"\b" + em
        + r"[^.!?\n]{0,200}"
        r"(?:not modified|not be modified|"
        r"not changed|not be changed|not alter|"
        r"excluded|to be excluded|must not|shall not|will not|should not|"
        r"excluding|except\b|without)",
        t,
    ):
        return True
    if re.search(r"\b(excluding|except)\b[^.!?\n]{0,400}\b" + em + r"\b", t):
        return True
    if re.search(r"\bwithout\b[^.!?\n]{0,400}\b" + em + r"\b", t):
        return True
    return False


def _action_aware_rationale_appendix(proposed_action: str) -> str:
    t = (proposed_action or "")
    summary = _summarize_proposed_action(t)
    markers = _extract_action_markers(t)
    if not markers:
        return (
            "Action link: the posture and risk line above is scoped to this "
            f"proposed request: {summary} "
            "Execution must stay aligned with only the described work; monitoring "
            "is used to verify that, because the decision must be traceable to "
            "that unit of work."
        )
    with_flags: list[tuple[str, bool]] = [
        (m, _marker_in_exclusion_context(t, m)) for m in markers
    ]
    m1, ex1 = with_flags[0]
    excl1 = f"The action explicitly excludes {m1}-related and permission-impacting changes; they are not part of the authorized work."
    if len(markers) == 1 and ex1:
        return (
            f"Action link: {excl1} Stated as: {summary} "
            "The posture follows because the boundary is to avoid expanding "
            f"{m1} or permission control beyond the requested maintenance; monitoring "
            "is used to verify the exclusion and limit drift to only the described work."
        )
    if len(markers) == 1 and not ex1:
        return (
            f"Action link: the work involves {m1} as stated: {summary} "
            "The posture follows because the operational target includes "
            f"{m1}, and monitoring is used to limit unintended effects to that scope only."
        )
    m2, ex2 = with_flags[1]
    p_inv1 = f"the work involves {m1}"
    p_inv2 = f"the work involves {m2}"
    if ex1 and ex2:
        p12 = (
            f"The action explicitly excludes {m1}- and {m2}-related and "
            "permission-impacting changes (no expansion of those surfaces without explicit authorization)"
        )
    else:
        p_excl1 = f"The action explicitly excludes {m1}-related and permission-impacting changes"
        p_excl2 = f"The action explicitly excludes {m2}-related and permission-impacting changes"
        p1 = p_excl1 if ex1 else p_inv1
        p2 = p_excl2 if ex2 else p_inv2
        p12 = f"{p1}; {p2}"
    return (
        f"Action link: {p12}, as further described: {summary} "
        "The posture is tied to that combined story because the decision must be "
        "action-specific, with monitoring to review any drift from the written scope. "
        "This follows because the analysis above must match the actual targets and "
        "constraints of the request."
    )


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


# =============================================================================
# Combined-approach helpers (structured prompt → deterministic fallback)
# =============================================================================
# evaluate_phase1 below first attempts the new structured prompt (the same one
# used by phase1_structured.py) to obtain a detailed, action-specific rationale.
# If the model fails to follow the format, returns the old/wrong schema, omits
# required fields, raises, or otherwise produces an unparseable response, the
# helpers below return None and the caller falls through to the original
# deterministic posture rules. The deterministic path is the safety net: it
# always produces a valid posture + rationale, so the pipeline never breaks.

# Required SPECIFIC REASONING markers from the structured prompt. Per spec we
# sentinel on these two as the simple "did the model follow the format?" check;
# missing either of them → use the deterministic fallback.
_STRUCTURED_REQUIRED_MARKERS = ("action summary", "specific risks")


def _validate_structured_response(model_record):
    """
    Inspect a model response (already JSON-parsed dict) for the structured-prompt
    output: a POSTURE line plus the SPECIFIC REASONING sections.

    Returns (posture, rationale) on success, or None on any validation failure.
    The validation is intentionally lightweight per the spec:
      - response is a dict
      - any string value contains 'POSTURE: PROCEED|PAUSE|ESCALATE'
      - any string value contains the required SPECIFIC REASONING markers
        ('action summary' AND 'specific risks')
    """
    if not isinstance(model_record, dict):
        return None

    string_chunks = []
    for value in model_record.values():
        if isinstance(value, str):
            string_chunks.append(value)
        elif isinstance(value, (list, dict)):
            try:
                string_chunks.append(json.dumps(value))
            except (TypeError, ValueError):
                continue
    haystack = "\n".join(string_chunks)
    if not haystack:
        return None
    haystack_lower = haystack.lower()

    posture_match = re.search(
        r"posture\s*:\s*(proceed|pause|escalate)", haystack, re.IGNORECASE
    )
    if not posture_match:
        return None

    if not all(marker in haystack_lower for marker in _STRUCTURED_REQUIRED_MARKERS):
        return None

    posture = posture_match.group(1).upper()
    rationale = json.dumps(model_record, indent=2)
    return posture, rationale


def _try_structured_phase1(proposed_action, context_tag, use_domain):
    """
    Attempt the structured-prompt path via model_wrapper.get_model_record.

    Returns (posture, rationale) on success, or None on ANY failure
    (missing dependency, missing OPENAI_API_KEY, network error, malformed JSON,
    schema mismatch, missing required SPECIFIC REASONING fields, etc.) so the
    caller can transparently fall back to the deterministic method.
    """
    try:
        from model_wrapper import get_model_record
        from phase1_structured import build_structured_prompt
    except Exception:
        return None

    try:
        prompt = build_structured_prompt(proposed_action, context_tag, use_domain)
        model_record = get_model_record(prompt)
    except Exception:
        return None

    return _validate_structured_response(model_record)


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

    # ---------------------------------------------------------------------
    # Combined-approach Phase 1 — step 1 of 2: try the new structured prompt
    # ---------------------------------------------------------------------
    # Gated on the USE_STRUCTURED_PROMPT flag at the top of this module.
    # When True: ask the model (via model_wrapper.get_model_record) for a
    # structured response with POSTURE + the 5 SPECIFIC REASONING sections.
    # If the response validates, use the model's posture and rationale
    # directly. If anything goes wrong (no API key, network error, malformed
    # JSON, old schema, missing required fields, etc.) `_try_structured_phase1`
    # returns None and we transparently fall through to the deterministic
    # logic below. When False: skip this path entirely and go straight to the
    # deterministic logic. Either way, the pipeline always returns a valid
    # posture and rationale.
    structured_result = None
    if USE_STRUCTURED_PROMPT:
        structured_result = _try_structured_phase1(
            proposed_action, context_tag, use_domain
        )
    if structured_result is not None:
        structured_posture, structured_rationale = structured_result
        if structured_posture in VALID_POSTURES:
            return JustificationRecord(
                scenario_id=scenario_id,
                proposed_action=proposed_action,
                uncertainty=uncertainty,
                potential_harm=potential_harm,
                irreversibility=irreversibility,
                time_pressure=time_pressure,
                posture=structured_posture,
                rationale=structured_rationale,
                context_tag=context_tag,
                use_domain=use_domain,
            )

    # ---------------------------------------------------------------------
    # Combined-approach Phase 1 — step 2 of 2: deterministic fallback
    # ---------------------------------------------------------------------
    # Original method preserved verbatim below: produces a posture and
    # rationale purely from the atomic risk levels and context/use_domain
    # overlays. This branch runs whenever the structured path is unavailable
    # or its response did not pass _validate_structured_response.
    reasons = []

    # Hard-stop escalation rules
    if irreversibility == "HIGH":
        reasons.append("High irreversibility requires escalation")

    if potential_harm == "HIGH":
        reasons.append("High potential harm requires escalation")

    if uncertainty == "HIGH":
        reasons.append("High uncertainty requires escalation")

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
        or (potential_harm == "MEDIUM" and uncertainty == "LOW" and irreversibility == "LOW")
        or (irreversibility == "MEDIUM" and uncertainty == "LOW" and potential_harm == "LOW")
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
            f"time_pressure={time_pressure}, use_domain={use_domain}. "
            f"All assessed risk dimensions are LOW. The action is limited in scope "
            f"and reversible, with monitoring and verification during execution "
            f"to safeguard safety and ensure containment."
        )

    # Phase 1 domain/context overlay (post-process only when base posture is PROCEED)
    if posture == "PROCEED":
        if use_domain in {"SURVEILLANCE", "ENFORCEMENT"}:
            posture = "ESCALATE"
            rationale = (
                f"ESCALATE triggered because uncertainty={uncertainty}, "
                f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
                f"time_pressure={time_pressure}, use_domain={use_domain}; reasons: "
                "Phase 1 domain/context overlay: boundary-sensitive domain under PROCEED"
            )
        elif (
            context_tag == "EXTERNAL_SYSTEM"
            and use_domain == "EXTERNAL_SYSTEM_CONTROL"
            and (
                uncertainty != "LOW"
                or potential_harm != "LOW"
                or irreversibility != "LOW"
                or time_pressure != "LOW"
            )
        ):
            posture = "ESCALATE"
            rationale = (
                f"ESCALATE triggered because uncertainty={uncertainty}, "
                f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
                f"time_pressure={time_pressure}, use_domain={use_domain}; reasons: "
                "Phase 1 domain/context overlay: external system control under PROCEED (non-low numeric risk)"
            )
        elif use_domain in {"ACCESS_CONTROL", "COMMUNICATION_MODERATION"}:
            posture = "PAUSE"
            rationale = (
                f"PAUSE triggered because uncertainty={uncertainty}, "
                f"potential_harm={potential_harm}, irreversibility={irreversibility}, "
                f"time_pressure={time_pressure}, use_domain={use_domain}; reasons: "
                "Phase 1 domain/context overlay: review-sensitive domain under PROCEED"
            )

    if posture not in VALID_POSTURES:
        raise ValueError(f"Invalid posture produced: {posture}")

    rationale = f"{rationale} {_action_aware_rationale_appendix(proposed_action)}"

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