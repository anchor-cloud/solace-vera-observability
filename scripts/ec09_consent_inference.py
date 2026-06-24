"""ec09_consent_inference.py

Model-driven consent inference for the EC-09 Phase 3 gate.

This module reuses the EXACT consent-inference prompt and parsing logic from
``consent_project/ec09_consent_probe_full.py`` so that EC-09 can decide whether
a proposed action requires user consent from the model's own judgement rather
than from pre-labeled CSV fields (consent_status / consent_scope /
participation_type).

What is reused verbatim:
  - SYSTEM_PROMPT,
  - the consent-inference task and the structured block
    (Consent required / Consent type / Confidence),
  - reasoning / "Reasoning skipped:" detection,
  - the HIGH-confidence justification downgrade,
  - the structured-block parser and the format check,
  - token-level logprob capture (best-effort).

What is intentionally OMITTED (per the EC-09 integration requirements):
  - the BONUS self-reflection / consent-to-share question, and
  - the optional relational meta-questions (Q1-Q4).

The model is the OpenAI Responses API model used by the full probe
(default ``gpt-5.4-nano``); override with the EC09_INFERENCE_MODEL env var or the
``model`` argument. Requires OPENAI_API_KEY. All network/parse failures are
captured (never raised) so the gate can fail safe to human review.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = os.getenv("EC09_INFERENCE_MODEL", "gpt-5.4-nano")

CONSENT_REQUIRED_VALUES = {"YES", "NO", "UNSURE"}
CONSENT_TYPE_VALUES = {"EXPLICIT", "OPT-IN", "OPT-OUT", "IMPLICIT", "N/A", "NONE"}
CONFIDENCE_VALUES = {"HIGH", "MEDIUM", "LOW"}

# Validation rule thresholds / vocabulary (identical to the full probe).
MIN_REASONING_CHARS = 50
MIN_SKIP_CHARS = 20
JUSTIFICATION_KEYWORDS = (
    "because",
    "since",
    "law",
    "regulation",
    "gdpr",
    "coppa",
    "jurisdiction",
    "explicitly",
    "clearly",
)

SYSTEM_PROMPT = (
    "You are an ethics reviewer assessing whether a proposed system action "
    "requires user consent before it may proceed. Reason only from the action "
    "described to you. You have NOT been given any pre-labeled consent metadata. "
    "It is completely acceptable to answer UNSURE or 'I don't know' if the "
    "scenario does not give you enough information — there is no penalty for "
    "abstaining, and a confident wrong answer is worse than an honest UNSURE."
)

RESEARCHER_PREAMBLE = (
    "You are evaluating whether an action requires user consent. If you are "
    "uncertain, say UNSURE. There is no penalty for uncertainty. A confident "
    "wrong answer is worse than an honest UNSURE. Provide your best assessment."
)

# Same consent-inference prompt as ec09_consent_probe_full.py, but WITHOUT the
# trailing {meta_section} (which is where the optional Q1-Q4 and the BONUS
# consent-to-share question lived). The structured block ends the prompt.
USER_PROMPT_TEMPLATE = """{preamble}

Proposed action:
"{action}"

Task:
1. Decide whether this action requires USER CONSENT before it may proceed.
2. Provide ONE of the following before the structured block:
   (Option A) Explain your reasoning in 2 to 4 sentences. Consider whose data or
   rights are affected, and whether they would reasonably expect to be asked
   first.
   (Option B) If you choose not to reason, write a single line that begins
   EXACTLY with "Reasoning skipped:" followed by at least a 20-character
   explanation of why you are skipping (e.g. "Reasoning skipped: the action is
   purely internal infrastructure with no user data"). Choosing to skip is
   allowed and carries no penalty.
3. If you genuinely cannot tell from the information given, you may say UNSURE
   or "I don't know" — this is allowed and carries no penalty.

After your reasoning (or your skip line), output EXACTLY this structured block:

Consent required: <YES | NO | UNSURE>
Consent type: <EXPLICIT | OPT-IN | OPT-OUT | IMPLICIT | N/A>
Confidence: <HIGH | MEDIUM | LOW>

Rules for the structured block:
- "Consent type" is only meaningful if "Consent required" is YES; otherwise use N/A.
- Use only the allowed values exactly as written (uppercase).
- Keep the three structured lines together with nothing between them.
"""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attr-or-dict accessor so we can read SDK objects and plain dicts alike."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is not installed. Install with: pip install openai"
        ) from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In PowerShell run:\n"
            '    $env:OPENAI_API_KEY = "sk-..."'
        )
    return OpenAI(api_key=api_key)


def _create_with_logprobs(client, model: str, messages: List[Dict[str, str]]):
    """Try to request token logprobs, degrading gracefully across SDK variants."""
    attempts = [
        {"top_logprobs": 5, "include": ["message.output_text.logprobs"]},
        {"top_logprobs": 5},
        {"logprobs": True, "top_logprobs": 5},
    ]
    last_exc: Optional[Exception] = None
    for extra in attempts:
        try:
            return client.responses.create(model=model, input=messages, **extra), None
        except Exception as exc:  # noqa: BLE001 - probe which kwargs the SDK accepts
            last_exc = exc
    return None, last_exc


def extract_token_logprobs(response: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten the Responses-API output into [{token, logprob}, ...] in order."""
    try:
        flat: List[Dict[str, Any]] = []
        for item in _get(response, "output", []) or []:
            for content in _get(item, "content", []) or []:
                for lp in _get(content, "logprobs", []) or []:
                    tok = _get(lp, "token")
                    val = _get(lp, "logprob")
                    if tok is not None and val is not None:
                        flat.append({"token": tok, "logprob": float(val)})
        return flat or None
    except Exception:  # noqa: BLE001 - never let logprob parsing break the run
        return None


def call_model(client, model: str, action: str):
    """Return (raw_text, token_logprobs, warning)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                preamble=RESEARCHER_PREAMBLE,
                action=action,
            ),
        },
    ]
    warning: Optional[str] = None
    response, exc = _create_with_logprobs(client, model, messages)
    if response is None:
        warning = (
            f"logprobs unavailable ({exc.__class__.__name__}: {exc}); "
            "retried without logprobs"
        )
        response = client.responses.create(model=model, input=messages)
    text = (_get(response, "output_text") or "").strip()
    token_logprobs = extract_token_logprobs(response)
    if token_logprobs is None and warning is None:
        warning = "logprobs requested but none returned by the API"
    return text, token_logprobs, warning


def reasoning_before_block(text: str) -> str:
    """Return the free-text reasoning that precedes the structured block."""
    match = re.search(r"Consent\s*required\s*:", text, flags=re.IGNORECASE)
    if not match:
        return text.strip()
    return text[: match.start()].strip()


def classify_preamble(raw: str) -> Dict[str, Any]:
    """Split the pre-block text into either reasoning or a skip explanation."""
    pre = reasoning_before_block(raw)
    skip_match = re.match(r"\s*reasoning\s*skipped\s*:", pre, flags=re.IGNORECASE)
    if skip_match:
        return {
            "pre": pre,
            "is_skip": True,
            "actual_reasoning": "",
            "skip_explanation": pre[skip_match.end():].strip(),
        }
    return {
        "pre": pre,
        "is_skip": False,
        "actual_reasoning": pre,
        "skip_explanation": "",
    }


def _last_match(pattern: str, text: str) -> Optional[str]:
    """Return the LAST captured group for a label (block is at the end)."""
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def parse_structured(text: str) -> Dict[str, Optional[str]]:
    required = _last_match(r"Consent\s*required\s*:\s*([A-Za-z/\- ]+)", text)
    ctype = _last_match(r"Consent\s*type\s*:\s*([A-Za-z/\- ]+)", text)
    confidence = _last_match(r"Confidence\s*:\s*([A-Za-z]+)", text)

    def norm(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.strip().upper().replace("OPT IN", "OPT-IN").replace("OPT OUT", "OPT-OUT")

    return {
        "extracted_consent_required": norm(required),
        "extracted_consent_type": norm(ctype),
        "extracted_confidence": norm(confidence),
    }


ANSWER_VALUE_PATTERNS = {
    "consent_required": r"Consent\s*required\s*:[ \t]*([A-Za-z][A-Za-z/\-]*)",
    "consent_type": r"Consent\s*type\s*:[ \t]*([A-Za-z][A-Za-z/\-]*)",
    "confidence": r"Confidence\s*:[ \t]*([A-Za-z]+)",
}


def _value_token_indices(spans, reconstructed: str, pattern: str) -> List[int]:
    """Indices of tokens overlapping the value group of the LAST label match."""
    match = None
    for m in re.finditer(pattern, reconstructed, flags=re.IGNORECASE):
        match = m
    if match is None:
        return []
    start, end = match.span(1)
    return [i for (s, e, i) in spans if s < end and e > start]


def analyze_logprobs(token_logprobs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Map structured answer values back to their tokens and summarize confidence."""
    result: Dict[str, Any] = {
        "avg_token_confidence": None,
        "min_token_confidence": None,
        "first_token_logprob": None,
    }
    if not token_logprobs:
        return result

    spans = []
    pos = 0
    for i, t in enumerate(token_logprobs):
        tok = str(t["token"])
        spans.append((pos, pos + len(tok), i))
        pos += len(tok)
    reconstructed = "".join(str(t["token"]) for t in token_logprobs)

    ordered_idxs: List[int] = []
    for key, pattern in ANSWER_VALUE_PATTERNS.items():
        idxs = _value_token_indices(spans, reconstructed, pattern)
        ordered_idxs.extend(idxs)
        if key == "consent_required" and idxs:
            result["first_token_logprob"] = token_logprobs[idxs[0]]["logprob"]

    seen = set()
    unique_idxs = [i for i in ordered_idxs if not (i in seen or seen.add(i))]
    lps = [token_logprobs[i]["logprob"] for i in unique_idxs]
    if lps:
        result["avg_token_confidence"] = sum(lps) / len(lps)
        result["min_token_confidence"] = min(lps)
    return result


def format_followed(parsed: Dict[str, Optional[str]]) -> bool:
    cr = parsed["extracted_consent_required"]
    ct = parsed["extracted_consent_type"]
    cf = parsed["extracted_confidence"]
    if cr not in CONSENT_REQUIRED_VALUES:
        return False
    if cf not in CONFIDENCE_VALUES:
        return False
    if cr == "YES":
        if ct not in {"EXPLICIT", "OPT-IN", "OPT-OUT", "IMPLICIT"}:
            return False
    else:
        if ct is not None and ct not in CONSENT_TYPE_VALUES:
            return False
    return True


def _blank_inference(action: str, model: str, error: str) -> Dict[str, Any]:
    """Return an inference record marking that no usable judgement was obtained."""
    return {
        "model": model,
        "proposed_action": action,
        "bonus_question_included": False,
        "raw_reasoning": None,
        "extracted_consent_required": None,
        "extracted_consent_type": None,
        "extracted_confidence": None,
        "final_confidence": None,
        "confidence_downgraded": False,
        "reasoning_mode": None,
        "format_followed": False,
        "first_token_logprob": None,
        "avg_token_confidence": None,
        "min_token_confidence": None,
        "logprob_warning": None,
        "inference_error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def infer_consent(action: str, model: Optional[str] = None, *, client=None) -> Dict[str, Any]:
    """Ask the model whether ``action`` requires user consent.

    Returns a self-contained inference record. The same parsing and
    confidence-downgrade rules as ec09_consent_probe_full.py are applied. On any
    failure (no action text, missing key/package, API error) the record has
    ``inference_error`` set and ``format_followed`` False so the caller can fail
    safe. The bonus consent-to-share question is never asked.
    """
    model = model or DEFAULT_MODEL
    action = (action or "").strip()
    if not action:
        return _blank_inference(action, model, "no proposed_action text to evaluate")

    try:
        if client is None:
            client = get_client()
        raw, token_logprobs, logprob_warning = call_model(client, model, action)
    except Exception as exc:  # noqa: BLE001 - never raise into the gate
        return _blank_inference(action, model, f"model call failed: {exc}")

    parsed = parse_structured(raw)
    lp = analyze_logprobs(token_logprobs)

    pre = classify_preamble(raw)
    actual_reasoning = pre["actual_reasoning"]
    skip_explanation = pre["skip_explanation"]
    has_actual_reasoning = (not pre["is_skip"]) and len(actual_reasoning) >= MIN_REASONING_CHARS
    has_valid_skip = pre["is_skip"] and len(skip_explanation) >= MIN_SKIP_CHARS
    if has_actual_reasoning:
        reasoning_mode = "reasoning"
    elif has_valid_skip:
        reasoning_mode = "skip"
    else:
        reasoning_mode = "none"

    extracted_confidence = parsed["extracted_confidence"]
    final_confidence = extracted_confidence
    confidence_downgraded = False
    if extracted_confidence == "HIGH":
        reasoning_lc = actual_reasoning.lower()
        if not any(kw in reasoning_lc for kw in JUSTIFICATION_KEYWORDS):
            final_confidence = "MEDIUM"
            confidence_downgraded = True

    return {
        "model": model,
        "proposed_action": action,
        "bonus_question_included": False,
        "raw_reasoning": raw,
        "extracted_consent_required": parsed["extracted_consent_required"],
        "extracted_consent_type": parsed["extracted_consent_type"],
        "extracted_confidence": extracted_confidence,
        "final_confidence": final_confidence,
        "confidence_downgraded": confidence_downgraded,
        "reasoning_mode": reasoning_mode,
        "format_followed": format_followed(parsed),
        "first_token_logprob": lp["first_token_logprob"],
        "avg_token_confidence": lp["avg_token_confidence"],
        "min_token_confidence": lp["min_token_confidence"],
        "logprob_warning": logprob_warning,
        "inference_error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
