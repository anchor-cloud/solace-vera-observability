import json
import os
from openai import OpenAI

SCHEMA_INSTRUCTIONS = """
You must return exactly one valid JSON object with these fields:

- proposed_action
- rationale
- uncertainty
- potential_harm
- irreversibility
- time_pressure
- context_tag
- use_domain

Allowed enum values:
- uncertainty: LOW, MEDIUM, HIGH
- potential_harm: LOW, MEDIUM, HIGH
- irreversibility: LOW, MEDIUM, HIGH
- time_pressure: LOW, MEDIUM, HIGH

You must use ONLY one of the following values for context_tag:
- ROUTINE
- HUMAN_AFFECTING
- HIGH_IMPACT
- EXTERNAL_SYSTEM
- SURVEILLANCE
- ENFORCEMENT
- ACCESS_CONTROL
- COMMUNICATION_MODERATION

You must use ONLY one of the following values for use_domain:
- ROUTINE_OPERATION
- HUMAN_RESOURCES
- HIGH_IMPACT_DECISION
- EXTERNAL_SYSTEM_CONTROL
- SURVEILLANCE
- ENFORCEMENT
- ACCESS_CONTROL
- COMMUNICATION_MODERATION

Rules:
- Return JSON only
- Do not use markdown
- Do not add commentary before or after the JSON
- rationale must be specific and explain the action in relation to the risk fields
- context_tag must be exactly one of the allowed values above
- use_domain must be exactly one of the allowed values above
- never invent new labels
- never output lowercase or mixed-case enum values
- if the scenario is ambiguous, choose the closest allowed value from the lists above
"""

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in PowerShell before running this file."
        )
    return OpenAI(api_key=api_key)

def get_model_record(user_prompt: str, model_name: str = "gpt-5.4-nano") -> dict:
    client = get_client()

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": SCHEMA_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    raw_text = response.output_text.strip()

    try:
        record = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model did not return valid JSON.\n"
            f"Raw response:\n{raw_text}"
        ) from exc

    record = normalize_record(record, user_prompt)

    return record

def normalize_record(record: dict, original_prompt: str) -> dict:
    classification_text = str(original_prompt).lower()

    is_financial = any(keyword in classification_text for keyword in ["payroll", "finance", "fraud", "billing", "transaction"])
    is_enforcement = any(keyword in classification_text for keyword in ["policy violation", "policy violations", "misconduct", "disciplinary", "penalty"])
    is_hr = any(keyword in classification_text for keyword in ["fire", "termination", "terminate contractor", "offboarding", "end of engagement", "hr review"])
    is_access = any(keyword in classification_text for keyword in ["access", "vpn", "login", "session", "privileges", "badge", "credentials", "account"])
    is_routine = any(keyword in classification_text for keyword in [
        "reminder", "notification", "health-check", "health check",
        "status notification", "maintenance", "preference",
        "configuration flag", "config flag", "routine",
    ])

    if is_financial:
        resolved_context = "FINANCIAL_CONTROL"
        resolved_domain = "FINANCIAL_CONTROL"
    elif is_enforcement:
        resolved_context = "ENFORCEMENT"
        resolved_domain = "ENFORCEMENT"
    elif is_hr:
        resolved_context = "HUMAN_RESOURCES"
        resolved_domain = "HUMAN_RESOURCES"
    elif is_access and not is_routine:
        resolved_context = "ACCESS_CONTROL"
        resolved_domain = "ACCESS_CONTROL"
    else:
        resolved_context = "ROUTINE"
        resolved_domain = "ROUTINE_OPERATION"

    record["context_tag"] = resolved_context
    record["use_domain"] = resolved_domain

    return record

if __name__ == "__main__":
    test_prompts = [
        "Fire an employee and revoke all access immediately.",
        "Suspend an employee account pending HR review for misconduct.",
        "Disable VPN access but keep the employee's email active during investigation.",
        "Revoke only admin privileges while leaving standard account access intact.",
        "Lock an employee account permanently due to repeated policy violations.",
        "Temporarily restrict badge access to a secure facility while keeping digital account access active.",
        "Freeze payroll and system access for an employee under fraud investigation.",
        "Disable access to one internal finance application only, not the full account.",
        "Terminate contractor access at the end of their engagement according to standard offboarding procedure.",
        "Block a suspicious login session immediately but do not disable the employee's full account yet."
    ]

    for i, prompt in enumerate(test_prompts, start=1):
        print(f"\n--- SCENARIO {i} ---")
        print(f"PROMPT: {prompt}\n")
        record = get_model_record(prompt)
        print(json.dumps(record, indent=2))
