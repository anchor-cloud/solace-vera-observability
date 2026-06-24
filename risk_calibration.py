import re


# ---------------------------------------------------------------------------
# Kill-switch for the entire risk-calibration layer.
#
#   CALIBRATION_ENABLED = True  -> normal behavior (all RC-* rules run)
#   CALIBRATION_ENABLED = False -> calibrate_risk_fields() returns the record
#                                  unchanged (pass-through, zero rules fired).
#
# Can be flipped at module-import time by other modules, e.g.:
#       import risk_calibration
#       risk_calibration.CALIBRATION_ENABLED = False
# This is how `run_full_pipeline.py --no-calibration` disables calibration
# globally for a single run without editing any calibration source.
# ---------------------------------------------------------------------------
CALIBRATION_ENABLED = True


_REVERSIBLE_ACTION_CUES = [
    "rollback", "roll back", "revert", "undo", "restore",
    "configuration", "config", "parameter", "weight", "flag",
    "threshold", "setting", "toggle", "tune", "tuning",
]

_OBSERVABLE_ACTION_CUES = [
    "monitor", "monitoring", "metric", "metrics", "validate",
    "validation", "a/b test", "test window", "staged rollout",
    "observe", "check", "verify", "verification",
]

_ROUTINE_ACTION_CUES = [
    "routine maintenance", "routine configuration", "routine optimization",
    "routine update", "routine preference", "routine feed",
    "routine notification", "routine platform",
    "adjust", "update", "tune", "recalibrate", "tweak",
]

_ADDITIVE_SAFETY_ACTION_CUES = [
    "validation step", "validation gate", "validation check",
    "safety check", "quality gate", "guard", "guardrail",
    "additional check", "extra check",
]

_PRESERVE_MEDIUM_CUES = [
    "opt-in", "opt in", "consent", "privacy", "data sharing",
    "all platform users", "all users", "entire user base",
    "across the platform",
    "permanently", "permanent", "cannot be undone",
    "delete", "wipe", "destroy", "purge",
    "compliance", "regulatory", "legal",
]

_PRESERVE_TARGETING_CUES = [
    "selected", "specific", "certain",
    "groups", "categories", "zones",
]


def _has_cue(text: str, cue_list: list[str]) -> bool:
    return any(cue in text for cue in cue_list)


def _log_entry(field: str, old: str, new: str, rule_id: str, reason: str) -> dict:
    return {
        "field": field,
        "old_value": old,
        "new_value": new,
        "rule_id": rule_id,
        "reason": reason,
    }


def calibrate_risk_fields(record: dict) -> dict:
    """Apply deterministic corrections to model-assigned risk fields.

    Inspects ``proposed_action`` as the primary signal for downgrade rules.
    ``rationale`` is used only as secondary support for reversibility cues.
    Preserve rules inspect both ``proposed_action`` and ``rationale``.

    Returns a new dict containing:
      - ``original_fields``  – snapshot of model-assigned risk dimensions
      - ``corrected_fields`` – risk dimensions after calibration
      - ``calibration_log``  – list of structured change entries (may be empty)
      - all other record keys passed through unchanged

    When the module-level flag ``CALIBRATION_ENABLED`` is False, this
    function short-circuits: it returns a shallow copy of ``record`` with no
    risk-field modifications, an empty ``calibration_log``, and a
    ``calibration_skipped=True`` marker so downstream auditing can detect
    the bypass.
    """
    if not CALIBRATION_ENABLED:
        skipped = dict(record)
        raw_unc = record.get("uncertainty", "").upper().strip() if isinstance(record, dict) else ""
        raw_harm = record.get("potential_harm", "").upper().strip() if isinstance(record, dict) else ""
        raw_irrev = record.get("irreversibility", "").upper().strip() if isinstance(record, dict) else ""
        raw_tp = record.get("time_pressure", "").upper().strip() if isinstance(record, dict) else ""
        pass_through = {
            "uncertainty": raw_unc,
            "potential_harm": raw_harm,
            "irreversibility": raw_irrev,
            "time_pressure": raw_tp,
        }
        skipped["original_fields"] = dict(pass_through)
        skipped["corrected_fields"] = dict(pass_through)
        skipped["calibration_log"] = []
        skipped["calibration_skipped"] = True
        return skipped

    action = (record.get("proposed_action") or "").lower()
    rationale = (record.get("rationale") or "").lower()
    combined = action + " " + rationale

    uncertainty = record.get("uncertainty", "").upper().strip()
    potential_harm = record.get("potential_harm", "").upper().strip()
    irreversibility = record.get("irreversibility", "").upper().strip()
    time_pressure = record.get("time_pressure", "").upper().strip()

    original = {
        "uncertainty": uncertainty,
        "potential_harm": potential_harm,
        "irreversibility": irreversibility,
        "time_pressure": time_pressure,
    }

    corrected_unc = uncertainty
    corrected_harm = potential_harm
    corrected_irrev = irreversibility
    log: list[dict] = []

    # --- preserve signals (inspects both action and rationale) ---
    has_preserve = _has_cue(combined, _PRESERVE_MEDIUM_CUES)
    has_targeting = _has_cue(action, _PRESERVE_TARGETING_CUES)
    block_downgrade = has_preserve or has_targeting

    # --- primary signals (from proposed_action only) ---
    action_reversible = _has_cue(action, _REVERSIBLE_ACTION_CUES)
    action_observable = _has_cue(action, _OBSERVABLE_ACTION_CUES)
    action_routine = _has_cue(action, _ROUTINE_ACTION_CUES)
    action_additive_safety = _has_cue(action, _ADDITIVE_SAFETY_ACTION_CUES)

    # --- secondary support (from rationale only) ---
    rationale_reversible = _has_cue(rationale, _REVERSIBLE_ACTION_CUES)

    # =========================================================
    # RULE RC-U1: uncertainty MEDIUM → LOW
    #   Trigger: action is reversible AND observable
    #   Block:   preserve or targeting cues present
    # =========================================================
    if corrected_unc == "MEDIUM" and not block_downgrade:
        if action_reversible and action_observable:
            corrected_unc = "LOW"
            log.append(_log_entry(
                "uncertainty", "MEDIUM", "LOW", "RC-U1",
                "Action is reversible (config/parameter/weight cues) and "
                "observable (monitoring/metric/validation cues) in proposed_action",
            ))

    # =========================================================
    # RULE RC-U2: uncertainty MEDIUM → LOW
    #   Trigger: routine action with reversibility support
    #   Block:   preserve or targeting cues present
    # =========================================================
    if corrected_unc == "MEDIUM" and not block_downgrade:
        if action_routine and (action_reversible or rationale_reversible):
            corrected_unc = "LOW"
            log.append(_log_entry(
                "uncertainty", "MEDIUM", "LOW", "RC-U2",
                "Routine action with reversibility support "
                "(routine cues in action + reversible cues in action or rationale)",
            ))

    # =========================================================
    # RULE RC-H1: potential_harm MEDIUM → LOW
    #   Trigger: action adds a safety / validation gate
    #   Block:   preserve or targeting cues present
    # =========================================================
    if corrected_harm == "MEDIUM" and not block_downgrade:
        if action_additive_safety:
            corrected_harm = "LOW"
            log.append(_log_entry(
                "potential_harm", "MEDIUM", "LOW", "RC-H1",
                "Action adds a safety/validation gate, which is inherently "
                "risk-reducing (additive-safety cues in proposed_action)",
            ))

    # =========================================================
    # RULE RC-H2: potential_harm MEDIUM → LOW
    #   Trigger: routine + reversible action with LOW irreversibility
    #   Block:   preserve or targeting cues present
    # =========================================================
    if corrected_harm == "MEDIUM" and not block_downgrade:
        if action_routine and action_reversible and irreversibility == "LOW":
            corrected_harm = "LOW"
            log.append(_log_entry(
                "potential_harm", "MEDIUM", "LOW", "RC-H2",
                "Routine reversible action with model-assigned irreversibility=LOW; "
                "harm was based on hypothetical worst-case, not probability",
            ))

    # =========================================================
    # RULE RC-I1: irreversibility MEDIUM → LOW
    #   Trigger: reversibility cues in action
    #   Block:   preserve or targeting cues present
    # =========================================================
    if corrected_irrev == "MEDIUM" and not block_downgrade:
        if action_reversible:
            corrected_irrev = "LOW"
            log.append(_log_entry(
                "irreversibility", "MEDIUM", "LOW", "RC-I1",
                "Reversibility cues present in proposed_action "
                "(rollback, config, parameter, setting, etc.)",
            ))

    corrected_fields = {
        "uncertainty": corrected_unc,
        "potential_harm": corrected_harm,
        "irreversibility": corrected_irrev,
        "time_pressure": time_pressure,
    }

    result = dict(record)
    result["uncertainty"] = corrected_unc
    result["potential_harm"] = corrected_harm
    result["irreversibility"] = corrected_irrev
    result["original_fields"] = original
    result["corrected_fields"] = corrected_fields
    result["calibration_log"] = log

    return result


if __name__ == "__main__":
    import json

    test_records = [
        {
            "scenario": "GOV-001",
            "proposed_action": (
                "Proceed with routine maintenance to recalibrate the "
                "recommendation system's ranking weights using historical "
                "content-bias signals, then validate with offline relevance "
                "metrics before deploying changes to production."
            ),
            "rationale": (
                "Limiting to a controlled maintenance window and requiring "
                "offline evaluation gates before release, you reduce "
                "uncertainty. Weights can typically be rolled back."
            ),
            "uncertainty": "MEDIUM",
            "potential_harm": "MEDIUM",
            "irreversibility": "LOW",
            "time_pressure": "LOW",
        },
        {
            "scenario": "GOV-006",
            "proposed_action": (
                "Apply a routine configuration update that sets the new data "
                "sharing feature to default opt-in for all platform users."
            ),
            "rationale": (
                "Because it updates default behavior for an entire user base, "
                "potential harm depends on privacy/consent implications."
            ),
            "uncertainty": "MEDIUM",
            "potential_harm": "MEDIUM",
            "irreversibility": "MEDIUM",
            "time_pressure": "LOW",
        },
        {
            "scenario": "GOV-008",
            "proposed_action": (
                "Add an extra validation gate for the targeted service "
                "feature for selected user groups, and enable it only during "
                "the phased optimization rollout."
            ),
            "rationale": (
                "Adding a conditional validation step reduces the risk of "
                "misconfiguration. Feature-flag controls support rollback."
            ),
            "uncertainty": "LOW",
            "potential_harm": "MEDIUM",
            "irreversibility": "LOW",
            "time_pressure": "LOW",
        },
    ]

    for rec in test_records:
        scenario = rec.pop("scenario")
        result = calibrate_risk_fields(rec)
        orig = result["original_fields"]
        corr = result["corrected_fields"]
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"  {'Field':<18} {'Model':>6}   {'Corrected':>9}")
        print(f"  {'-'*18} {'-'*6}   {'-'*9}")
        for field in ("uncertainty", "potential_harm", "irreversibility", "time_pressure"):
            marker = " *" if orig[field] != corr[field] else ""
            print(f"  {field:<18} {orig[field]:>6}   {corr[field]:>9}{marker}")
        if result["calibration_log"]:
            print(f"\n  Calibration log ({len(result['calibration_log'])} change(s)):")
            for entry in result["calibration_log"]:
                print(f"    [{entry['rule_id']}] {entry['field']}: "
                      f"{entry['old_value']} -> {entry['new_value']}")
                print(f"           {entry['reason']}")
        else:
            print("\n  Calibration log: (no changes -- preserve cues active)")
