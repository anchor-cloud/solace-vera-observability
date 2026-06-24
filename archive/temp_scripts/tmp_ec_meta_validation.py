"""Validation-only harness for recent Phase 3 ambiguity-layer edits.

Does NOT modify production code. Builds minimal records in-memory and
invokes phase3_gate.evaluate_phase3 directly.
"""

import copy
import json
import traceback

from phase3_gate import evaluate_phase3


BASE = {
    "scenario_id": "TEST",
    "proposed_action": "Routine operational health check",
    "uncertainty": "LOW",
    "potential_harm": "LOW",
    "irreversibility": "LOW",
    "time_pressure": "LOW",
    "posture": "PROCEED",
    "rationale": (
        "PROCEED baseline UNCERTAINTY=LOW POTENTIAL_HARM=LOW "
        "IRREVERSIBILITY=LOW TIME_PRESSURE=LOW USE_DOMAIN=ROUTINE_OPERATION"
    ),
    "context_tag": "ROUTINE",
    "use_domain": "ROUTINE_OPERATION",
}


CLEAN_ATOMIC = {
    "affected_groups": "GENERAL_POPULATION",
    "distribution_of_impact": "UNIFORM_OR_PROPORTIONAL",
    "benefit_distribution": "BENEFIT_GENERAL_OR_NEUTRAL",
    "population_vulnerability_flag": "NONE",
    "consent_status": "EXPLICIT_CONSENT",
    "consent_scope": "WITHIN_DECLARED_SCOPE",
    "participation_type": "VOLUNTARY_OPT_IN",
    "participation_information_quality": "CLEAR_AND_COMPREHENSIVE",
}


def make_record(overrides=None, drop=None, atomic=None):
    r = copy.deepcopy(BASE)
    if atomic:
        r.update(atomic)
    if overrides:
        r.update(overrides)
    if drop:
        for k in drop:
            r.pop(k, None)
    return r


def trace_status(result, constraint_id):
    for entry in result.get("trace", []):
        if entry["constraint_id"] == constraint_id:
            return entry["status"], entry["message"]
    return None, None


CASES = [
    # --- EC-04 blank-aware missing ---
    dict(
        name="A_EC04_atomic_absent",
        record=make_record(),
        expect_top="ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
        check=("EC-04", "AMBIGUITY"),
    ),
    dict(
        name="B_EC04_atomic_blank_strings",
        record=make_record(atomic={
            "affected_groups": "",
            "distribution_of_impact": "",
            "benefit_distribution": "",
            "population_vulnerability_flag": "",
        }),
        expect_top="ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
        check=("EC-04", "AMBIGUITY"),
    ),
    # --- EC-06 blank-aware missing ---
    dict(
        name="C_EC06_atomic_blank_strings",
        record=make_record(atomic={
            "population_vulnerability_flag": "",
            # potential_harm comes from BASE (LOW), so only vuln flag is blank
        }),
        expect_top="ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
        check=("EC-06", "AMBIGUITY"),
    ),
    # --- EC-09 blank-aware missing ---
    dict(
        name="D_EC09_atomic_blank_strings",
        record=make_record(atomic={
            "consent_status": "",
            "consent_scope": "",
            "participation_type": "",
        }),
        expect_top="ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
        check=("EC-09", "AMBIGUITY"),
    ),
    # --- EC-04 explicit FAIL path unchanged ---
    dict(
        name="E_EC04_explicit_FAIL_preserved",
        record=make_record(atomic={
            "affected_groups": "PROTECTED_OR_HISTORICALLY_DISADVANTAGED_GROUP",
            "distribution_of_impact": "DISPROPORTIONATE_IMPACT_ON_DISADVANTAGED",
            "benefit_distribution": "BENEFIT_TO_UNINVOLVED_OR_MORE_POWERFUL_GROUPS",
            "population_vulnerability_flag": "NONE",
            "consent_status": "EXPLICIT_CONSENT",
            "consent_scope": "WITHIN_DECLARED_SCOPE",
            "participation_type": "VOLUNTARY_OPT_IN",
        }),
        expect_top="ETHICAL_FAIL_CONSTRAINT_VIOLATION",
        check=("EC-04", "FAIL"),
    ),
    # --- EC-06 explicit FAIL path unchanged ---
    dict(
        name="F_EC06_explicit_FAIL_preserved",
        record=make_record(
            overrides={
                "potential_harm": "MEDIUM",
                "irreversibility": "LOW",
                "context_tag": "ROUTINE",
                "rationale": (
                    "PROCEED baseline UNCERTAINTY=LOW POTENTIAL_HARM=MEDIUM "
                    "IRREVERSIBILITY=LOW TIME_PRESSURE=LOW "
                    "USE_DOMAIN=ROUTINE_OPERATION"
                ),
            },
            atomic={
                "affected_groups": "GENERAL_POPULATION",
                "distribution_of_impact": "UNIFORM_OR_PROPORTIONAL",
                "benefit_distribution": "BENEFIT_GENERAL_OR_NEUTRAL",
                "population_vulnerability_flag": "CLEARLY_VULNERABLE",
                "consent_status": "EXPLICIT_CONSENT",
                "consent_scope": "WITHIN_DECLARED_SCOPE",
                "participation_type": "VOLUNTARY_OPT_IN",
            },
        ),
        expect_top="ETHICAL_FAIL_CONSTRAINT_VIOLATION",
        check=("EC-06", "FAIL"),
    ),
    # --- EC-09 explicit FAIL path unchanged ---
    dict(
        name="G_EC09_explicit_FAIL_preserved",
        record=make_record(atomic={
            "affected_groups": "GENERAL_POPULATION",
            "distribution_of_impact": "UNIFORM_OR_PROPORTIONAL",
            "benefit_distribution": "BENEFIT_GENERAL_OR_NEUTRAL",
            "population_vulnerability_flag": "NONE",
            "consent_status": "NO_CONSENT",
            "consent_scope": "WITHIN_DECLARED_SCOPE",
            "participation_type": "VOLUNTARY_OPT_IN",
        }),
        expect_top="ETHICAL_FAIL_CONSTRAINT_VIOLATION",
        check=("EC-09", "FAIL"),
    ),
    # --- EC-08 KeyError guard: context_tag absent ---
    dict(
        name="H_EC08_context_tag_absent",
        record=make_record(drop=["context_tag"], atomic=CLEAN_ATOMIC),
        expect_top="ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
        check=("EC-08", "AMBIGUITY"),
    ),
    # --- EC-11 KeyError guard: time_pressure blank ---
    # Must use blank (not absent) because EC-05 also reads time_pressure
    # and its behavior was not in scope for this patch.
    dict(
        name="I_EC11_time_pressure_blank",
        record=make_record(
            overrides={"time_pressure": ""},
            atomic=CLEAN_ATOMIC,
        ),
        expect_top=None,  # top-level depends on other evaluators; check trace only
        check=("EC-11", "AMBIGUITY"),
    ),
    # --- Clean safe case must still be ETHICAL_PASS ---
    dict(
        name="J_clean_safe_all_low_ETHICAL_PASS",
        record=make_record(atomic=CLEAN_ATOMIC),
        expect_top="ETHICAL_PASS",
        check=None,
    ),
]


def run():
    passes = 0
    fails = 0
    for case in CASES:
        name = case["name"]
        record = case["record"]
        expect_top = case["expect_top"]
        check = case["check"]
        try:
            result = evaluate_phase3(record)
        except Exception as e:
            fails += 1
            print(f"[FAIL] {name}: EXCEPTION {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        top = result["phase3_output"]
        ok = True
        notes = []
        if expect_top is not None and top != expect_top:
            ok = False
            notes.append(f"top expected={expect_top} got={top}")
        if check is not None:
            cid, cstatus = check
            got_status, got_msg = trace_status(result, cid)
            if got_status != cstatus:
                ok = False
                notes.append(f"{cid} expected={cstatus} got={got_status} msg={got_msg!r}")
            else:
                notes.append(f"{cid}={got_status}")
        if ok:
            passes += 1
            print(f"[PASS] {name}: top={top} | {'; '.join(notes)}")
        else:
            fails += 1
            print(f"[FAIL] {name}: top={top} | {'; '.join(notes)}")

    print()
    print(f"SUMMARY: {passes} passed, {fails} failed, {len(CASES)} total")
    return fails


import csv
from collections import Counter
from pathlib import Path


SCENARIO_PACKS = [
    "scenarios/phase3_core_smoke.csv",
    "scenarios/phase3_tests_v2.csv",
    "scenarios/adversarial_pack_core.csv",
    "scenarios/break_pack_v1.csv",
    "scenarios/hostile_pack_v1.csv",
]


def sweep_pack(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    top_counts = Counter()
    trace_status_counts = {"EC-04": Counter(), "EC-06": Counter(), "EC-09": Counter(),
                           "EC-08": Counter(), "EC-11": Counter()}
    exceptions = []
    expected_vs_actual = Counter()

    for row in rows:
        record = {
            "scenario_id": row.get("scenario_id", ""),
            "proposed_action": row.get("proposed_action", ""),
            "uncertainty": row.get("uncertainty", ""),
            "potential_harm": row.get("potential_harm", ""),
            "irreversibility": row.get("irreversibility", ""),
            "time_pressure": row.get("time_pressure", ""),
            "posture": row.get("expected_phase1") or row.get("posture") or "PROCEED",
            "context_tag": row.get("context_tag", ""),
            "use_domain": row.get("use_domain", ""),
        }
        record["rationale"] = (
            f"{record['posture']} synthesized UNCERTAINTY={record['uncertainty']} "
            f"POTENTIAL_HARM={record['potential_harm']} IRREVERSIBILITY={record['irreversibility']} "
            f"TIME_PRESSURE={record['time_pressure']} USE_DOMAIN={record['use_domain']}"
        )
        try:
            result = evaluate_phase3(record)
        except Exception as e:
            exceptions.append((row.get("scenario_id", ""), type(e).__name__, str(e)))
            continue

        top_counts[result["phase3_output"]] += 1
        for cid in trace_status_counts:
            s, _ = trace_status(result, cid)
            if s is not None:
                trace_status_counts[cid][s] += 1

        expected = row.get("expected_phase3", "").strip()
        if expected:
            actual = result["phase3_output"]
            if expected == actual:
                expected_vs_actual[("match", expected)] += 1
            else:
                expected_vs_actual[("diff", expected, actual)] += 1

    return top_counts, trace_status_counts, exceptions, expected_vs_actual, len(rows)


def sweep_all():
    print()
    print("=== Scenario pack sweep (Phase 3 only, records built from CSV core fields) ===")
    for pack in SCENARIO_PACKS:
        p = Path(pack)
        if not p.exists():
            print(f"(skip) {pack} not found")
            continue
        top, traces, exc, diff, n = sweep_pack(p)
        print(f"\n-- {pack} (n={n}) --")
        print(f"  top-level: {dict(top)}")
        for cid, c in traces.items():
            print(f"  {cid}: {dict(c)}")
        if exc:
            print(f"  EXCEPTIONS: {exc}")
        else:
            print("  exceptions: none")
        match = sum(v for k, v in diff.items() if k[0] == "match")
        mismatch = sum(v for k, v in diff.items() if k[0] == "diff")
        print(f"  expected_phase3: match={match} diff={mismatch}")
        diffs = {k[1:]: v for k, v in diff.items() if k[0] == "diff"}
        if diffs:
            for (exp, act), v in sorted(diffs.items()):
                print(f"    expected={exp} -> actual={act}: {v}")


if __name__ == "__main__":
    rc = run()
    sweep_all()
    raise SystemExit(rc)
