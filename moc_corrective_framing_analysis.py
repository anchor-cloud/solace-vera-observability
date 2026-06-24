"""Corrective-action rationale framing (problem vs action vs both)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

HIST = Path("phase4_model_history")
PACK = Path("scenarios/moc_evidence_pack.csv")
MODELS = ("gpt", "gemini", "claude", "grok")
RUN_ID = "v1"

# Scenario prompts that are corrective per user taxonomy
CORRECTIVE_PROMPT = re.compile(
    r"\b("
    r"roll\s*back|rollback|"
    r"revoke|"
    r"disable|"
    r"patch|"
    r"remove|"
    r"quarantine|"
    r"rotate\s+the\s+master\s+api|"
    r"lock\s+all\s+user\s+accounts\s+matching\s+a\s+breach"
    r")\b",
    re.I,
)

# Exclude routine / forward / policy actions that match verbs but aren't incident correction
EXCLUDE_PROMPT = re.compile(
    r"rotate\s+daily\s+log\s+files|"
    r"recalibrate.*rollback\s+capability|"
    r"decommission.*remove\s+its\s+dns|"
    r"disable\s+backup\s+replication|"
    r"tune\s+the\s+load-balancer|"
    r"disable\s+direct\s+messaging\s+permanently.*comply",
    re.I,
)

PROBLEM_SIGNALS = re.compile(
    r"\b("
    r"overly\s+permissive|erroneous|regression|faulty|defective|"
    r"breach|compromised|unauthorized|exposure|vulnerabilit|"
    r"misconfigur|incident|violation|audit\s+finding|security\s+audit|"
    r"following\s+a\s+security|after\s+(?:the\s+)?(?:recent|latest)|"
    r"introduced\s+by|caused\s+by|due\s+to\s+(?:the\s+)?(?:recent|erroneous)|"
    r"negative\s+(?:engagement|impact)|billing\s+error|pricing\s+error|"
    r"problem|issue\s+being|corrects?|remediat|mitigat(?:es|ing)\s+(?:the\s+)?risk\s+of|"
    r"identified\s+by|detected|harmful\s+change|bad\s+change|"
    r"permissive\s+iam|dns\s+.*\s+fail|outage|misrouting|"
    r"repeated\s+policy\s+violations|investigation|breach[- ]indicator"
    r")\b",
    re.I,
)

ACTION_SIGNALS = re.compile(
    r"\b("
    r"roll(?:ing)?\s+back|rollback|revert(?:ing)?|"
    r"revok(?:e|ing)|"
    r"disabl(?:e|ing)|"
    r"patch(?:ing)?|"
    r"remov(?:e|ing)|"
    r"quarantin|"
    r"rotat(?:e|ing)\s+(?:the\s+)?(?:master\s+)?api|credential\s+rotation|"
    r"password\s+reset|"
    r"this\s+(?:corrective\s+)?action|"
    r"by\s+(?:rolling\s+back|revoking|disabling|removing)|"
    r"least[- ]privilege\s+review|verify\s+via|"
    r"cutover|restore|auto[- ]restore"
    r")\b",
    re.I,
)


def load_corrective_scenarios() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with PACK.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("scenario_id") or "").strip()
            prompt = (row.get("proposed_action") or "").strip()
            if not sid or not prompt or EXCLUDE_PROMPT.search(prompt):
                continue
            if CORRECTIVE_PROMPT.search(prompt):
                out[sid] = prompt
    return out


def load_v1_rationales() -> Dict[str, Dict[str, str]]:
    by_sid: Dict[str, Dict[str, str]] = defaultdict(dict)
    for model in MODELS:
        with (HIST / f"{model}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("run_id") == RUN_ID:
                    sid = rec.get("scenario_id")
                    if sid:
                        by_sid[sid][model] = (rec.get("model_rationale") or "").strip()
    return by_sid


def classify_framing(text: str) -> str:
    """Return A, B, or C."""
    if not text:
        return "MISSING"
    has_problem = bool(PROBLEM_SIGNALS.search(text))
    has_action = bool(ACTION_SIGNALS.search(text))
    if has_problem and has_action:
        return "C"
    if has_problem:
        return "A"
    if has_action:
        return "B"
    # Fallback: corrective scenarios usually describe the action in prompt echo
    if re.search(r"\b(risk|uncertainty|harm|irreversib|time\s+pressure)\b", text, re.I):
        return "B"
    return "A"


def main() -> None:
    scenarios = load_corrective_scenarios()
    rats = load_v1_rationales()
    sids = sorted(scenarios.keys())

    model_counts = {m: Counter() for m in MODELS}
    examples: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    print("=== Corrective-action scenarios (from moc_evidence_pack.csv) ===")
    print(f"Count: {len(sids)}\n")
    for sid in sids:
        print(f"  {sid}: {scenarios[sid][:85]}...")

    print(f"\n=== Framing classification (run {RUN_ID}) ===")
    print("  A = underlying problem emphasized")
    print("  B = corrective action emphasized")
    print("  C = both\n")

    total = 0
    for sid in sids:
        for model in MODELS:
            text = rats.get(sid, {}).get(model, "")
            cat = classify_framing(text)
            if cat == "MISSING":
                continue
            total += 1
            model_counts[model][cat] += 1
            if len(examples[(model, cat)]) < 1:
                examples[(model, cat)].append((sid, text[:220]))

    print(f"Classified rationales: {total} (missing scenarios use failed-run gaps)\n")
    print(f"{'Model':<8} {'n':>4}  {'A':>5} {'B':>5} {'C':>5}  |  {'A%':>6} {'B%':>6} {'C%':>6}")
    for m in MODELS:
        c = model_counts[m]
        n = sum(c.values()) or 1
        print(
            f"{m.upper():<8} {n:>4}  {c['A']:>5} {c['B']:>5} {c['C']:>5}  |  "
            f"{100*c['A']/n:>5.1f}% {100*c['B']/n:>5.1f}% {100*c['C']/n:>5.1f}%"
        )

    all_c = Counter()
    for m in MODELS:
        all_c += model_counts[m]
    n_all = sum(all_c.values()) or 1
    print(
        f"{'ALL':<8} {n_all:>4}  {all_c['A']:>5} {all_c['B']:>5} {all_c['C']:>5}  |  "
        f"{100*all_c['A']/n_all:>5.1f}% {100*all_c['B']/n_all:>5.1f}% {100*all_c['C']/n_all:>5.1f}%"
    )

    print("\n=== MOC-033 (IAM rollback) by model ===")
    for m in MODELS:
        t = rats.get("MOC-033", {}).get(m, "")
        print(f"  {m}: {classify_framing(t)} — {t[:160]}...")

    print("\n=== Sample excerpts per model × framing ===")
    for m in MODELS:
        for cat in ("C", "B", "A"):
            ex = examples.get((m, cat), [])
            if ex:
                sid, snip = ex[0]
                print(f"  {m.upper()} [{cat}] {sid}: {snip}...")

    Path("moc_corrective_framing_v1.json").write_text(
        json.dumps(
            {
                "scenario_ids": sids,
                "per_model": {m: dict(model_counts[m]) for m in MODELS},
                "total": dict(all_c),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
