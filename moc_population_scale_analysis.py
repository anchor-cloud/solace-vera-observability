"""Population-scale rationale analysis (MOC corpus, v1, no API)."""
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

# Scenarios whose CSV prompt implies large population / broad rollout
SCALE_PROMPT_PAT = re.compile(
    r"\b("
    r"all\s+users?|all\s+user\s+accounts?|all\s+platform\s+users?|"
    r"all\s+customers?|"
    r"mass[- ]|globally|global\s|worldwide|"
    r"entire\s+(department|platform|organization)|"
    r"platform[- ]wide|across\s+the\s+platform|"
    r"eu\s+region|"
    r"customer[- ]facing\s+cdn|"
    r"every\s+user"
    r")\b",
    re.I,
)

CATEGORIES = {
    "population_size": re.compile(
        r"\b("
        r"all\s+users?|all\s+eu|eu[- ]region|affects\s+all|"
        r"entire\s+\w+|large\s+(user\s+)?base|population|demographic|"
        r"platform[- ]wide|all\s+platform|globally|worldwide|"
        r"every\s+user|mass\s+|bulk\s+|widespread|"
        r"entire\s+department|all\s+customers?|"
        r"under\s+16|minors?\b|many\s+users|user\s+base|"
        r"thousands|millions|scale\s+of\s+impact|broad\s+"
        r")\b",
        re.I,
    ),
    "rollout_impact": re.compile(
        r"\b("
        r"rollout|deploy(ment)?|release|launch|"
        r"staged|phased|gradual|canary|feature[- ]flag|"
        r"region(al)?\s+rollout|production\s+rollout|"
        r"system[- ]wide|downstream|customer[- ]facing|"
        r"edge\s+nodes?|cdn|all\s+regions?"
        r")\b",
        re.I,
    ),
    "governance_consequences": re.compile(
        r"\b("
        r"governance|oversight|approval|stakeholder|"
        r"regulat(ory|ion)|compliance|legal\s+obligation|"
        r"gdpr|ccpa|eprivacy|audit|penalt|fine|"
        r"reputational|accountability|human\s+review|"
        r"escalat|executive|policy\s+change|"
        r"high[- ]impact|access\s+control"
        r")\b",
        re.I,
    ),
    "implementation_risk": re.compile(
        r"\b("
        r"misconfigur|implementation|deploy(ment)?\s+risk|"
        r"operational\s+risk|execution\s+risk|"
        r"rollback|reversib|irreversib|"
        r"validation|testing|pre[- ]deploy|"
        r"coordination|monitoring|safeguard|"
        r"feature[- ]flag|controlled\s+rollout|"
        r"unintended|side\s+effect|failure\s+mode"
        r")\b",
        re.I,
    ),
}


def load_large_group_scenarios() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with PACK.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("scenario_id", "").strip()
            prompt = row.get("proposed_action") or row.get("csv_prompt") or ""
            if sid and SCALE_PROMPT_PAT.search(prompt):
                out[sid] = prompt.strip()
    return out


def load_v1_rationales() -> Dict[str, Dict[str, str]]:
    by_scenario: Dict[str, Dict[str, str]] = defaultdict(dict)
    for model in MODELS:
        path = HIST / f"{model}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("run_id") != RUN_ID:
                    continue
                sid = rec.get("scenario_id")
                if sid:
                    by_scenario[sid][model] = (rec.get("model_rationale") or "").strip()
    return by_scenario


def hits(text: str, pat: re.Pattern) -> bool:
    return bool(pat.search(text))


def main() -> None:
    large = load_large_group_scenarios()
    rats = load_v1_rationales()
    scenario_ids = sorted(large.keys())

    # per model: count scenarios with each category mentioned
    model_cat_counts = {m: Counter() for m in MODELS}
    model_any = Counter()
    scenario_detail: List[dict] = []

    for sid in scenario_ids:
        row = {"scenario_id": sid, "prompt": large[sid][:100]}
        for model in MODELS:
            text = rats.get(sid, {}).get(model, "")
            if not text:
                row[model] = "MISSING"
                continue
            cats = {k: hits(text, p) for k, p in CATEGORIES.items()}
            row[model] = cats
            if any(cats.values()):
                model_any[model] += 1
            for k, v in cats.items():
                if v:
                    model_cat_counts[model][k] += 1
        scenario_detail.append(row)

    n = len(scenario_ids)
    print("=== Population-scale MOC scenarios (from moc_evidence_pack.csv) ===")
    print(f"Count: {n}")
    for sid in scenario_ids:
        print(f"  {sid}: {large[sid][:90]}...")
    print()

    print(f"=== Coverage: rationales mentioning >=1 category (run {RUN_ID}, n={n}) ===")
    for m in MODELS:
        print(f"  {m.upper():6} {model_any[m]:3}/{n} ({100*model_any[m]/n:.1f}%)")
    print()

    print("=== Per-category mention rate (scenarios with hit / n) ===")
    print(f"{'Category':<28} " + " ".join(f"{m:>8}" for m in MODELS))
    for cat in CATEGORIES:
        print(f"{cat:<28} ", end="")
        for m in MODELS:
            c = model_cat_counts[m][cat]
            print(f"{c:>5}/{n}  ", end="")
        print()

    print()
    print("=== Per-category % of large-group scenarios ===")
    for cat in CATEGORIES:
        parts = [f"{m.upper()} {100*model_cat_counts[m][cat]/n:.0f}%" for m in MODELS]
        print(f"  {cat}: {', '.join(parts)}")

    # All four categories at once
    print()
    print("=== Scenarios where ALL four categories appear (by model, v1) ===")
    for m in MODELS:
        all4 = 0
        for sid in scenario_ids:
            t = rats.get(sid, {}).get(m, "")
            if t and all(hits(t, p) for p in CATEGORIES.values()):
                all4 += 1
        print(f"  {m.upper():6} {all4}/{n}")

    # Examples: MOC-013 per model
    print()
    print("=== MOC-013 category hits by model ===")
    for m in MODELS:
        t = rats.get("MOC-013", {}).get(m, "")
        if not t:
            print(f"  {m}: missing")
            continue
        flags = {k: hits(t, p) for k, p in CATEGORIES.items()}
        print(f"  {m}: {flags}")
        print(f"       excerpt: {t[:200]}...")

    out_path = Path("moc_population_scale_v1.json")
    summary = {
        "run_id": RUN_ID,
        "large_group_scenario_count": n,
        "scenario_ids": scenario_ids,
        "per_model_any_category": dict(model_any),
        "per_model_per_category": {m: dict(model_cat_counts[m]) for m in MODELS},
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
