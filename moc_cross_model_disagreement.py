"""Cross-model score disagreement analysis (MOC corpus, no API)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase4_drift_per_model import _tfidf_cosine

HIST = Path("phase4_model_history")
MODELS = ("gpt", "gemini", "claude", "grok")
RUN_ID = "v1"
LEVEL = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "": -1}
DIMS = ("uncertainty", "potential_harm", "irreversibility", "time_pressure")
DIM_LETTER = {"uncertainty": "U", "potential_harm": "H", "irreversibility": "I", "time_pressure": "T"}


def load_run(model: str, run_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    path = HIST / f"{model}.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                out[rec["scenario_id"]] = rec
    return out


def scores(rec: Dict[str, Any]) -> Dict[str, str]:
    return {
        d: str(rec.get(f"raw_{d}") or "").strip().upper()
        for d in DIMS
    }


def score_str(sc: Dict[str, str]) -> str:
    abbr = {"LOW": "L", "MEDIUM": "M", "HIGH": "H"}
    return "/".join(abbr.get(sc[d], "?") for d in DIMS)


def proposed_action(rec: Dict[str, Any]) -> str:
    rd = rec.get("run_directory")
    sid = rec.get("scenario_id")
    if not rd or not sid:
        return ""
    p = Path(rd) / f"{sid}.json"
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    raw = data.get("raw_model_record") or {}
    return (raw.get("proposed_action") or data.get("csv_prompt") or "").strip()


def action_markers(text: str) -> set[str]:
    t = (text or "").lower()
    markers = set()
    patterns = {
        "opt_in": r"\bopt[- ]?in\b",
        "opt_out": r"\bopt[- ]?out\b",
        "eu_scope": r"\beu\b|\beu[- ]region\b",
        "all_users": r"\ball\s+(users|platform)\b",
        "delete": r"\bdelete|delet|remov|purge\b",
        "archive": r"\barchiv|cold\s+storage\b",
        "reversible": r"\breversib|rollback|undo\b",
        "irreversible": r"\birreversib|permanent|cannot\s+be\s+undone\b",
        "privacy_enhancing": r"\bprivacy[- ]enhanc|beneficial\b",
        "high_harm": r"\bhigh\s+(potential\s+)?harm|significant\s+risk\b",
        "low_harm": r"\blow\s+(potential\s+)?harm|minimal\s+risk|low[- ]risk\b",
        "gdpr": r"\bgdpr\b",
        "consent": r"\bconsent\b",
        "default_on": r"\bdefault\s+(to\s+)?on|enabled\s+by\s+default\b",
        "default_off": r"\bdefault\s+(to\s+)?off|disabled\s+by\s+default\b",
    }
    for name, pat in patterns.items():
        if re.search(pat, t, re.I):
            markers.add(name)
    return markers


def is_significant(model_scores: Dict[str, Dict[str, str]]) -> Tuple[bool, int, int]:
    """Return (flag, max_spread, total_dim_mismatches_across_pairs)."""
    if len(model_scores) < 2:
        return False, 0, 0
    max_spread = 0
    for d in DIMS:
        vals = [LEVEL.get(model_scores[m][d], -1) for m in model_scores]
        vals = [v for v in vals if v >= 0]
        if vals:
            max_spread = max(max_spread, max(vals) - min(vals))
    mism = 0
    mods = list(model_scores.keys())
    for a, b in combinations(mods, 2):
        mism += sum(1 for d in DIMS if model_scores[a][d] != model_scores[b][d])
    # Significant: 2+ level gap on any dim, or heavy multi-dim disagreement
    harm_vals = [LEVEL.get(s["potential_harm"], -1) for s in model_scores.values()]
    harm_vals = [v for v in harm_vals if v >= 0]
    harm_spread = (max(harm_vals) - min(harm_vals)) if harm_vals else 0
    flag = max_spread >= 2 or mism >= 10 or harm_spread >= 2
    return flag, max_spread, mism


def mean_rationale_similarity(rationales: Dict[str, str]) -> float:
    texts = list(rationales.values())
    if len(texts) < 2:
        return 1.0
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(_tfidf_cosine(texts[i], texts[j]))
    return sum(sims) / len(sims)


def classify_disagreement(
    model_scores: Dict[str, Dict[str, str]],
    rationales: Dict[str, str],
    actions: Dict[str, str],
) -> str:
    """A, B, or C."""
    combined = " ".join(
        rationales.get(m, "") + " " + actions.get(m, "")
        for m in rationales
    )
    markers_by_model = {
        m: action_markers(rationales.get(m, "") + " " + actions.get(m, ""))
        for m in rationales
    }
    sim = mean_rationale_similarity(rationales)

    polar = []
    for m in rationales:
        t = rationales[m].lower()
        if re.search(
            r"\b(privacy[- ]enhanc|beneficial|low[- ]risk overall|"
            r"minimizes?\s+potential\s+harm)\b",
            t,
        ):
            polar.append("minimize")
        if re.search(
            r"\b(non[- ]compliant|violat|regulatory\s+penalt|"
            r"conflicts?\s+with|high\s+harm|significant\s+risk)\b",
            t,
        ):
            polar.append("elevate")
    polar_conflict = "minimize" in polar and "elevate" in polar

    key_markers = [
        "opt_in", "opt_out", "delete", "archive",
        "privacy_enhancing", "default_on", "default_off",
    ]
    divergent = 0
    for km in key_markers:
        has = [m for m in markers_by_model if km in markers_by_model[m]]
        lacks = [m for m in markers_by_model if km not in markers_by_model[m]]
        if has and lacks:
            divergent += 1

    action_texts = [actions.get(m, "") for m in MODELS if m in actions and actions.get(m)]
    action_sim = 1.0
    if len(action_texts) >= 2:
        sims = [
            _tfidf_cosine(action_texts[i], action_texts[j])
            for i in range(len(action_texts))
            for j in range(i + 1, len(action_texts))
        ]
        action_sim = sum(sims) / len(sims)

    harm_vals = [LEVEL.get(model_scores[m]["potential_harm"], -1) for m in model_scores]
    unc_vals = [LEVEL.get(model_scores[m]["uncertainty"], -1) for m in model_scores]
    harm_spread = max(harm_vals) - min(harm_vals) if harm_vals else 0
    unc_spread = max(unc_vals) - min(unc_vals) if unc_vals else 0

    shared_themes = sum(
        1
        for pat in (
            r"\bgdpr\b", r"\bconsent\b", r"\breversib", r"\bharm\b",
            r"\bcompliance\b", r"\bsecurity\b", r"\boperational\b",
        )
        if re.search(pat, combined, re.I)
    )

    # B: models describe different action framing or opposing narratives
    if polar_conflict:
        return "B"
    if divergent >= 3 and sim < 0.48:
        return "B"
    if action_sim < 0.55 and sim < 0.42:
        return "B"
    if sim < 0.34:
        return "B"

    # C: shared risk frame, different dimension emphasis / levels
    if shared_themes >= 2 and (harm_spread >= 1 or unc_spread >= 1) and sim >= 0.42:
        return "C"
    if shared_themes >= 3 and sim >= 0.38 and not polar_conflict:
        return "C"

    # A: largely aligned narrative, different scalar assessments
    if sim >= 0.52 and harm_spread <= 1 and unc_spread <= 1:
        return "A"

    if sim >= 0.45:
        return "C"
    return "A"


def main() -> None:
    by_model = {m: load_run(m, RUN_ID) for m in MODELS}
    scenarios = sorted(set().union(*(set(d.keys()) for d in by_model.values())))

    results: List[Dict[str, Any]] = []
    for sid in scenarios:
        present = {m: by_model[m][sid] for m in MODELS if sid in by_model[m]}
        if len(present) < 3:
            continue
        sc = {m: scores(present[m]) for m in present}
        sig, max_spread, mism = is_significant(sc)
        if not sig:
            continue
        rats = {m: (present[m].get("model_rationale") or "").strip() for m in present}
        acts = {m: proposed_action(present[m]) for m in present}
        cat = classify_disagreement(sc, rats, acts)
        results.append({
            "scenario_id": sid,
            "classification": cat,
            "max_spread": max_spread,
            "mismatches": mism,
            "mean_sim": round(mean_rationale_similarity(rats), 3),
            "scores": {m: score_str(sc[m]) for m in present},
            "rationales": rats,
            "actions": acts,
        })

    counts = Counter(r["classification"] for r in results)
    print("=== Cross-model disagreement (run v1, 3+ models present) ===")
    print(f"Scenarios analyzed: {len(scenarios)}")
    print(f"Significant disagreements: {len(results)}")
    print(f"  A (different risk assessment): {counts['A']}")
    print(f"  B (different action interpretation): {counts['B']}")
    print(f"  C (different weighting): {counts['C']}")
    print()

    for cat in ("B", "C", "A"):
        sub = [r for r in results if r["classification"] == cat]
        sub.sort(key=lambda x: (-x["max_spread"], -x["mismatches"]))
        print(f"--- Examples: {cat} ({len(sub)} total) ---")
        for r in sub[:3]:
            print(f"\n{r['scenario_id']} spread={r['max_spread']} mism={r['mismatches']} sim={r['mean_sim']}")
            for m in MODELS:
                if m in r["scores"]:
                    print(f"  {m:6} {r['scores'][m]}")
            for m in MODELS:
                if m in r["rationales"]:
                    print(f"  {m} rat: {r['rationales'][m][:140]}...")

    out = Path("moc_cross_model_disagreement_v1.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
