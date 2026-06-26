"""Framing (A/B/C) vs risk scores on corrective-action scenarios."""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from moc_corrective_framing_analysis import (
    MODELS,
    RUN_ID,
    classify_framing,
    load_corrective_scenarios,
)

HIST = Path("phase4_model_history")
LEVEL_NUM = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "": None}
DIMS = ("uncertainty", "potential_harm", "irreversibility", "time_pressure")


def load_records() -> List[Dict[str, Any]]:
    scenarios = set(load_corrective_scenarios().keys())
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        with (HIST / f"{model}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("run_id") != RUN_ID:
                    continue
                sid = rec.get("scenario_id")
                if sid not in scenarios:
                    continue
                rat = (rec.get("model_rationale") or "").strip()
                framing = classify_framing(rat)
                if framing == "MISSING":
                    continue
                scores = {}
                valid = True
                for d in DIMS:
                    v = str(rec.get(f"raw_{d}") or "").strip().upper()
                    n = LEVEL_NUM.get(v)
                    if n is None:
                        valid = False
                        break
                    scores[d] = v
                    scores[f"{d[0].upper() if d != 'potential_harm' else 'H'}"] = n
                if not valid:
                    continue
                u = LEVEL_NUM[scores["uncertainty"]]
                h = LEVEL_NUM[scores["potential_harm"]]
                i = LEVEL_NUM[scores["irreversibility"]]
                t = LEVEL_NUM[scores["time_pressure"]]
                rows.append({
                    "model": model,
                    "scenario_id": sid,
                    "framing": framing,
                    "U": u,
                    "H": h,
                    "I": i,
                    "T": t,
                    "composite": u + h + i + t,
                    "U_label": scores["uncertainty"],
                    "H_label": scores["potential_harm"],
                    "I_label": scores["irreversibility"],
                    "T_label": scores["time_pressure"],
                    "posture": str(rec.get("phase1_posture") or "").strip().upper(),
                    "rationale_excerpt": rat[:240],
                })
    return rows


def avg(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def summarize_group(rows: List[Dict[str, Any]], framing: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["framing"] == framing]
    if not sub:
        return {"n": 0}
    return {
        "n": len(sub),
        "avg_U": round(avg([r["U"] for r in sub]), 3),
        "avg_H": round(avg([r["H"] for r in sub]), 3),
        "avg_I": round(avg([r["I"] for r in sub]), 3),
        "avg_T": round(avg([r["T"] for r in sub]), 3),
        "avg_composite": round(avg([r["composite"] for r in sub]), 3),
        "posture": posture_pct(sub),
    }


def posture_pct(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    c = Counter(r["posture"] for r in rows if r.get("posture"))
    n = sum(c.values()) or 1
    return {
        k: round(100 * c.get(k, 0) / n, 1)
        for k in ("PROCEED", "PAUSE", "ESCALATE")
    }


def scenario_b_vs_c_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Within same scenario, compare B model vs C model composite scores."""
    by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["framing"] in ("B", "C"):
            by_sid[r["scenario_id"]].append(r)
    pairs = []
    for sid, group in by_sid.items():
        bs = [r for r in group if r["framing"] == "B"]
        cs = [r for r in group if r["framing"] == "C"]
        for b, c in [(b, c) for b in bs for c in cs]:
            pairs.append({
                "scenario_id": sid,
                "composite_diff": c["composite"] - b["composite"],  # positive => C higher
                "b": b,
                "c": c,
            })
    return pairs


def main() -> None:
    rows = load_records()
    out_path = Path("moc_framing_risk_dataset.json")
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("=== STEP 1: Dataset (corrective scenarios, v1) ===")
    print(f"Records: {len(rows)}")
    print(f"By framing: {dict(Counter(r['framing'] for r in rows))}\n")

    print("=== STEP 2: Framing vs risk (pooled) ===")
    for cat in ("A", "B", "C"):
        s = summarize_group(rows, cat)
        if s["n"] == 0:
            print(f"  {cat}: n=0")
            continue
        print(
            f"  {cat} (n={s['n']}): "
            f"U={s['avg_U']} H={s['avg_H']} I={s['avg_I']} T={s['avg_T']} "
            f"composite={s['avg_composite']}"
        )
    b = summarize_group(rows, "B")
    c = summarize_group(rows, "C")
    if b["n"] and c["n"]:
        print(
            f"\n  B vs C composite delta (B - C): "
            f"{round(b['avg_composite'] - c['avg_composite'], 3)} "
            f"(negative => B lower)"
        )

    print("\n=== STEP 3: Model breakdown (avg composite by framing) ===")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        parts = []
        for cat in ("A", "B", "C"):
            g = summarize_group(sub, cat)
            if g["n"]:
                parts.append(f"{cat}={g['avg_composite']} (n={g['n']})")
        b_g = summarize_group(sub, "B")
        c_g = summarize_group(sub, "C")
        flag = ""
        if b_g["n"] and c_g["n"]:
            if b_g["avg_composite"] < c_g["avg_composite"]:
                flag = " [B < C]"
            elif b_g["avg_composite"] > c_g["avg_composite"]:
                flag = " [B > C]"
        print(f"  {model.upper():6} {' | '.join(parts)}{flag}")

    print("\n=== STEP 4: Posture frequencies ===")
    for cat in ("A", "B", "C"):
        sub = [r for r in rows if r["framing"] == cat]
        p = posture_pct(sub)
        print(
            f"  {cat} (n={len(sub)}): "
            f"PROCEED={p.get('PROCEED', 0)}% "
            f"PAUSE={p.get('PAUSE', 0)}% "
            f"ESCALATE={p.get('ESCALATE', 0)}%"
        )

    print("\n=== STEP 5: Same-scenario B vs C pairs ===")
    pairs = scenario_b_vs_c_pairs(rows)
    print(f"Comparable B/C pairs (same scenario, different models): {len(pairs)}")
    if pairs:
        diffs = [p["composite_diff"] for p in pairs]
        print(
            f"  Mean(C composite - B composite): {round(avg(diffs), 3)} "
            f"(positive => C higher risk)"
        )
        print(f"  Pairs where B < C: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
        print(f"  Pairs where B > C: {sum(1 for d in diffs if d < 0)}/{len(diffs)}")

    # Examples
    pairs_sorted_high_c = sorted(pairs, key=lambda p: -p["composite_diff"])
    pairs_sorted_high_b = sorted(pairs, key=lambda p: p["composite_diff"])

    print("\n--- 5 examples: C higher risk than B (same scenario) ---")
    shown = 0
    for p in pairs_sorted_high_c:
        if p["composite_diff"] <= 0:
            continue
        b, c = p["b"], p["c"]
        print(
            f"\n  {p['scenario_id']}: C({c['model']}) composite={c['composite']} "
            f"{c['U_label']}/{c['H_label']}/{c['I_label']}/{c['T_label']} "
            f"vs B({b['model']})={b['composite']} "
            f"{b['U_label']}/{b['H_label']}/{b['I_label']}/{b['T_label']} "
            f"(diff=+{p['composite_diff']})"
        )
        print(f"    B excerpt: {b['rationale_excerpt']}...")
        print(f"    C excerpt: {c['rationale_excerpt']}...")
        shown += 1
        if shown >= 5:
            break

    print("\n--- 5 examples: B higher risk than C (same scenario) ---")
    shown = 0
    for p in pairs_sorted_high_b:
        if p["composite_diff"] >= 0:
            continue
        b, c = p["b"], p["c"]
        print(
            f"\n  {p['scenario_id']}: B({b['model']}) composite={b['composite']} "
            f"vs C({c['model']})={c['composite']} (diff={p['composite_diff']})"
        )
        print(f"    B excerpt: {b['rationale_excerpt']}...")
        print(f"    C excerpt: {c['rationale_excerpt']}...")
        shown += 1
        if shown >= 5:
            break

    # Effect size note
    print("\n=== FINAL: Quantitative summary ===")
    a_n = sum(1 for r in rows if r["framing"] == "A")
    b_n = sum(1 for r in rows if r["framing"] == "B")
    c_n = sum(1 for r in rows if r["framing"] == "C")
    print(f"Sample sizes: A={a_n}, B={b_n}, C={c_n} (total={len(rows)})")
    if b_n and c_n:
        b_s = summarize_group(rows, "B")
        c_s = summarize_group(rows, "C")
        bc = b_s["avg_composite"] - c_s["avg_composite"]
        pct = 100 * bc / c_s["avg_composite"] if c_s["avg_composite"] else 0
        print(f"Pooled B−C composite: {round(bc, 3)} ({round(pct, 1)}% relative)")


if __name__ == "__main__":
    main()
