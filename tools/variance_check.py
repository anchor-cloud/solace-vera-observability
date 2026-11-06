
"""
variance_check.py - compares the two most recent runs per model.

Usage:  python variance_check.py path\to\pipeline_outputs

Read-only: reads audit JSON files, writes nothing.
Sorts each scenario's change between two runs into four levels:
  1 wording only        (same risk levels, constraints, and decision)
  2 risk change         (a self-reported risk level moved)
  3 constraint change   (different constraints violated)
  4 decision change     (Phase 1 posture, Phase 3 output, or final gate flipped)
Compares structured fields, not rationale text.
"""
import json, os, re, sys
from collections import Counter

BASE = sys.argv[1] if len(sys.argv) > 1 else "pipeline_outputs"

def load_all(d):
    out, prov = {}, Counter()
    for root, _, files in os.walk(d):
        for f in files:
            if not f.lower().endswith(".json"):
                continue
            try:
                r = json.load(open(os.path.join(root, f), encoding="utf-8-sig"))
                p = r["pipeline_result"]
                out[r["scenario_id"]] = dict(
                    risks=r["raw_risk_fields"],
                    decision=(p["phase1_posture"],
                              p["phase3_result"]["phase3_output"],
                              p["final_execution_gate"]["final_disposition"]),
                    viol=sorted(map(str, p["phase3_result"]["violated_constraints"])))
                prov[r.get("provider", "unknown")] += 1
            except Exception:
                continue
    return out, prov

ts_pat = re.compile(r"(\d{8}T\d{6}Z)")
runs, empty = {}, []
for name in sorted(os.listdir(BASE)):
    full = os.path.join(BASE, name)
    m = ts_pat.search(name)
    if not os.path.isdir(full) or not m or "moc" not in name:
        continue
    recs, prov = load_all(full)
    if not recs:
        empty.append(name)
        continue
    provider = prov.most_common(1)[0][0]
    runs.setdefault(provider, []).append((m.group(1), name, recs))

names = {1: "wording only", 2: "risk change", 3: "constraint change", 4: "decision change"}

for provider in sorted(runs):
    rl = sorted(runs[provider], key=lambda t: t[0])
    print("\n==", provider.upper(), "==")
    if len(rl) < 2:
        print("Only 1 non-empty run found:", rl[0][1])
        continue
    (ta, na, A), (tb, nb, B) = rl[-2], rl[-1]
    print("Run A:", na, "(", len(A), "scenarios )")
    print("Run B:", nb, "(", len(B), "scenarios )")
    levels = {1: [], 2: [], 3: [], 4: []}
    for sid in sorted(A):
        if sid not in B:
            continue
        x, y = A[sid], B[sid]
        if x["decision"] != y["decision"]: lv = 4
        elif x["viol"] != y["viol"]: lv = 3
        elif x["risks"] != y["risks"]: lv = 2
        else: lv = 1
        levels[lv].append(sid)
    n = sum(len(v) for v in levels.values())
    for lv in (1, 2, 3, 4):
        pct = 100 * len(levels[lv]) / n if n else 0
        extra = levels[lv] if lv > 1 else ""
        print(f"{names[lv]}: {len(levels[lv])}/{n} ({pct:.0f}%)", extra)

if empty:
    print("\nEmpty folders skipped:", len(empty))
    for e in empty:
        print("  ", e)
