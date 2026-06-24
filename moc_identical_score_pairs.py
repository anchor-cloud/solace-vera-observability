"""List identical U/H/I/T pairs across runs (same model, same scenario)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HIST = Path("phase4_model_history")
MODELS = ("gpt", "gemini", "claude", "grok")
OUT = Path("moc_identical_score_pairs.csv")


def score_tuple(rec: dict) -> tuple[str, str, str, str]:
    return (
        str(rec.get("raw_uncertainty") or "").strip().upper(),
        str(rec.get("raw_potential_harm") or "").strip().upper(),
        str(rec.get("raw_irreversibility") or "").strip().upper(),
        str(rec.get("raw_time_pressure") or "").strip().upper(),
    )


def score_str(t: tuple[str, str, str, str]) -> str:
    abbr = {"LOW": "L", "MEDIUM": "M", "HIGH": "H"}
    return "/".join(abbr.get(x, "?") for x in t)


def run_sort_key(rid: str) -> tuple:
    if rid == "v1":
        return (1, rid)
    if rid == "v2":
        return (2, rid)
    return (3, rid)


def flatten_just(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\n", " ").strip()


def main() -> None:
    index: dict = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        path = HIST / f"{model}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                rid = rec.get("run_id")
                if sid and rid:
                    index[model][sid][rid] = rec

    rows: list[dict] = []
    per_model: Counter = Counter()
    pair_types: dict[str, Counter] = defaultdict(Counter)

    for model in MODELS:
        for sid in sorted(index[model].keys()):
            runs = index[model][sid]
            run_ids = sorted(runs.keys(), key=run_sort_key)
            if len(run_ids) < 2:
                continue
            for ra, rb in combinations(run_ids, 2):
                rec_a, rec_b = runs[ra], runs[rb]
                ta, tb = score_tuple(rec_a), score_tuple(rec_b)
                if ta != tb or "" in ta or "" in tb:
                    continue
                rows.append({
                    "model": model,
                    "scenario_id": sid,
                    "run_pair": f"{ra}|{rb}",
                    "run_a": ra,
                    "run_b": rb,
                    "U": ta[0],
                    "H": ta[1],
                    "I": ta[2],
                    "T": ta[3],
                    "risk_scores": score_str(ta),
                    "justification_1": flatten_just(rec_a.get("model_rationale")),
                    "justification_2": flatten_just(rec_b.get("model_rationale")),
                })
                per_model[model] += 1
                pair_types[model][f"{ra}|{rb}"] += 1

    rows.sort(key=lambda r: (r["model"], r["scenario_id"], r["run_a"], r["run_b"]))

    fieldnames = [
        "model",
        "scenario_id",
        "run_pair",
        "run_a",
        "run_b",
        "U",
        "H",
        "I",
        "T",
        "risk_scores",
        "justification_1",
        "justification_2",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT} ({len(rows)} rows)\n")
    print("=== SUMMARY STATISTICS ===\n")
    print(f"Total identical-score pairs: {len(rows)}\n")
    print("Per model:")
    for m in MODELS:
        print(f"  {m.upper():6} {per_model[m]}")
    print("\nPer model by run-pair:")
    for m in MODELS:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(pair_types[m].items()))
        print(f"  {m.upper():6} {parts}")

    print("\n=== CSV PREVIEW (first 10 rows; justifications truncated) ===\n")
    print(
        "model,scenario_id,run_pair,run_a,run_b,U,H,I,T,risk_scores,"
        "justification_1_preview,justification_2_preview"
    )
    for r in rows[:10]:
        j1 = r["justification_1"][:60] + ("..." if len(r["justification_1"]) > 60 else "")
        j2 = r["justification_2"][:60] + ("..." if len(r["justification_2"]) > 60 else "")
        print(
            f"{r['model']},{r['scenario_id']},{r['run_pair']},{r['run_a']},{r['run_b']},"
            f"{r['U']},{r['H']},{r['I']},{r['T']},{r['risk_scores']},"
            f"\"{j1}\",\"{j2}\""
        )


if __name__ == "__main__":
    main()
