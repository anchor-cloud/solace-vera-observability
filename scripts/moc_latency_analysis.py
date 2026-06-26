"""MOC corpus latency analysis (stored records only)."""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase4_drift_per_model import _tfidf_cosine

HIST = Path("phase4_model_history")
PACK = Path("scenarios/moc_evidence_pack.csv")
CLASSIFIED = Path("moc_identical_score_classified.csv")
OUT_CSV = Path("moc_latency_dataset.csv")
OUT_JSON = Path("moc_latency_analysis_summary.json")
MODELS = ("gpt", "gemini", "claude", "grok")
LEVEL_NUM = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "": None}

CORRECTIVE_PAT = re.compile(
    r"roll\s*back|revoke|disable|rotate\s+the\s+master|lock\s+all\s+user\s+accounts",
    re.I,
)
POPULATION_PAT = re.compile(
    r"all\s+users?|all\s+platform|globally|entire\s+department|mass[- ]|across\s+the\s+platform",
    re.I,
)


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_pack_meta() -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    with PACK.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("scenario_id") or "").strip()
            if sid:
                meta[sid] = row
    return meta


def scenario_categories(sid: str, row: Dict[str, str], prompt: str) -> List[str]:
    cats = []
    ct = (row.get("context_tag") or "").upper()
    ud = (row.get("use_domain") or "").upper()
    notes = (row.get("notes") or "").lower()
    p = prompt.lower()
    if ct == "ROUTINE" or ud == "ROUTINE_OPERATION":
        cats.append("routine_infrastructure")
    if "HUMAN_AFFECTING" in ct or "HUMAN_SERVICE" in ud:
        cats.append("human_affecting")
    if "ACCESS_CONTROL" in ud or "HIGH_IMPACT" in ct:
        cats.append("access_control")
    if "EXTERNAL_SYSTEM" in ct:
        cats.append("external_system")
    if "FINANCIAL" in ud:
        cats.append("financial")
    if re.search(r"gdpr|consent|privacy|legal|compliance", p + notes, re.I):
        cats.append("privacy_compliance")
    if POPULATION_PAT.search(prompt):
        cats.append("population_scale")
    if CORRECTIVE_PAT.search(prompt):
        cats.append("corrective_action")
    if re.search(r"security|breach|audit|iam|token|vpn", p + notes, re.I):
        cats.append("security_incident")
    if re.search(r"governance|oversight|high-impact", p + notes, re.I):
        cats.append("governance")
    if not cats:
        cats.append("other")
    return cats


def load_json_timing(run_dir: str, sid: str) -> Dict[str, Any]:
    p = Path(run_dir) / f"{sid}.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "started_at_utc": data.get("started_at_utc"),
        "finished_at_utc": data.get("finished_at_utc"),
        "duration_s": data.get("duration_s"),
        "token_count": data.get("token_count") or data.get("usage", {}).get("total_tokens")
        if isinstance(data.get("usage"), dict)
        else data.get("token_count"),
    }


def build_dataset() -> List[Dict[str, Any]]:
    pack = load_pack_meta()
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        path = HIST / f"{model}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                if not sid:
                    continue
                rd = str(rec.get("run_directory") or "")
                jt = load_json_timing(rd, sid) if rd else {}
                started = jt.get("started_at_utc") or ""
                finished = jt.get("finished_at_utc") or rec.get("timestamp_utc") or ""
                dur = jt.get("duration_s")
                if dur is None:
                    dur = rec.get("duration_s")
                try:
                    dur_f = float(dur) if dur is not None else None
                except (TypeError, ValueError):
                    dur_f = None
                if dur_f is None and started and finished:
                    t0, t1 = parse_iso(started), parse_iso(finished)
                    if t0 and t1:
                        dur_f = (t1 - t0).total_seconds()
                rat = rec.get("model_rationale") or ""
                pack_row = pack.get(sid, {})
                prompt = pack_row.get("proposed_action") or ""
                rows.append({
                    "model": model,
                    "scenario_id": sid,
                    "run_id": rec.get("run_id"),
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "latency_s": dur_f,
                    "U": rec.get("raw_uncertainty"),
                    "H": rec.get("raw_potential_harm"),
                    "I": rec.get("raw_irreversibility"),
                    "T": rec.get("raw_time_pressure"),
                    "posture": rec.get("phase1_posture"),
                    "justification_length": len(rat),
                    "token_count": jt.get("token_count"),
                    "categories": "|".join(scenario_categories(sid, pack_row, prompt)),
                })
    return rows


def load_classified_by_scenario() -> Dict[str, List[float]]:
    sev: Dict[str, List[float]] = defaultdict(list)
    if not CLASSIFIED.exists():
        return sev
    with CLASSIFIED.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                sev[row["scenario_id"]].append(float(row["severity"]))
            except (KeyError, ValueError):
                pass
    return {k: v for k, v in sev.items()}


def main() -> None:
    rows = build_dataset()
    with_lat = [r for r in rows if r["latency_s"] is not None]
    missing = len(rows) - len(with_lat)

    # Write CSV
    fields = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("=" * 60)
    print("STEP 1 — Latency dataset")
    print("=" * 60)
    print(f"Total records: {len(rows)}")
    print(f"With latency: {len(with_lat)}")
    print(f"Missing latency: {missing}")
    print(f"Token count present: {sum(1 for r in rows if r.get('token_count'))}")

    print("\nAverage / median latency by model (seconds):")
    for m in MODELS:
        d = [r["latency_s"] for r in with_lat if r["model"] == m]
        if d:
            print(
                f"  {m.upper():6} n={len(d):4}  mean={statistics.mean(d):.2f}  "
                f"median={statistics.median(d):.2f}  stdev={statistics.stdev(d):.2f}"
                if len(d) > 1
                else f"  {m.upper():6} n={len(d):4}  mean={statistics.mean(d):.2f}  median={statistics.median(d):.2f}"
            )

    slowest = sorted(with_lat, key=lambda r: -r["latency_s"])[:20]
    fastest = sorted(with_lat, key=lambda r: r["latency_s"])[:20]
    print("\nSlowest 20:")
    for r in slowest:
        print(
            f"  {r['latency_s']:.1f}s {r['model']} {r['run_id']} {r['scenario_id']} "
            f"len={r['justification_length']}"
        )
    print("\nFastest 20:")
    for r in fastest:
        print(
            f"  {r['latency_s']:.1f}s {r['model']} {r['run_id']} {r['scenario_id']}"
        )

    # Step 2 scenario latency
    print("\n" + "=" * 60)
    print("STEP 2 — Scenario latency")
    print("=" * 60)
    by_sid: Dict[str, List[float]] = defaultdict(list)
    sid_cats: Dict[str, str] = {}
    for r in with_lat:
        by_sid[r["scenario_id"]].append(r["latency_s"])
        sid_cats[r["scenario_id"]] = r["categories"]
    scen_avg = [(sid, statistics.mean(v), len(v)) for sid, v in by_sid.items()]
    scen_avg.sort(key=lambda x: -x[1])
    print("\nTop 20 slowest scenarios (mean latency, all models/runs):")
    slow_sids = []
    for sid, avg, n in scen_avg[:20]:
        slow_sids.append(sid)
        print(f"  {sid}: {avg:.2f}s (n={n}) [{sid_cats.get(sid,'')}]")
    print("\nTop 20 fastest scenarios:")
    for sid, avg, n in sorted(scen_avg, key=lambda x: x[1])[:20]:
        print(f"  {sid}: {avg:.2f}s (n={n})")

    cat_in_slow = Counter()
    for sid in slow_sids:
        for c in (sid_cats.get(sid) or "").split("|"):
            if c:
                cat_in_slow[c] += 1
    cat_all = Counter()
    for sid in by_sid:
        for c in (sid_cats.get(sid) or "").split("|"):
            if c:
                cat_all[c] += 1
    print("\nCategory tags in slowest-20 scenarios (count / 20):")
    for c, n in cat_in_slow.most_common():
        print(f"  {c}: {n}")
    print("(Reference: tag frequency across all scenarios in corpus)")

    # Step 3 disagreement - v1 cross-model per scenario
    print("\n" + "=" * 60)
    print("STEP 3 — Latency vs disagreement (v1 cross-model)")
    print("=" * 60)
    v1 = [r for r in with_lat if r["run_id"] in ("v1", "v2", "v3_20260528") or str(r["run_id"]).startswith("v")]
    v1_only = [r for r in with_lat if r["run_id"] == "v1"]
    per_sid_v1: Dict[str, List[Dict]] = defaultdict(list)
    for r in v1_only:
        per_sid_v1[r["scenario_id"]].append(r)

    scen_metrics = []
    for sid, grp in per_sid_v1.items():
        if len(grp) < 3:
            continue
        lats = [r["latency_s"] for r in grp]
        risks = []
        postures = []
        rats = []
        for r in grp:
            u, h, i, t = (
                LEVEL_NUM.get(str(r["U"]).upper()),
                LEVEL_NUM.get(str(r["H"]).upper()),
                LEVEL_NUM.get(str(r["I"]).upper()),
                LEVEL_NUM.get(str(r["T"]).upper()),
            )
            if None not in (u, h, i, t):
                risks.append(u + h + i + t)
            postures.append(str(r["posture"] or "").upper())
            rats.append(r.get("justification_length", 0))
        risk_var = statistics.pvariance(risks) if len(risks) > 1 else 0
        posture_var = len(set(postures)) / max(len(postures), 1)
        rat_var = statistics.pvariance(rats) if len(rats) > 1 else 0
        # pairwise justification dissimilarity proxy
        texts = [
            (HIST / f"{r['model']}.jsonl")  # placeholder - skip full text load
            for r in grp
        ]
        scen_metrics.append({
            "scenario_id": sid,
            "mean_latency": statistics.mean(lats),
            "risk_var": risk_var,
            "posture_var": posture_var,
            "rat_len_var": rat_var,
            "n_models": len(grp),
        })

    # reload rationales for sim
    for sid in list(per_sid_v1.keys()):
        texts = []
        for r in per_sid_v1[sid]:
            for rec_row in rows:
                if (
                    rec_row["scenario_id"] == sid
                    and rec_row["model"] == r["model"]
                    and rec_row["run_id"] == "v1"
                ):
                    # get rationale from history - already have length only
                    pass
        break

    # Simpler: load rationales from rows in build - extend dataset with rationale hash
    # Re-load v1 with rationale for sim
    per_sid_rats: Dict[str, Dict[str, str]] = defaultdict(dict)
    for model in MODELS:
        with (HIST / f"{model}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("run_id") == "v1" and rec.get("scenario_id"):
                    per_sid_rats[rec["scenario_id"]][model] = (
                        rec.get("model_rationale") or ""
                    )

    scen_metrics = []
    for sid, rats_by_m in per_sid_rats.items():
        if len(rats_by_m) < 3:
            continue
        grp = [r for r in v1_only if r["scenario_id"] == sid]
        lats = [r["latency_s"] for r in grp]
        risks = []
        postures = []
        for r in grp:
            u, h, i, t = (
                LEVEL_NUM.get(str(r["U"]).upper()),
                LEVEL_NUM.get(str(r["H"]).upper()),
                LEVEL_NUM.get(str(r["I"]).upper()),
                LEVEL_NUM.get(str(r["T"]).upper()),
            )
            if None not in (u, h, i, t):
                risks.append(u + h + i + t)
            postures.append(str(r["posture"] or "").upper())
        sims = []
        ms = list(rats_by_m.keys())
        texts = [rats_by_m[m] for m in ms]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(_tfidf_cosine(texts[i], texts[j]))
        just_dis = 1.0 - (statistics.mean(sims) if sims else 1.0)
        scen_metrics.append({
            "sid": sid,
            "mean_lat": statistics.mean(lats),
            "risk_var": statistics.pvariance(risks) if len(risks) > 1 else 0,
            "posture_var": len(set(postures)) / len(postures),
            "just_dis": just_dis,
            "disagree_score": (
                (statistics.pvariance(risks) if len(risks) > 1 else 0)
                + just_dis
                + len(set(postures)) / len(postures)
            ),
        })

    scen_metrics.sort(key=lambda x: x["disagree_score"], reverse=True)
    high_d = scen_metrics[:10]
    low_d = scen_metrics[-10:]
    high_lat = statistics.mean([x["mean_lat"] for x in high_d])
    low_lat = statistics.mean([x["mean_lat"] for x in low_d])
    print(f"Scenarios with v1 cross-model data: {len(scen_metrics)}")
    print(f"Mean latency — top-10 disagreement group: {high_lat:.2f}s")
    print(f"Mean latency — bottom-10 disagreement group: {low_lat:.2f}s")
    # correlation
    if len(scen_metrics) >= 5:
        xs = [x["disagree_score"] for x in scen_metrics]
        ys = [x["mean_lat"] for x in scen_metrics]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        r = num / den if den else 0
        print(f"Pearson r(disagreement_score, mean_latency): {r:.3f}")

    # Step 4
    print("\n" + "=" * 60)
    print("STEP 4 — Latency vs justification behavior")
    print("=" * 60)
    lens = [(r["latency_s"], r["justification_length"]) for r in with_lat]
    if len(lens) > 2:
        xs = [a for a, _ in lens]
        ys = [b for _, b in lens]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        r_len = num / den if den else 0
        print(f"A. Latency vs justification length: r={r_len:.3f}")
    sev_by_sc = load_classified_by_scenario()
    pairs_lat_sev = []
    for r in with_lat:
        sid = r["scenario_id"]
        if sid in sev_by_sc and sev_by_sc[sid]:
            pairs_lat_sev.append((r["latency_s"], statistics.mean(sev_by_sc[sid])))
    if len(pairs_lat_sev) > 5:
        xs = [a for a, _ in pairs_lat_sev]
        ys = [b for _, b in pairs_lat_sev]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        print(f"B. Latency vs identical-score justification severity: r={num/den if den else 0:.3f}")

    # identical score instability per scenario - count from classified
    instab = Counter()
    if CLASSIFIED.exists():
        with CLASSIFIED.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                instab[row["scenario_id"]] += 1
    lat_instab = []
    for r in with_lat:
        if r["scenario_id"] in instab:
            lat_instab.append((r["latency_s"], instab[r["scenario_id"]]))
    if lat_instab:
        xs = [a for a, _ in lat_instab]
        ys = [b for _, b in lat_instab]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        print(f"C. Latency vs identical-score pair count per scenario: r={num/den if den else 0:.3f}")

    # Step 5 model specific
    print("\n" + "=" * 60)
    print("STEP 5 — Model-specific")
    print("=" * 60)
    for m in MODELS:
        sub = [r for r in with_lat if r["model"] == m]
        d = [r["latency_s"] for r in sub]
        if not d:
            continue
        by_sc = defaultdict(list)
        for r in sub:
            by_sc[r["scenario_id"]].append(r["latency_s"])
        top_sc = sorted(
            ((sid, statistics.mean(v)) for sid, v in by_sc.items()),
            key=lambda x: -x[1],
        )[:10]
        print(f"\n{m.upper()}:")
        print(f"  mean={statistics.mean(d):.2f}s median={statistics.median(d):.2f}s stdev={statistics.stdev(d):.2f}s")
        print("  slowest scenarios:", ", ".join(f"{s}({v:.1f}s)" for s, v in top_sc[:5]))

    # Step 6 outliers
    print("\n" + "=" * 60)
    print("STEP 6 — Outliers")
    print("=" * 60)
    med_by_model = {m: statistics.median([r["latency_s"] for r in with_lat if r["model"] == m]) for m in MODELS}
    for mult in (2, 3, 5):
        outliers = []
        for r in with_lat:
            med = med_by_model[r["model"]]
            if med and r["latency_s"] >= mult * med:
                outliers.append((mult, r))
        print(f"\n>= {mult}x model median: {len(outliers)} records")
        for mult, r in sorted(outliers, key=lambda x: -x[1]["latency_s"])[:8]:
            print(
                f"  {r['latency_s']:.1f}s ({mult}x) {r['model']} {r['scenario_id']} "
                f"{r['U']}/{r['H']}/{r['I']}/{r['T']} {r['posture']} len={r['justification_length']}"
            )

    # Final
    print("\n" + "=" * 60)
    print("FINAL — Pattern strength")
    print("=" * 60)
    grok_mean = statistics.mean([r["latency_s"] for r in with_lat if r["model"] == "grok"])
    others = [r["latency_s"] for r in with_lat if r["model"] != "grok"]
    print(f"Grok mean latency >> others: {grok_mean:.1f}s vs {statistics.mean(others):.1f}s")
    print(f"Wrote {OUT_CSV}")

    summary = {
        "total_records": len(rows),
        "with_latency": len(with_lat),
        "missing_latency": missing,
        "model_stats": {
            m: {
                "mean": statistics.mean([r["latency_s"] for r in with_lat if r["model"] == m]),
                "median": statistics.median([r["latency_s"] for r in with_lat if r["model"] == m]),
            }
            for m in MODELS
            if any(r["model"] == m for r in with_lat)
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
