"""build_crossmodel_verdict_table.py

Emit a canonical, per-scenario cross-model verdict table as a native pipeline
artifact (JSON + CSV) instead of leaving cross-model disagreement derivable only
by ad-hoc post-hoc analysis.

For each scenario id it places all four models' Phase 3 verdicts side by side,
plus each model's execution decision and the EC-inference model actually used
(provenance), and flags unanimous scenarios.

Run-directory selection
-----------------------
By default the newest run directory per model is auto-selected from
``pipeline_outputs/`` by prefix:

    Claude -> claude_moc_*      GPT    -> moc_evidence_*
    Gemini -> gemini_moc_*      Grok   -> grok_moc_*

Override any of them explicitly with ``--claude/--gpt/--gemini/--grok <dir>``.

Outputs (into ``pipeline_outputs/crossmodel_verdicts/`` by default)::

    crossmodel_verdicts_<timestamp>.json
    crossmodel_verdicts_<timestamp>.csv
    crossmodel_verdicts_latest.json     (stable copy of the newest table)
    crossmodel_verdicts_latest.csv

Read-only over the scenario JSONs; safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_OUTPUTS = Path("pipeline_outputs")
DEFAULT_OUT_DIR = PIPELINE_OUTPUTS / "crossmodel_verdicts"

# Model -> run-directory glob prefix. Order defines column order.
MODEL_PREFIXES = {
    "Claude": "claude_moc_",
    "GPT": "moc_evidence_",
    "Gemini": "gemini_moc_",
    "Grok": "grok_moc_",
}
MODELS = list(MODEL_PREFIXES)

# Phase 3 output -> short verdict. Includes the infrastructure-failure state so
# an infra failure is never silently rendered as ambiguity.
SHORT = {
    "ETHICAL_PASS": "PASS",
    "ETHICAL_FAIL_CONSTRAINT_VIOLATION": "FAIL",
    "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED": "AMBIG",
    "ETHICAL_INFRASTRUCTURE_FAILURE": "INFRA",
}


def _newest_run_dir(prefix: str) -> Optional[Path]:
    """Return the most recently modified pipeline_outputs dir matching prefix."""
    if not PIPELINE_OUTPUTS.exists():
        return None
    candidates = [
        d for d in PIPELINE_OUTPUTS.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def resolve_run_dirs(overrides: Dict[str, Optional[str]]) -> Dict[str, Optional[Path]]:
    resolved: Dict[str, Optional[Path]] = {}
    for model, prefix in MODEL_PREFIXES.items():
        override = overrides.get(model)
        if override:
            resolved[model] = Path(override)
        else:
            resolved[model] = _newest_run_dir(prefix)
    return resolved


def _load_scenarios(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load every MOC-*.json in a run dir keyed by scenario id."""
    out: Dict[str, Dict[str, Any]] = {}
    for jf in sorted(run_dir.glob("MOC-*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = data.get("scenario_id") or jf.stem
        out[sid] = data
    return out


def _verdict_for(data: Dict[str, Any]) -> str:
    """Short verdict for one scenario record.

    Distinguishes model_call/pipeline failures (ERROR) and missing verdicts
    (MISSING) from the four real Phase 3 states.
    """
    status = data.get("status")
    if status not in (None, "ok"):
        return "ERROR"
    pr = data.get("pipeline_result") or {}
    output = pr.get("phase3_output")
    if not output:
        output = (pr.get("phase3_result") or {}).get("phase3_output")
    if not output:
        return "MISSING"
    return SHORT.get(str(output).upper(), str(output))


def _execution_allowed(data: Dict[str, Any]) -> Optional[bool]:
    pr = data.get("pipeline_result") or {}
    gate = pr.get("final_execution_gate") or {}
    val = gate.get("execution_allowed")
    return bool(val) if isinstance(val, bool) else None


def _ec_inference_model(data: Dict[str, Any]) -> str:
    pr = data.get("pipeline_result") or {}
    p1 = pr.get("phase1_record") or {}
    return p1.get("ec_inference_model", "") or ""


def _prompt_for(data: Dict[str, Any]) -> str:
    raw = data.get("raw_model_record") or {}
    return (raw.get("proposed_action") or data.get("csv_prompt") or "").strip()


def build_table(run_dirs: Dict[str, Optional[Path]]) -> Dict[str, Any]:
    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    used_dirs: Dict[str, Optional[str]] = {}
    for model, rd in run_dirs.items():
        if rd and rd.exists():
            loaded[model] = _load_scenarios(rd)
            used_dirs[model] = str(rd)
        else:
            loaded[model] = {}
            used_dirs[model] = None

    all_sids = sorted(
        {sid for recs in loaded.values() for sid in recs},
        key=lambda s: (len(s), s),
    )

    rows: List[Dict[str, Any]] = []
    for sid in all_sids:
        verdicts: Dict[str, str] = {}
        exec_allowed: Dict[str, Optional[bool]] = {}
        ec_models: Dict[str, str] = {}
        prompt = ""
        for model in MODELS:
            rec = loaded[model].get(sid)
            if rec is None:
                verdicts[model] = "MISSING"
                exec_allowed[model] = None
                ec_models[model] = ""
                continue
            verdicts[model] = _verdict_for(rec)
            exec_allowed[model] = _execution_allowed(rec)
            ec_models[model] = _ec_inference_model(rec)
            if not prompt:
                prompt = _prompt_for(rec)

        present = [verdicts[m] for m in MODELS if verdicts[m] not in ("MISSING", "ERROR")]
        distinct = sorted(set(present))
        unanimous = len(present) == len(MODELS) and len(distinct) == 1

        rows.append({
            "scenario_id": sid,
            "prompt": prompt,
            "verdicts": verdicts,
            "execution_allowed": exec_allowed,
            "ec_inference_model": ec_models,
            "unanimous": unanimous,
            "num_distinct_verdicts": len(distinct),
        })

    # Aggregate: per-model verdict counts + agreement summary.
    per_model_counts: Dict[str, Dict[str, int]] = {m: {} for m in MODELS}
    for row in rows:
        for m in MODELS:
            v = row["verdicts"][m]
            per_model_counts[m][v] = per_model_counts[m].get(v, 0) + 1

    unanimous_count = sum(1 for r in rows if r["unanimous"])

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "models": MODELS,
        "run_directories": used_dirs,
        "scenario_count": len(rows),
        "unanimous_count": unanimous_count,
        "per_model_verdict_counts": per_model_counts,
        "rows": rows,
    }


def write_outputs(table: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    json_path = out_dir / f"crossmodel_verdicts_{ts}.json"
    csv_path = out_dir / f"crossmodel_verdicts_{ts}.csv"
    latest_json = out_dir / "crossmodel_verdicts_latest.json"
    latest_csv = out_dir / "crossmodel_verdicts_latest.csv"

    payload = json.dumps(table, indent=2, ensure_ascii=False)
    json_path.write_text(payload, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")

    header = (
        ["scenario_id"]
        + [f"{m}_verdict" for m in MODELS]
        + [f"{m}_execution_allowed" for m in MODELS]
        + ["unanimous", "num_distinct_verdicts", "prompt"]
    )
    for path in (csv_path, latest_csv):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in table["rows"]:
                writer.writerow(
                    [row["scenario_id"]]
                    + [row["verdicts"][m] for m in MODELS]
                    + [row["execution_allowed"][m] for m in MODELS]
                    + [row["unanimous"], row["num_distinct_verdicts"], row["prompt"]]
                )

    return {
        "json": json_path,
        "csv": csv_path,
        "latest_json": latest_json,
        "latest_csv": latest_csv,
    }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_crossmodel_verdict_table",
        description=(
            "Build a canonical per-scenario cross-model verdict table (JSON+CSV) "
            "from the four model run directories."
        ),
    )
    parser.add_argument("--claude", default=None, help="Claude run dir (default: newest claude_moc_*).")
    parser.add_argument("--gpt", default=None, help="GPT run dir (default: newest moc_evidence_*).")
    parser.add_argument("--gemini", default=None, help="Gemini run dir (default: newest gemini_moc_*).")
    parser.add_argument("--grok", default=None, help="Grok run dir (default: newest grok_moc_*).")
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    overrides = {
        "Claude": args.claude,
        "GPT": args.gpt,
        "Gemini": args.gemini,
        "Grok": args.grok,
    }
    run_dirs = resolve_run_dirs(overrides)

    print("[build_crossmodel_verdict_table] Run directories:")
    for model in MODELS:
        rd = run_dirs[model]
        print(f"  {model:<7}: {rd if rd else '(none found)'}")

    if all(rd is None for rd in run_dirs.values()):
        print("ERROR: no run directories found for any model.", file=sys.stderr)
        return 1

    table = build_table(run_dirs)
    paths = write_outputs(table, Path(args.out_dir))

    print()
    print(f"Scenarios      : {table['scenario_count']}")
    print(f"Unanimous      : {table['unanimous_count']}")
    print("Verdict counts :")
    for model in MODELS:
        counts = table["per_model_verdict_counts"][model]
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {model:<7}: {pretty or '(no data)'}")
    print()
    print(f"JSON : {paths['json']}")
    print(f"CSV  : {paths['csv']}")
    print(f"Latest JSON : {paths['latest_json']}")
    print(f"Latest CSV  : {paths['latest_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
