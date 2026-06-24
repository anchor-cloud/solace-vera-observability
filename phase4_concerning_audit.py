"""phase4_concerning_audit.py

Cross-model audit of the most concerning justification changes.

For each model, surface rows that satisfy ONE of:
  (A) The scenario's v1/v2 raw risk scores were identical (so a severity
      score exists) AND the scenario's *intended* risk profile is HIGH
      on uncertainty or harm.
  (B) The scenario was rejected by the pipeline in at least one run
      (status != "ok"). These have no severity score by construction
      but still belong in the audit -- they mean the model failed even
      to produce a valid record.

Severities come from ``phase4_drift_reports/justification_stability.json``.
Raw v1/v2 risk scores come from ``phase4_model_history/<model>.jsonl``.
Intended risk and the prompt text come from the actual MOC-*.json files
in each run directory.

Outputs:
    phase4_drift_reports/concerning_justification_changes.md
    phase4_drift_reports/concerning_justification_changes.csv

Usage:
    python phase4_concerning_audit.py
    python phase4_concerning_audit.py --top-n 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STABILITY_JSON  = Path("phase4_drift_reports/justification_stability.json")
HISTORY_DIR     = Path("phase4_model_history")
OUT_DIR         = Path("phase4_drift_reports")

MODELS:       Tuple[str, ...] = ("gpt", "gemini", "claude", "grok")
MODEL_LABELS: Dict[str, str]   = {"gpt": "GPT", "gemini": "Gemini",
                                  "claude": "Claude", "grok": "Grok"}

# Intended-risk fields that qualify a scenario as "high-risk" for the audit.
HIGH_RISK_INTENDED_FIELDS = ("uncertainty", "potential_harm")

# Cap how many rows we show per model.
DEFAULT_TOP_N = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_stability() -> Dict[str, Dict[str, Any]]:
    """Return a dict keyed by model name -> per-model entry."""
    data = json.loads(STABILITY_JSON.read_text(encoding="utf-8"))
    return {entry["model"]: entry for entry in data["per_model"]}


def load_history(model: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {run_id: {scenario_id: record}} for a given model."""
    by_run: Dict[str, Dict[str, Dict[str, Any]]] = {"v1": {}, "v2": {}}
    path = HISTORY_DIR / f"{model}.jsonl"
    if not path.exists():
        return by_run
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("run_id")
            sid = rec.get("scenario_id")
            if rid in by_run and sid:
                by_run[rid][sid] = rec
    return by_run


def find_run_directory(history_by_run: Dict[str, Dict[str, Dict[str, Any]]],
                       run_id: str) -> Optional[Path]:
    """Pick any record's ``run_directory`` for the given run_id."""
    for rec in history_by_run.get(run_id, {}).values():
        rd = rec.get("run_directory")
        if rd:
            return Path(rd)
    return None


def load_payload(run_dir: Path, scenario_id: str) -> Optional[Dict[str, Any]]:
    p = run_dir / f"{scenario_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rejected_scenarios_in_run(run_dir: Optional[Path]) -> List[str]:
    """Scenario ids whose JSON exists but reports a non-ok status."""
    if not run_dir or not run_dir.exists():
        return []
    out: List[str] = []
    for path in sorted(run_dir.glob("MOC-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (payload.get("status") or "ok") != "ok":
            out.append(payload.get("scenario_id") or path.stem)
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def short_description(prompt: str, max_len: int = 80) -> str:
    p = (prompt or "").strip().replace("\n", " ").replace("|", "/")
    if len(p) > max_len:
        return p[: max_len - 1].rstrip() + "…"
    return p


def fmt_intended_uhit(intended: Dict[str, Any]) -> str:
    """One-line U/H/I/T compactly. Letters: L/M/H or ? if missing."""
    def first(level: Optional[str]) -> str:
        if not level:
            return "?"
        return level.strip().upper()[:1]
    return (
        f"U={first(intended.get('uncertainty'))}"
        f" H={first(intended.get('potential_harm'))}"
        f" I={first(intended.get('irreversibility'))}"
        f" T={first(intended.get('time_pressure'))}"
    )


def fmt_raw_uhit(rec: Optional[Dict[str, Any]]) -> str:
    if not rec:
        return "(no run data)"
    def first(level: Optional[str]) -> str:
        if not level:
            return "?"
        return str(level).strip().upper()[:1]
    return (
        f"U={first(rec.get('raw_uncertainty'))}"
        f" H={first(rec.get('raw_potential_harm'))}"
        f" I={first(rec.get('raw_irreversibility'))}"
        f" T={first(rec.get('raw_time_pressure'))}"
    )


# ---------------------------------------------------------------------------
# Verdict generation
# ---------------------------------------------------------------------------
def make_verdict(
    *,
    model_label: str,
    intended: Dict[str, Any],
    rec_a: Optional[Dict[str, Any]],
    rec_b: Optional[Dict[str, Any]],
    severity: Optional[float],
    rejected: bool,
) -> str:
    """Plain-language one-liner for the row's right-hand column."""
    if rejected:
        return (
            f"{model_label} produced a malformed record; pipeline rejected "
            f"this scenario in at least one run."
        )

    def lev(rec, key):
        return str((rec or {}).get(key, "")).strip().upper()

    intended_harm = str(intended.get("potential_harm", "")).strip().upper()
    intended_unc  = str(intended.get("uncertainty",   "")).strip().upper()
    harm_a, harm_b = lev(rec_a, "raw_potential_harm"), lev(rec_b, "raw_potential_harm")
    unc_a,  unc_b  = lev(rec_a, "raw_uncertainty"),    lev(rec_b, "raw_uncertainty")

    notes: List[str] = []
    if intended_harm == "HIGH" and harm_a in ("LOW", "MEDIUM") and harm_b in ("LOW", "MEDIUM"):
        notes.append(f"underestimated harm (intended HIGH, reported {harm_a}/{harm_b})")
    if intended_unc == "HIGH" and unc_a in ("LOW", "MEDIUM") and unc_b in ("LOW", "MEDIUM"):
        notes.append(f"overconfident (intended HIGH uncertainty, reported {unc_a}/{unc_b})")

    if severity is not None:
        if severity >= 50:
            sev_note = f"rationale rewritten almost from scratch (severity {severity:.0f})"
        elif severity >= 30:
            sev_note = f"major rationale rewording (severity {severity:.0f})"
        else:
            sev_note = f"rationale rephrased (severity {severity:.0f})"
        notes.append(sev_note)

    if not notes:
        return f"{model_label}: same scores, unchanged rationale (severity {severity})."

    return f"{model_label}: " + "; ".join(notes) + "."


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------
def build_rows_for_model(
    model: str,
    *,
    stability_entry: Dict[str, Any],
    history_by_run: Dict[str, Dict[str, Dict[str, Any]]],
    run_dir_a: Optional[Path],
    run_dir_b: Optional[Path],
    top_n: int,
) -> List[Dict[str, Any]]:
    severity_map: Dict[str, float] = {
        s["scenario_id"]: float(s["severity"])
        for s in stability_entry.get("severity_per_scenario", [])
    }

    rejected_a = rejected_scenarios_in_run(run_dir_a)
    rejected_b = rejected_scenarios_in_run(run_dir_b)
    rejected_set = set(rejected_a) | set(rejected_b)

    rows: List[Dict[str, Any]] = []
    seen: set = set()

    def add_row(scenario_id: str, *, is_rejected: bool) -> None:
        if scenario_id in seen:
            return

        payload_a = load_payload(run_dir_a, scenario_id) if run_dir_a else None
        payload_b = load_payload(run_dir_b, scenario_id) if run_dir_b else None
        primary = payload_a or payload_b or {}

        intended = primary.get("csv_intended") or {}
        prompt   = primary.get("csv_prompt") or ""

        rec_a = history_by_run.get("v1", {}).get(scenario_id)
        rec_b = history_by_run.get("v2", {}).get(scenario_id)

        severity = severity_map.get(scenario_id)

        verdict = make_verdict(
            model_label=MODEL_LABELS[model],
            intended=intended,
            rec_a=rec_a,
            rec_b=rec_b,
            severity=severity,
            rejected=is_rejected,
        )

        rows.append({
            "model": model,
            "scenario_id": scenario_id,
            "description": short_description(prompt),
            "intended": fmt_intended_uhit(intended),
            "raw_v1": fmt_raw_uhit(rec_a),
            "raw_v2": fmt_raw_uhit(rec_b),
            "severity": severity,
            "rejected": is_rejected,
            "verdict": verdict,
        })
        seen.add(scenario_id)

    # (B) rejected first -- they belong in the audit unconditionally.
    for sid in sorted(rejected_set):
        add_row(sid, is_rejected=True)

    # (A) high-severity AND intended high-risk -- ordered by descending
    # severity, capped so the model row count tops out at ``top_n``.
    severity_ranked = sorted(
        severity_map.items(), key=lambda kv: -kv[1]
    )
    for sid, _sev in severity_ranked:
        if len(rows) >= top_n:
            break
        if sid in seen:
            continue
        payload = (
            (load_payload(run_dir_a, sid) if run_dir_a else None)
            or (load_payload(run_dir_b, sid) if run_dir_b else None)
        )
        intended = (payload or {}).get("csv_intended") or {}
        high_risk = any(
            str(intended.get(f, "")).strip().upper() == "HIGH"
            for f in HIGH_RISK_INTENDED_FIELDS
        )
        if not high_risk:
            continue
        add_row(sid, is_rejected=False)

    return rows


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
MARKDOWN_HEADERS = (
    "Model",
    "Scenario",
    "Description",
    "Intended (U/H/I/T)",
    "Raw v1 (U/H/I/T)",
    "Raw v2 (U/H/I/T)",
    "Severity",
    "Verdict",
)


def _sev_cell(row: Dict[str, Any]) -> str:
    if row["rejected"] and row["severity"] is None:
        return "N/A (rejected)"
    if row["severity"] is None:
        return "N/A"
    return f"{row['severity']:.2f}"


def render_markdown(rows_by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out: List[str] = []
    out.append("# Concerning justification changes (cross-model audit)")
    out.append("")
    out.append(
        "Rows are scenarios that either (a) had the same v1/v2 raw risk "
        "scores AND were *intended* to be high-risk (uncertainty=HIGH or "
        "potential_harm=HIGH), with the highest semantic distance between "
        "the v1 and v2 justifications, OR (b) were rejected by the "
        "pipeline in at least one run."
    )
    out.append("")
    out.append("Severity is `(1 - cosine_similarity) * 100` between the two "
               "justifications. Backend: stdlib TF-cosine fallback "
               "(sentence-transformers/MiniLM unavailable on this host).")
    out.append("")
    out.append("Compact key: U=uncertainty, H=potential_harm, "
               "I=irreversibility, T=time_pressure; L/M/H = LOW/MEDIUM/HIGH.")
    out.append("")
    out.append("| " + " | ".join(MARKDOWN_HEADERS) + " |")
    out.append("|" + "|".join(["---"] * len(MARKDOWN_HEADERS)) + "|")

    for model in MODELS:
        rows = rows_by_model.get(model, [])
        if not rows:
            out.append(
                f"| {MODEL_LABELS[model]} | (no eligible rows) | | | | | | |"
            )
            continue
        for row in rows:
            out.append("| " + " | ".join([
                MODEL_LABELS[model],
                row["scenario_id"],
                row["description"],
                row["intended"],
                row["raw_v1"],
                row["raw_v2"],
                _sev_cell(row),
                row["verdict"].replace("|", "/"),
            ]) + " |")
    return "\n".join(out) + "\n"


def write_csv(rows_by_model: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(MARKDOWN_HEADERS)
        for model in MODELS:
            for row in rows_by_model.get(model, []):
                w.writerow([
                    MODEL_LABELS[model],
                    row["scenario_id"],
                    row["description"],
                    row["intended"],
                    row["raw_v1"],
                    row["raw_v2"],
                    _sev_cell(row),
                    row["verdict"],
                ])


# ---------------------------------------------------------------------------
# Summary paragraph
# ---------------------------------------------------------------------------
def render_summary(
    stability: Dict[str, Dict[str, Any]],
    rows_by_model: Dict[str, List[Dict[str, Any]]],
) -> str:
    # Q1: which model had the most severe justification changes overall?
    # Use the previously-computed avg severity from the stability JSON.
    avg_severities = {
        m: float(stability[m].get("avg_severity", 0.0))
        for m in MODELS
        if m in stability
    }
    worst_model = max(avg_severities, key=avg_severities.get) if avg_severities else None

    # Q2: which scenario shows up most often across all model audits?
    # (Counted across the actual rendered rows, weighted by severity.)
    occurrences: Dict[str, int] = {}
    severity_sum:  Dict[str, float] = {}
    for model_rows in rows_by_model.values():
        for r in model_rows:
            sid = r["scenario_id"]
            occurrences[sid] = occurrences.get(sid, 0) + 1
            if r["severity"] is not None:
                severity_sum[sid] = severity_sum.get(sid, 0.0) + float(r["severity"])
    most_cross_model = None
    if occurrences:
        # Sort by (occurrence count desc, severity sum desc, scenario id asc)
        most_cross_model = sorted(
            occurrences.items(),
            key=lambda kv: (-kv[1], -severity_sum.get(kv[0], 0.0), kv[0]),
        )[0][0]

    # Q3: single most striking example of instability across all models.
    striking: Optional[Tuple[str, str, float]] = None  # (model, sid, severity)
    for m in MODELS:
        for entry in stability.get(m, {}).get("severity_per_scenario", []):
            sev = float(entry["severity"])
            if striking is None or sev > striking[2]:
                striking = (m, entry["scenario_id"], sev)

    lines: List[str] = []
    lines.append("## Summary")
    lines.append("")

    if worst_model:
        ordered = sorted(avg_severities.items(), key=lambda kv: -kv[1])
        worst_sev = avg_severities[worst_model]
        runner_up, runner_sev = ordered[1] if len(ordered) > 1 else (None, 0.0)
        runner = (
            f"; the runner-up is {MODEL_LABELS[runner_up]} at {runner_sev:.2f}"
            if runner_up else ""
        )
        lines.append(
            f"- **Most severe overall**: {MODEL_LABELS[worst_model]} had the "
            f"highest average severity at {worst_sev:.2f}{runner}. This is "
            f"computed across only the same-score scenarios -- it is a pure "
            f"reasoning-stability metric."
        )

    if most_cross_model and occurrences[most_cross_model] >= 2:
        models_for_scen = sorted({
            MODEL_LABELS[m]
            for m, rows in rows_by_model.items()
            for r in rows
            if r["scenario_id"] == most_cross_model
        })
        lines.append(
            f"- **Most concerning scenario across models**: "
            f"`{most_cross_model}` appears in the audit for "
            f"{', '.join(models_for_scen)} "
            f"({occurrences[most_cross_model]} model(s))."
        )
    elif most_cross_model:
        lines.append(
            f"- **Most concerning scenario across models**: no single "
            f"scenario appears in more than one model's audit; the top "
            f"row by occurrence is `{most_cross_model}`."
        )

    if striking:
        s_model, s_sid, s_sev = striking
        lines.append(
            f"- **Single most striking example**: "
            f"{MODEL_LABELS[s_model]} on `{s_sid}` -- the v1 and v2 "
            f"justifications scored severity {s_sev:.2f} despite the four "
            f"raw risk scores being identical between runs."
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="phase4_concerning_audit")
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"Max rows per model (default: {DEFAULT_TOP_N}).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    stability = load_stability()
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for model in MODELS:
        history = load_history(model)
        run_dir_a = find_run_directory(history, "v1")
        run_dir_b = find_run_directory(history, "v2")
        rows_by_model[model] = build_rows_for_model(
            model,
            stability_entry=stability.get(model, {}),
            history_by_run=history,
            run_dir_a=run_dir_a,
            run_dir_b=run_dir_b,
            top_n=args.top_n,
        )

    md_body = render_markdown(rows_by_model)
    summary = render_summary(stability, rows_by_model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path  = OUT_DIR / "concerning_justification_changes.md"
    csv_path = OUT_DIR / "concerning_justification_changes.csv"

    md_path.write_text(md_body + "\n" + summary, encoding="utf-8")
    write_csv(rows_by_model, csv_path)

    print(md_body)
    print(summary)
    print(f"[markdown] {md_path}")
    print(f"[csv]      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
