"""phase4_justification_drift.py

Cross-run analysis answering one focused question:

    "When the raw risk scores (uncertainty, potential_harm, irreversibility,
    time_pressure) are IDENTICAL between two runs of the same model, how
    often does the model's free-text justification still change, and how
    semantically different are those rewordings?"

This is a stronger reasoning-stability test than the headline drift
report: when nothing about the scored evaluation moved, any meaningful
rationale change indicates that the model's reasoning trajectory is not
reproducible run-to-run.

Inputs
------
    phase4_model_history/<model>.jsonl  (produced by
    ``phase4_drift_per_model.py --register``)

Each line must include the extended fields written by the updated
``to_phase4_record``:
    raw_uncertainty, raw_potential_harm, raw_irreversibility,
    raw_time_pressure, model_rationale, run_id.

Output
------
    phase4_drift_reports/justification_stability.txt
    phase4_drift_reports/justification_stability.json

Severity score
--------------
Severity = (1 - cosine_similarity(embedding(rationale_v1),
                                  embedding(rationale_v2))) * 100

The script prefers ``sentence-transformers/all-MiniLM-L6-v2`` and falls
back to a stdlib TF-cosine if the package is unavailable. The backend
actually used is printed at the top of the report.

Usage
-----
    python phase4_justification_drift.py
    python phase4_justification_drift.py --compare-all
    python phase4_justification_drift.py --model gpt --run-a v1 --run-b v2

Run-pair selection
------------------
With no ``--run-a`` / ``--run-b`` flags, every registered run for each
model is read out of ``phase4_model_history/<model>.jsonl``, sorted by
its earliest scenario timestamp (with ``registered_at_utc`` as a
tiebreaker), and **consecutive** pairs are compared (v1->v2, v2->v3, ...).
``--compare-all`` switches to every-pair-against-every-other.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from phase4_per_model_analysis import RISK_DIMENSIONS, raw_risks_from_record
from phase4_drift_per_model import (
    DEFAULT_SEVERITY_MODEL,
    HISTORY_DIR_DEFAULT,
    DRIFT_REPORTS_DIR_DEFAULT,
    KNOWN_MODELS,
    _SeverityEncoder,
    filter_records_for_run,
    load_history,
)


# ---------------------------------------------------------------------------
# Run-pair selection
# ---------------------------------------------------------------------------
def runs_in_chronological_order(history: List[Dict[str, Any]]) -> List[str]:
    """Return distinct ``run_id`` values in chronological order.

    Sort key per run is ``(earliest_timestamp_utc, first_appearance_index)``.
    ``timestamp_utc`` (when the scenario was scored) is preferred; if it's
    absent we fall back to ``registered_at_utc`` (when the record was added
    to history). The append-order index breaks ties deterministically.
    """
    first_appearance: Dict[str, int] = {}
    earliest: Dict[str, str] = {}
    for idx, rec in enumerate(history):
        rid = rec.get("run_id")
        if not rid:
            continue
        if rid not in first_appearance:
            first_appearance[rid] = idx
        ts = rec.get("timestamp_utc") or rec.get("registered_at_utc")
        if not ts:
            continue
        prev = earliest.get(rid)
        if prev is None or ts < prev:
            earliest[rid] = ts
    return sorted(
        first_appearance.keys(),
        key=lambda rid: (earliest.get(rid, ""), first_appearance[rid]),
    )


def select_pairs(
    run_ids: List[str],
    *,
    mode: str,
) -> List[Tuple[str, str]]:
    """Return the list of (run_a, run_b) pairs to compare.

    ``mode`` is one of:
        - ``"consecutive"`` (default): v1->v2, v2->v3, ...
        - ``"all-pairs"``: every i<j pair, ordered (oldest, newest).
    """
    if mode not in {"consecutive", "all-pairs"}:
        raise ValueError(f"unknown pair-selection mode: {mode!r}")
    if len(run_ids) < 2:
        return []
    if mode == "all-pairs":
        return [
            (run_ids[i], run_ids[j])
            for i in range(len(run_ids))
            for j in range(i + 1, len(run_ids))
        ]
    return [(run_ids[i], run_ids[i + 1]) for i in range(len(run_ids) - 1)]


def comparison_label(run_a: str, run_b: str) -> str:
    """Render a compact arrow label for a run pair, e.g. ``v1->v2``."""
    return f"{run_a}->{run_b}"


def _matched_same_score_pairs(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
) -> Tuple[
    List[Tuple[str, Dict[str, Any], Dict[str, Any]]],  # same-score pairs
    List[str],                                          # diff-score scenario ids
    List[str],                                          # missing-score scenario ids
    List[str],                                          # unmatched (only in one run) ids
]:
    """Bucket scenarios by whether all four raw risk scores match between runs."""
    by_id_a = {r.get("scenario_id"): r for r in records_a if r.get("scenario_id")}
    by_id_b = {r.get("scenario_id"): r for r in records_b if r.get("scenario_id")}
    shared = sorted(set(by_id_a) & set(by_id_b))
    only_a = sorted(set(by_id_a) - set(by_id_b))
    only_b = sorted(set(by_id_b) - set(by_id_a))
    unmatched = only_a + only_b

    same: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    diff_score: List[str] = []
    missing_score: List[str] = []
    for sid in shared:
        ra = raw_risks_from_record(by_id_a[sid])
        rb = raw_risks_from_record(by_id_b[sid])
        if any(not ra[d] for d in RISK_DIMENSIONS) or any(
            not rb[d] for d in RISK_DIMENSIONS
        ):
            missing_score.append(sid)
            continue
        if all(ra[d] == rb[d] for d in RISK_DIMENSIONS):
            same.append((sid, by_id_a[sid], by_id_b[sid]))
        else:
            diff_score.append(sid)
    return same, diff_score, missing_score, unmatched


def _is_text_different(a: str, b: str) -> bool:
    """Conservative 'changed' criterion: strip-trimmed text not identical."""
    return (a or "").strip() != (b or "").strip()


def analyze_model(
    model: str,
    *,
    run_a_id: str,
    run_b_id: str,
    history_dir: Path,
    encoder: _SeverityEncoder,
) -> Dict[str, Any]:
    history = load_history(model, history_dir=history_dir)
    records_a = filter_records_for_run(history, run_a_id)
    records_b = filter_records_for_run(history, run_b_id)

    same_score, diff_score, missing_score, unmatched = _matched_same_score_pairs(
        records_a, records_b
    )
    n_a = len(records_a)
    n_b = len(records_b)

    # Of the same-score scenarios, which have textually different rationales?
    changed_pairs: List[Tuple[str, str, str]] = []
    unchanged_pairs: List[str] = []
    missing_rationale: List[str] = []
    for sid, rec_a, rec_b in same_score:
        text_a = (rec_a.get("model_rationale") or "").strip()
        text_b = (rec_b.get("model_rationale") or "").strip()
        if not text_a or not text_b:
            missing_rationale.append(sid)
            continue
        if _is_text_different(text_a, text_b):
            changed_pairs.append((sid, text_a, text_b))
        else:
            unchanged_pairs.append(sid)

    # Severity (semantic distance) for the changed pairs only.
    severities: List[Dict[str, Any]] = []
    if changed_pairs:
        sims = encoder.cosine_similarities(
            [(t_a, t_b) for _, t_a, t_b in changed_pairs]
        )
        for (sid, _, _), sim in zip(changed_pairs, sims):
            severity = round((1.0 - float(sim)) * 100.0, 2)
            severities.append({
                "scenario_id": sid,
                "cosine_similarity": round(float(sim), 4),
                "severity": severity,
            })

    severities.sort(key=lambda e: (-e["severity"], e["scenario_id"]))

    same_n = len(same_score)
    changed_n = len(changed_pairs)
    variance_rate = round((changed_n / same_n) * 100.0, 2) if same_n else 0.0
    avg_severity = (
        round(sum(s["severity"] for s in severities) / len(severities), 2)
        if severities
        else 0.0
    )
    max_severity = max((s["severity"] for s in severities), default=0.0)

    return {
        "model": model,
        "run_a_id": run_a_id,
        "run_b_id": run_b_id,
        "records_in_run_a": n_a,
        "records_in_run_b": n_b,
        "scenarios_only_in_one_run": unmatched,
        "scenarios_with_missing_raw_scores": missing_score,
        "scenarios_with_same_scores": same_n,
        "scenarios_with_different_scores": len(diff_score),
        "different_score_scenario_ids": diff_score,
        "justifications_changed": changed_n,
        "justifications_unchanged": len(unchanged_pairs),
        "scenarios_missing_rationale": missing_rationale,
        "variance_rate_pct": variance_rate,
        "avg_severity": avg_severity,
        "max_severity": max_severity,
        "severity_per_scenario": severities,
        "top_changes": severities[:3],
    }


def render_table(per_comparison: List[Dict[str, Any]], backend: str) -> str:
    headers = (
        "Model",
        "Comparison",
        "Same scores",
        "Justifications changed",
        "Variance rate",
        "Avg severity",
        "Max severity",
    )
    rows: List[Tuple[str, ...]] = [headers]
    for r in per_comparison:
        same_n = r["scenarios_with_same_scores"]
        paired = same_n + r["scenarios_with_different_scores"]
        rows.append((
            r["model"].upper(),
            comparison_label(r["run_a_id"], r["run_b_id"]),
            f"{same_n} of {paired}",
            f"{r['justifications_changed']} of {same_n}",
            f"{r['variance_rate_pct']}%",
            f"{r['avg_severity']}",
            f"{r['max_severity']}",
        ))

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    out: List[str] = []
    out.append("=== Phase 4 justification stability (same scores, different rationale) ===")
    out.append(f"Severity backend: {backend}")
    out.append("")
    out.append("  ".join(h.ljust(w) for h, w in zip(rows[0], widths)))
    out.append(sep)
    for row in rows[1:]:
        out.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    out.append("")
    out.append(
        "Note: 'Same scores' counts paired scenarios whose four raw risk "
        "scores were identical between the two runs."
    )
    return "\n".join(out)


def render_top_changes_block(per_comparison: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("")
    out.append("Top 3 most-dramatic justification changes per comparison")
    out.append(
        "(scenarios where the four raw risk scores were identical between "
        "the two runs but the rationale text changed the most)"
    )
    out.append("")
    for r in per_comparison:
        label = comparison_label(r["run_a_id"], r["run_b_id"])
        out.append(f"--- {r['model'].upper()} {label} ---")
        if not r["top_changes"]:
            out.append("  (no scenarios with same-score / changed rationale)")
        else:
            for entry in r["top_changes"]:
                out.append(
                    f"  {entry['scenario_id']:<10}  "
                    f"severity={entry['severity']:>6.2f}  "
                    f"cos_sim={entry['cosine_similarity']:+.4f}"
                )
        out.append("")
    return "\n".join(out)


def _resolve_pairs_for_model(
    model: str,
    *,
    history_dir: Path,
    explicit_pair: Optional[Tuple[str, str]],
    compare_all: bool,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Return ``(chronological_run_ids, pairs)`` for one model."""
    history = load_history(model, history_dir=history_dir)
    run_ids = runs_in_chronological_order(history)
    if explicit_pair is not None:
        # Honor the explicit pair even if one side isn't registered yet --
        # ``analyze_model`` will then just see zero records on that side.
        pairs = [explicit_pair]
    else:
        mode = "all-pairs" if compare_all else "consecutive"
        pairs = select_pairs(run_ids, mode=mode)
    return run_ids, pairs


def _latest_per_model(per_comparison: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick the chronologically-last comparison per model.

    Returned as a list in the order each model is first encountered. This
    is what downstream tools (``make_blog_figures.py``,
    ``phase4_concerning_audit.py``) consume via the ``per_model`` key, so
    they keep working without modification but now reflect the newest
    registered pair instead of being silently stuck on v1->v2.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for entry in per_comparison:
        m = entry["model"]
        if m not in latest:
            order.append(m)
        latest[m] = entry
    return [latest[m] for m in order]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase4_justification_drift",
        description=(
            "For each model, find scenarios whose four raw risk scores are "
            "identical between two runs, then measure how often (and how "
            "much) the model's free-text justification still changes. By "
            "default every consecutive pair of registered runs is compared."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Restrict to one or more models. Default: all known models "
            f"({', '.join(KNOWN_MODELS)})."
        ),
    )
    parser.add_argument(
        "--run-a",
        default=None,
        help=(
            "Run id for the 'A' side. If provided with --run-b, only that "
            "single pair is compared (overrides auto-detection)."
        ),
    )
    parser.add_argument(
        "--run-b",
        default=None,
        help="Run id for the 'B' side. Must be used together with --run-a.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help=(
            "Compare every registered run against every other (i<j). "
            "Default is to compare only consecutive runs."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIR_DEFAULT,
        help=f"Per-model history dir (default: {HISTORY_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DRIFT_REPORTS_DIR_DEFAULT,
        help=f"Where to write the stability report (default: {DRIFT_REPORTS_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--severity-model",
        default=DEFAULT_SEVERITY_MODEL,
        help=(
            "sentence-transformers model id (default: "
            f"{DEFAULT_SEVERITY_MODEL}). Falls back to a stdlib TF-cosine "
            "if the package isn't installed."
        ),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Validate --run-a/--run-b: both or neither.
    if (args.run_a is None) ^ (args.run_b is None):
        parser.error("--run-a and --run-b must be provided together")
    explicit_pair: Optional[Tuple[str, str]] = (
        (args.run_a, args.run_b) if args.run_a is not None else None
    )
    if explicit_pair is not None and args.compare_all:
        parser.error("--compare-all cannot be combined with --run-a/--run-b")

    if explicit_pair is not None:
        mode_label = "explicit"
    elif args.compare_all:
        mode_label = "all-pairs"
    else:
        mode_label = "consecutive"

    models = [m.lower() for m in args.model] or list(KNOWN_MODELS)
    encoder = _SeverityEncoder(args.severity_model)

    per_comparison: List[Dict[str, Any]] = []
    runs_per_model: Dict[str, List[str]] = {}
    for model in models:
        try:
            run_ids, pairs = _resolve_pairs_for_model(
                model,
                history_dir=args.history_dir,
                explicit_pair=explicit_pair,
                compare_all=args.compare_all,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"ERROR resolving runs for {model}: {exc}", file=sys.stderr)
            continue

        runs_per_model[model] = run_ids
        if not pairs:
            print(
                f"WARNING: model '{model}' has only "
                f"{len(run_ids)} registered run(s); skipping "
                f"(need at least 2 for a comparison).",
                file=sys.stderr,
            )
            continue

        for run_a_id, run_b_id in pairs:
            try:
                result = analyze_model(
                    model,
                    run_a_id=run_a_id,
                    run_b_id=run_b_id,
                    history_dir=args.history_dir,
                    encoder=encoder,
                )
            except Exception as exc:  # pragma: no cover - belt and braces
                print(
                    f"ERROR analyzing {model} {comparison_label(run_a_id, run_b_id)}: {exc}",
                    file=sys.stderr,
                )
                continue
            per_comparison.append(result)

    if not per_comparison:
        print("ERROR: no comparison results produced.", file=sys.stderr)
        return 2

    backend = encoder.backend
    table = render_table(per_comparison, backend)
    top_changes = render_top_changes_block(per_comparison)
    full_text = table + "\n" + top_changes

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.reports_dir / "justification_stability.txt"
    json_path = args.reports_dir / "justification_stability.json"
    txt_path.write_text(full_text, encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "severity_backend": backend,
        "severity_model_requested": args.severity_model,
        "mode": mode_label,
        "runs_per_model": runs_per_model,
        "per_comparison": per_comparison,
        # Back-compat: downstream tools key by model and overwrite, so we
        # expose only the latest comparison per model under this key.
        "per_model": _latest_per_model(per_comparison),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(full_text)
    print()
    print(f"[report-text] {txt_path}")
    print(f"[report-json] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
