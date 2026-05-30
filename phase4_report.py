"""phase4_report.py

Unified cross-run Phase 4 report for one or all tracked models.

Reads registered runs from ``phase4_model_history/<model>.jsonl`` and
prints a table of key metrics per run (posture mix, blocked rate, mean
duration, justification stability vs the previous run). Saves the same
table to ``phase4_reports/<model>_summary.txt``.

Examples:

    python phase4_report.py --model grok
    python phase4_report.py --model gpt --latest-only
    python phase4_report.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phase4_drift_per_model import (
    HISTORY_DIR_DEFAULT,
    KNOWN_MODELS,
    _SeverityEncoder,
    _run_meta,
    compute_severity,
    enrich_records_from_disk,
    existing_run_ids,
    filter_records_for_run,
    load_history,
    summarize_run,
)

REPORTS_DIR_DEFAULT = Path("phase4_reports")

TABLE_HEADERS: Tuple[str, ...] = (
    "Run ID",
    "Registered (UTC)",
    "N",
    "%PROCEED",
    "%PAUSE",
    "%ESCALATE",
    "%BLOCKED",
    "Mean dur (s)",
    "Justif. stab vs prev",
)


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _run_row_metrics(
    run_id: str,
    records: List[Dict[str, Any]],
    *,
    prev_records: Optional[List[Dict[str, Any]]] = None,
    encoder: Optional[_SeverityEncoder] = None,
) -> Tuple[str, ...]:
    summary = summarize_run(records)
    meta = _run_meta(records, run_id)
    total = summary["total_records"] or 1

    p1 = summary["phase1_posture_percentages"]
    blocked_pct = summary.get("blocked_pct", 0.0)
    timing = summary.get("timing") or {}
    mean_dur = timing.get("mean_duration_s", 0.0)

    stab = "-"
    if prev_records is not None and records and prev_records:
        enc = encoder or _SeverityEncoder()
        sev = compute_severity(prev_records, records, encoder=enc)
        if sev["paired_scenarios"]:
            stab = f"mean={sev['mean_severity']:.1f} major={sev['major_change_pct']:.0f}%"

    registered = (meta.get("registered_at_utc") or "")[:19]

    return (
        run_id,
        registered,
        str(summary["total_records"]),
        _fmt_pct(p1.get("PROCEED", 0.0)),
        _fmt_pct(p1.get("PAUSE", 0.0)),
        _fmt_pct(p1.get("ESCALATE", 0.0)),
        _fmt_pct(blocked_pct),
        f"{mean_dur:.2f}" if mean_dur else "n/a",
        stab,
    )


def build_model_report_table(
    model: str,
    *,
    history_dir: Path = HISTORY_DIR_DEFAULT,
    latest_only: bool = False,
) -> str:
    """Build a text table for all (or latest) runs of one model."""
    model = model.lower()
    run_ids = existing_run_ids(model, history_dir=history_dir)
    if not run_ids:
        return (
            f"No registered runs for model '{model}'. "
            f"Run a MOC test script (e.g. run_{model}_moc_test.py) first."
        )

    if latest_only:
        run_ids = [run_ids[-1]]

    history = load_history(model, history_dir=history_dir)
    encoder = _SeverityEncoder()

    rows: List[Tuple[str, ...]] = [TABLE_HEADERS]
    prev_records: Optional[List[Dict[str, Any]]] = None

    for rid in run_ids:
        records = filter_records_for_run(history, rid)
        records, _ = enrich_records_from_disk(records)
        if not records:
            continue
        rows.append(
            _run_row_metrics(
                rid,
                records,
                prev_records=prev_records if not latest_only else None,
                encoder=encoder,
            )
        )
        if not latest_only:
            prev_records = records

    col_widths = [
        max(len(row[i]) for row in rows) for i in range(len(TABLE_HEADERS))
    ]

    lines: List[str] = []
    lines.append(f"=== Phase 4 cross-run summary: {model.upper()} ===")
    lines.append(
        "  ".join(h.ljust(w) for h, w in zip(rows[0], col_widths))
    )
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        lines.append(
            "  ".join(cell.ljust(w) for cell, w in zip(row, col_widths))
        )
    lines.append("")
    lines.append(
        "Justif. stab vs prev: mean semantic severity (0=identical) and "
        "% of scenarios flagged as major justification change vs the "
        "immediately prior registered run."
    )
    return "\n".join(lines)


def write_model_summary_report(
    model: str,
    *,
    history_dir: Path = HISTORY_DIR_DEFAULT,
    out_dir: Path = REPORTS_DIR_DEFAULT,
    latest_only: bool = False,
) -> Path:
    """Write and return the path to ``<out_dir>/<model>_summary.txt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = build_model_report_table(
        model, history_dir=history_dir, latest_only=latest_only
    )
    path = out_dir / f"{model.lower()}_summary.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phase4_report",
        description=(
            "Combined Phase 4 metrics table across registered runs for "
            "one or all models."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        help=f"Model label ({', '.join(KNOWN_MODELS)}, etc.).",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Generate a report for every model with a history file.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIR_DEFAULT,
        help=f"Per-model history directory (default: {HISTORY_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPORTS_DIR_DEFAULT,
        help=f"Report output directory (default: {REPORTS_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Include only the most recent registered run in the table.",
    )
    return parser.parse_args(argv)


def _models_with_history(history_dir: Path) -> List[str]:
    if not history_dir.exists():
        return []
    return sorted(
        p.stem for p in history_dir.glob("*.jsonl") if p.stat().st_size > 0
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.all:
        models = _models_with_history(args.history_dir)
        if not models:
            print(
                "No model history files found. Run a MOC test script first.",
                file=sys.stderr,
            )
            return 2
    else:
        models = [args.model.lower()]

    any_ok = False
    for model in models:
        try:
            path = write_model_summary_report(
                model,
                history_dir=args.history_dir,
                out_dir=args.out_dir,
                latest_only=args.latest_only,
            )
            text = path.read_text(encoding="utf-8").rstrip("\n")
        except Exception as exc:
            print(f"ERROR [{model}]: {exc}", file=sys.stderr)
            continue

        print(text)
        print(f"[saved] {path}")
        print()
        any_ok = True

    return 0 if any_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
