"""phase4_auto.py

Automatic Phase 4 tracking after MOC evidence test runs.

Called by ``run_moc_evidence.py`` and the provider-specific runners at the
end of a successful run to:

  1. Register scenario records in ``phase4_model_history/<model>.jsonl``
  2. Compare the last two runs for drift (when at least two exist)
  3. Refresh the combined cross-run report via ``phase4_report.py``

Use ``--no-register`` on any test runner to skip this post-run hook.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from phase4_drift_per_model import (
    DEFAULT_THRESHOLD_PP,
    DRIFT_REPORTS_DIR_DEFAULT,
    HISTORY_DIR_DEFAULT,
    _SeverityEncoder,
    _resolve_drift_run_ids,
    _run_meta,
    compute_drift,
    enrich_records_from_disk,
    existing_run_ids,
    filter_records_for_run,
    load_history,
    make_auto_run_id,
    register_run,
    summarize_run,
    write_drift_report,
)


def post_run_phase4_tracking(
    *,
    model: str,
    run_dir: Path,
    enabled: bool = True,
    run_id: Optional[str] = None,
    history_dir: Path = HISTORY_DIR_DEFAULT,
    reports_dir: Path = DRIFT_REPORTS_DIR_DEFAULT,
    drift_threshold_pp: float = DEFAULT_THRESHOLD_PP,
) -> Dict[str, Any]:
    """Register a completed run and refresh drift + summary reports.

    Returns a receipt dict with keys ``register``, ``drift``, ``report``.
    Non-fatal failures are captured under ``errors`` so the test runner can
    still exit 0 after a successful API run.
    """
    result: Dict[str, Any] = {
        "model": model.lower(),
        "run_id": None,
        "register": None,
        "drift": None,
        "report": None,
        "errors": [],
    }

    if not enabled:
        print("[phase4] Auto-registration disabled (--no-register).")
        return result

    model = model.lower()
    run_id = run_id or make_auto_run_id()
    result["run_id"] = run_id

    print()
    print("[phase4] Registering run for cross-run tracking...")
    try:
        receipt = register_run(
            model=model,
            run_id=run_id,
            run_dir=run_dir,
            history_dir=history_dir,
        )
        result["register"] = receipt
        print(
            f"[phase4] Registered model={receipt['model']} run_id={receipt['run_id']} "
            f"scenarios={receipt['scenario_records_appended']}"
        )
        print(f"[phase4] History file: {receipt['history_file']}")
    except Exception as exc:
        msg = f"registration failed: {exc}"
        result["errors"].append(msg)
        print(f"[phase4] WARNING: {msg}", file=sys.stderr)
        return result

    run_ids = existing_run_ids(model, history_dir=history_dir)
    if len(run_ids) >= 2:
        print("[phase4] Comparing last two registered runs for drift...")
        try:
            drift_paths = _auto_compare_latest_two(
                model=model,
                history_dir=history_dir,
                reports_dir=reports_dir,
                threshold_pp=drift_threshold_pp,
            )
            result["drift"] = drift_paths
            print(f"[phase4] Drift report: {drift_paths[0]}")
        except Exception as exc:
            msg = f"drift comparison failed: {exc}"
            result["errors"].append(msg)
            print(f"[phase4] WARNING: {msg}", file=sys.stderr)
    else:
        print(
            "[phase4] Only one run registered so far; drift comparison "
            "will run after the next successful test run."
        )

    print("[phase4] Updating combined per-model summary...")
    try:
        from phase4_per_model_analysis import regenerate_combined_summary

        json_path, txt_path = regenerate_combined_summary(
            model,
            history_dir=history_dir,
        )
        result["report"] = {
            "json": str(json_path),
            "txt": str(txt_path),
        }
        print(f"[phase4] Combined summary: {txt_path}")
    except Exception as exc:
        msg = f"summary report failed: {exc}"
        result["errors"].append(msg)
        print(f"[phase4] WARNING: {msg}", file=sys.stderr)

    return result


def _auto_compare_latest_two(
    *,
    model: str,
    history_dir: Path,
    reports_dir: Path,
    threshold_pp: float,
) -> tuple[Path, Path]:
    """Compare the two most recent runs and write drift artifacts."""
    run_a_id, run_b_id = _resolve_drift_run_ids(
        model, [], history_dir=history_dir
    )
    history = load_history(model, history_dir=history_dir)
    records_a = filter_records_for_run(history, run_a_id)
    records_b = filter_records_for_run(history, run_b_id)
    records_a, _ = enrich_records_from_disk(records_a)
    records_b, _ = enrich_records_from_disk(records_b)

    summary_a = summarize_run(records_a)
    summary_b = summarize_run(records_b)
    encoder = _SeverityEncoder()
    drift = compute_drift(
        summary_a,
        summary_b,
        threshold_pp=threshold_pp,
        records_a=records_a,
        records_b=records_b,
        severity_encoder=encoder,
    )
    return write_drift_report(
        model=model,
        run_a_meta=_run_meta(records_a, run_a_id),
        run_b_meta=_run_meta(records_b, run_b_id),
        summary_a=summary_a,
        summary_b=summary_b,
        drift=drift,
        out_dir=reports_dir,
    )
