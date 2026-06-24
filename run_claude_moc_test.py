"""Run the MOC evidence pack against Anthropic Claude with calibration disabled.

This is the Claude-provider counterpart to ``run_moc_evidence.py`` (GPT) and
``run_gemini_moc_test.py`` (Gemini). The CSV input, output JSON shape,
manifest format, resume logic, dry-run, --limit, and 2-second inter-call
delay all match the other provider scripts one-for-one so that downstream
analyzers (``analyze_moc_evidence.py``, ``show_moc_failures.py``) can read
any provider's run directory unchanged.

What differs from run_moc_evidence.py:
  * API provider          : Anthropic Claude via ``anthropic``
  * API key env var       : ``ANTHROPIC_API_KEY``
  * Default model         : ``claude-3-5-sonnet-20241022``
                            (use ``--model claude-3-5-haiku-20241022`` for
                             faster/cheaper smoke tests)
  * Default output prefix : ``claude_moc_<timestamp>``
  * Manifest ``provider`` : ``"anthropic"``

Everything else -- the calibration kill-switches, the Phase 1/2/3 pipeline,
the per-scenario JSON layout, the manifest schema -- is identical.

Install once:
    pip install anthropic

Set the key:
    PowerShell:  $env:ANTHROPIC_API_KEY = "your-key-here"
    cmd.exe   :  set ANTHROPIC_API_KEY=your-key-here
    bash/zsh  :  export ANTHROPIC_API_KEY="your-key-here"

Typical usage:
    # 1. Smoke-test 2 scenarios on the cheap haiku model:
    python run_claude_moc_test.py --limit 2 --model claude-3-5-haiku-20241022

    # 2. Full 50-scenario evidence run on sonnet:
    python run_claude_moc_test.py

    # 3. Resume an interrupted run:
    python run_claude_moc_test.py --resume --out-dir pipeline_outputs/claude_moc_<ts>
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# DISABLE CALIBRATION BEFORE ANY DOWNSTREAM CODE CAN READ THE FLAG.
# ---------------------------------------------------------------------------
import risk_calibration
risk_calibration.CALIBRATION_ENABLED = False

import governed_prompt_run
governed_prompt_run.CALIBRATION_ENABLED = False

from governed_prompt_run import adapt_wrapper_record_for_pipeline
from model_wrapper import SCHEMA_INSTRUCTIONS, normalize_record
from phase2_gate import validate_record
from phase3_gate import evaluate_phase3
from run_full_pipeline import (
    compute_final_execution_gate,
    run_phase1_adapter,
    safe_upper,
)


DEFAULT_CSV = Path("scenarios/moc_evidence_pack.csv")
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_DELAY_S = 2.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
PROVIDER = "anthropic"
PHASE4_MODEL = "claude"
RISK_FIELDS = ("uncertainty", "potential_harm", "irreversibility", "time_pressure")

# Anthropic Messages API requires the system prompt to enforce the schema
# (there is no equivalent of Gemini's response_mime_type). We append a
# very explicit "JSON only" reinforcer to SCHEMA_INSTRUCTIONS so Claude
# does not wrap the object in prose, code fences, or a preamble.
CLAUDE_SYSTEM_PROMPT = SCHEMA_INSTRUCTIONS + (
    "\n"
    "CRITICAL OUTPUT CONSTRAINT FOR THIS API:\n"
    "Your entire response must be a single JSON object and nothing else.\n"
    "Do not include any text before or after the JSON.\n"
    "Do not wrap the JSON in markdown code fences (no ``` backticks).\n"
    "Do not include explanations, headers, or preambles."
)


# ---------------------------------------------------------------------------
# Force UTF-8 stdout so non-ASCII output never crashes the run on Windows.
# ---------------------------------------------------------------------------
def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Anthropic client + structured request
# ---------------------------------------------------------------------------
def _build_claude_client():
    """Build an Anthropic client. Raises with a clear message on missing key/SDK."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic SDK is not installed. "
            "Install it with: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in your shell before "
            "running this script, or use --dry-run to preview."
        )

    return anthropic.Anthropic(api_key=api_key)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Pull a JSON object out of a Claude response.

    Defensive against the few ways Claude can deviate from a strict schema:
      * leading/trailing whitespace
      * markdown code fences (```json ... ``` or ``` ... ```)
      * a short preamble line before the JSON
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        # Trim trailing fence if any.
        if s.endswith("```"):
            s = s[:-3].strip()
    # Last resort: pull the first { ... } block out of free-form text.
    if not s.startswith("{"):
        m = _JSON_OBJECT_RE.search(s)
        if m:
            s = m.group(0)
    return s


def get_claude_record(
    user_prompt: str,
    model_name: str,
    client=None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Mirror of ``model_wrapper.get_model_record`` for Anthropic Claude.

    Returns a normalized record with the same keys the rest of the pipeline
    expects (proposed_action, rationale, uncertainty, potential_harm,
    irreversibility, time_pressure, context_tag, use_domain).
    """
    if client is None:
        client = _build_claude_client()

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Concatenate any text blocks (Claude returns a list of content blocks).
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    raw_text = "".join(parts).strip()

    if not raw_text:
        stop_reason = getattr(response, "stop_reason", None)
        raise ValueError(
            "Claude returned an empty response. "
            f"stop_reason={stop_reason}"
        )

    candidate = _extract_json_object(raw_text)
    try:
        record = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Claude did not return valid JSON.\n"
            f"Raw response:\n{raw_text}"
        ) from exc

    return normalize_record(record, user_prompt)


# ---------------------------------------------------------------------------
# CSV ingestion (identical to run_moc_evidence.py)
# ---------------------------------------------------------------------------
def load_scenarios(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Pipeline path -- calibration is NEVER called here (identical logic)
# ---------------------------------------------------------------------------
def run_pipeline_no_calibration(raw_record: dict, scenario_id: str) -> dict:
    adapted = adapt_wrapper_record_for_pipeline(raw_record)

    phase1_record = run_phase1_adapter(
        scenario_id=scenario_id,
        proposed_action=adapted.get("proposed_action", ""),
        uncertainty=adapted.get("uncertainty", ""),
        potential_harm=adapted.get("potential_harm", ""),
        irreversibility=adapted.get("irreversibility", ""),
        time_pressure=adapted.get("time_pressure", ""),
        context_tag=adapted.get("context_tag", ""),
        use_domain=adapted.get("use_domain", ""),
    )
    phase1_posture = safe_upper(phase1_record.get("posture", ""))

    phase2_outcome, phase2_reason = validate_record(phase1_record)
    phase2_outcome = safe_upper(phase2_outcome)

    phase3_result = evaluate_phase3(phase1_record)
    phase3_output = safe_upper(phase3_result.get("phase3_output", ""))

    final_gate = compute_final_execution_gate(
        phase1_posture=phase1_posture,
        phase2_outcome=phase2_outcome,
        phase3_output=phase3_output,
    )

    return {
        "adapted_record": adapted,
        "phase1_record": phase1_record,
        "phase1_posture": phase1_posture,
        "phase2_result": {"outcome": phase2_outcome, "reason": phase2_reason},
        "phase3_result": phase3_result,
        "phase3_output": phase3_output,
        "final_execution_gate": final_gate,
    }


# ---------------------------------------------------------------------------
# Per-scenario processing
# ---------------------------------------------------------------------------
def _extract_csv_intended(row: dict) -> dict:
    return {
        "uncertainty":       (row.get("uncertainty", "") or "").strip().upper(),
        "potential_harm":    (row.get("potential_harm", "") or "").strip().upper(),
        "irreversibility":   (row.get("irreversibility", "") or "").strip().upper(),
        "time_pressure":     (row.get("time_pressure", "") or "").strip().upper(),
        "context_tag":       (row.get("context_tag", "") or "").strip().upper(),
        "use_domain":        (row.get("use_domain", "") or "").strip().upper(),
        "expected_phase1":   (row.get("expected_phase1", "") or "").strip().upper(),
        "expected_phase3":   (row.get("expected_phase3", "") or "").strip().upper(),
    }


def _extract_raw_risk(raw_record: dict) -> dict:
    return {f: raw_record.get(f) for f in RISK_FIELDS}


def process_scenario(
    row: dict,
    out_dir: Path,
    model_name: str,
    claude_client,
    idx: int,
    total: int,
    resume: bool,
    max_tokens: int,
    temperature: float,
) -> dict:
    scenario_id = (row.get("scenario_id") or f"MOC-{idx:03d}").strip()
    prompt = (row.get("proposed_action") or "").strip()
    out_path = out_dir / f"{scenario_id}.json"

    if resume and out_path.exists():
        print(f"  [{idx:>2}/{total}] {scenario_id}: already exists -> SKIPPING (resume mode)")
        return {
            "scenario_id": scenario_id,
            "status": "skipped_resume",
            "path": str(out_path),
        }

    started_at = datetime.now(UTC).isoformat()
    t0 = time.time()
    print(f"  [{idx:>2}/{total}] {scenario_id}: calling Claude...")

    record_out: dict[str, Any] = {
        "scenario_id": scenario_id,
        "csv_prompt": prompt,
        "csv_intended": _extract_csv_intended(row),
        "csv_notes": (row.get("notes") or "").strip(),
        "provider": PROVIDER,
        "model_name": model_name,
        "calibration_skipped": True,
        "calibration_source_flags": {
            "risk_calibration.CALIBRATION_ENABLED": risk_calibration.CALIBRATION_ENABLED,
            "governed_prompt_run.CALIBRATION_ENABLED": governed_prompt_run.CALIBRATION_ENABLED,
        },
        "started_at_utc": started_at,
    }

    # ---- live model call ----
    try:
        raw_record = get_claude_record(
            prompt,
            model_name=model_name,
            client=claude_client,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        record_out["status"] = "model_call_failed"
        record_out["error"] = f"{type(exc).__name__}: {exc}"
        record_out["traceback"] = traceback.format_exc()
        record_out["duration_s"] = round(time.time() - t0, 3)
        record_out["finished_at_utc"] = datetime.now(UTC).isoformat()
        out_path.write_text(
            json.dumps(record_out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"       ERROR (model call): {exc}")
        return {
            "scenario_id": scenario_id,
            "status": "model_call_failed",
            "path": str(out_path),
            "error": str(exc),
        }

    record_out["raw_model_record"] = raw_record
    record_out["raw_risk_fields"] = _extract_raw_risk(raw_record)

    # ---- pipeline (no calibration) ----
    try:
        pipeline_result = run_pipeline_no_calibration(raw_record, scenario_id)
        record_out["pipeline_result"] = pipeline_result
        record_out["status"] = "ok"
    except Exception as exc:
        record_out["status"] = "pipeline_failed"
        record_out["pipeline_error"] = f"{type(exc).__name__}: {exc}"
        record_out["pipeline_traceback"] = traceback.format_exc()

    record_out["duration_s"] = round(time.time() - t0, 3)
    record_out["finished_at_utc"] = datetime.now(UTC).isoformat()
    out_path.write_text(
        json.dumps(record_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    raw_q = " / ".join(
        str(record_out["raw_risk_fields"].get(f) or "?")
        for f in RISK_FIELDS
    )
    posture = record_out.get("pipeline_result", {}).get("phase1_posture", "?")
    print(f"       raw={raw_q}  P1={posture}  ({record_out['duration_s']}s)")

    return {
        "scenario_id": scenario_id,
        "status": record_out["status"],
        "path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(out_dir: Path, results: list[dict], config: dict) -> None:
    manifest = {
        "run_started_at_utc": config["started_at_utc"],
        "run_finished_at_utc": datetime.now(UTC).isoformat(),
        "scenario_csv": str(config["csv_path"]),
        "provider": PROVIDER,
        "model_name": config["model"],
        "delay_s_between_calls": config["delay"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "calibration_state": {
            "risk_calibration.CALIBRATION_ENABLED": risk_calibration.CALIBRATION_ENABLED,
            "governed_prompt_run.CALIBRATION_ENABLED": governed_prompt_run.CALIBRATION_ENABLED,
        },
        "results": results,
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "model_call_failed": sum(1 for r in results if r["status"] == "model_call_failed"),
            "pipeline_failed": sum(1 for r in results if r["status"] == "pipeline_failed"),
            "skipped_resume": sum(1 for r in results if r["status"] == "skipped_resume"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_claude_moc_test",
        description=(
            "Run the MOC evidence pack against Anthropic Claude with risk "
            "calibration disabled. One JSON per scenario. Resume-safe."
        ),
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Scenario CSV path (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            f"Claude model name (default: {DEFAULT_MODEL}). "
            "Use claude-3-5-haiku-20241022 for cheaper/faster smoke tests."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens per call (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY_S,
        help=f"Seconds to sleep between API calls (default: {DEFAULT_DELAY_S}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir (default: pipeline_outputs/claude_moc_<timestamp>).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Skip scenarios that already have a JSON file in --out-dir. "
            "Combine with --out-dir <existing run> to resume an interrupted "
            "run without re-burning API credits."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N scenarios (use for smoke tests).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not call the API. Print what would be called and exit.",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help=(
            "Skip automatic Phase 4 registration, drift comparison, and "
            "summary report after a successful run."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    _force_utf8_stdout()
    args = _parse_args(argv)

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    scenarios = load_scenarios(args.csv)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    total = len(scenarios)
    if total == 0:
        print(f"ERROR: no scenarios loaded from {args.csv}", file=sys.stderr)
        return 1

    if args.out_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = Path("pipeline_outputs") / f"claude_moc_{timestamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_claude_moc_test] CSV          : {args.csv}")
    print(f"[run_claude_moc_test] Scenarios    : {total}")
    print(f"[run_claude_moc_test] Provider     : {PROVIDER}")
    print(f"[run_claude_moc_test] Model        : {args.model}")
    print(f"[run_claude_moc_test] Temperature  : {args.temperature}")
    print(f"[run_claude_moc_test] Max tokens   : {args.max_tokens}")
    print(f"[run_claude_moc_test] Delay        : {args.delay}s between calls")
    print(f"[run_claude_moc_test] Output dir   : {args.out_dir}")
    print(f"[run_claude_moc_test] Resume mode  : {args.resume}")
    print(f"[run_claude_moc_test] Dry run      : {args.dry_run}")
    print(
        f"[run_claude_moc_test] Calibration  : DISABLED "
        f"(risk_calibration={risk_calibration.CALIBRATION_ENABLED}, "
        f"governed_prompt_run={governed_prompt_run.CALIBRATION_ENABLED})"
    )
    print()

    if args.dry_run:
        for i, row in enumerate(scenarios, start=1):
            sid = row.get("scenario_id", "?")
            prompt = (row.get("proposed_action") or "").strip()
            print(f"  [{i:>2}/{total}] {sid}: would call Claude on prompt: "
                  f"{prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        return 0

    # Build the Claude client once and reuse for every scenario. This also
    # surfaces a missing API key / missing SDK before we touch the CSV loop.
    try:
        claude_client = _build_claude_client()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "csv_path": args.csv,
        "model": args.model,
        "delay": args.delay,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    results: list[dict] = []
    try:
        for i, row in enumerate(scenarios, start=1):
            result = process_scenario(
                row,
                args.out_dir,
                args.model,
                claude_client,
                i,
                total,
                args.resume,
                args.max_tokens,
                args.temperature,
            )
            results.append(result)
            write_manifest(args.out_dir, results, config)
            if i < total and result["status"] != "skipped_resume":
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n[run_claude_moc_test] Interrupted by user. Writing final manifest...")
        write_manifest(args.out_dir, results, config)
        return 130

    ok = sum(1 for r in results if r["status"] == "ok")
    mc_failed = sum(1 for r in results if r["status"] == "model_call_failed")
    pl_failed = sum(1 for r in results if r["status"] == "pipeline_failed")
    skipped = sum(1 for r in results if r["status"] == "skipped_resume")

    print()
    print("=" * 64)
    print(
        f"DONE. ok={ok}  model_call_failed={mc_failed}  "
        f"pipeline_failed={pl_failed}  skipped_resume={skipped}  total={total}"
    )
    print(f"Outputs : {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'manifest.json'}")
    print("=" * 64)

    if (mc_failed + pl_failed) == 0:
        from phase4_auto import post_run_phase4_tracking

        post_run_phase4_tracking(
            model=PHASE4_MODEL,
            run_dir=args.out_dir,
            enabled=not args.no_register,
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
