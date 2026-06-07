"""phase4_history_loader.py

Single source of truth for plotting scripts: load per-model phase-4
history (``phase4_model_history/<model>.jsonl``) and serve records
grouped by registered run, sorted chronologically.

History records carry the raw risk fields, model rationale, duration,
final disposition, etc. (see ``phase4_drift_per_model.to_phase4_record``).
Fields that are NOT stored in history -- notably ``csv_intended`` (the
expected baseline) and ``csv_prompt`` (the scenario text) -- can be
filled in on demand from the original ``MOC-*.json`` files via the
``run_directory`` field every record carries.

Typical usage in a plotting script::

    from phase4_history_loader import load_runs, enrich_with_intended

    runs = []
    for model in ("gpt", "gemini", "claude", "grok"):
        runs.extend(load_runs(model, include="all"))   # all v1, v2, v3, ...
    enrich_with_intended(runs)                          # adds csv_intended / csv_prompt

    for run in runs:
        print(run.label, len(run.records))              # "GPT v1", 49
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_HISTORY_DIR = Path("phase4_model_history")
KNOWN_MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude", "grok")

MODEL_DISPLAY: Dict[str, str] = {
    "gpt":    "GPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "grok":   "Grok",
}

RISK_FIELDS: Tuple[str, ...] = (
    "uncertainty",
    "potential_harm",
    "irreversibility",
    "time_pressure",
)
RISK_LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RunGroup:
    """All scenario records for one registered run of one model.

    Records are the exact dicts read from ``<model>.jsonl`` (mutable, so
    enrichment can be done in place).
    """
    model: str
    run_id: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    sort_key: str = ""  # earliest timestamp_utc seen in this run (for ordering)

    @property
    def label(self) -> str:
        """Short series label suitable for an axis tick, e.g. ``GPT v1``."""
        return f"{self.model.upper()} {self.run_id}"

    def by_scenario(self) -> Dict[str, Dict[str, Any]]:
        """``{scenario_id: record}`` for callers that prefer dict lookup."""
        return {
            str(r["scenario_id"]): r
            for r in self.records
            if r.get("scenario_id")
        }


# ---------------------------------------------------------------------------
# History I/O
# ---------------------------------------------------------------------------
def history_path(
    model: str, *, history_dir: Path = DEFAULT_HISTORY_DIR
) -> Path:
    return history_dir / f"{model.lower()}.jsonl"


def load_history(
    model: str, *, history_dir: Path = DEFAULT_HISTORY_DIR
) -> List[Dict[str, Any]]:
    """Return every record for ``model`` in file (append) order."""
    p = history_path(model, history_dir=history_dir)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    text = p.read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            print(
                f"WARNING: skipping malformed line {line_no} in {p}: {exc}",
                file=sys.stderr,
            )
    return records


def group_records_by_run(records: List[Dict[str, Any]]) -> List[RunGroup]:
    """Return per-run groups in chronological order.

    Sort key per run is ``(earliest_timestamp_utc, first_appearance_index)``
    -- matches ``phase4_justification_drift.runs_in_chronological_order``
    so the two scripts always agree on what "consecutive" means.
    """
    first_appearance: Dict[str, int] = {}
    earliest: Dict[str, str] = {}
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    model_name = ""
    for idx, rec in enumerate(records):
        rid = rec.get("run_id")
        if not rid:
            continue
        if rid not in first_appearance:
            first_appearance[rid] = idx
            by_run[rid] = []
        by_run[rid].append(rec)
        if not model_name:
            model_name = str(rec.get("model_name") or "")
        ts = rec.get("timestamp_utc") or rec.get("registered_at_utc") or ""
        if ts and (rid not in earliest or ts < earliest[rid]):
            earliest[rid] = ts

    ordered = sorted(
        by_run.keys(),
        key=lambda rid: (earliest.get(rid, ""), first_appearance[rid]),
    )
    return [
        RunGroup(
            model=model_name,
            run_id=rid,
            records=by_run[rid],
            sort_key=earliest.get(rid, ""),
        )
        for rid in ordered
    ]


def select_runs(
    groups: List[RunGroup], *, include: str = "all"
) -> List[RunGroup]:
    """Filter chronologically-ordered groups by ``include`` spec.

    ``include`` is one of:
      - ``"all"`` (default): every registered run
      - ``"latest"``: only the most recent run
      - ``"first"``:  only the oldest run
      - any other string: treated as a comma-separated list of explicit
        run_ids (e.g. ``"v1,v3"``)
    """
    if not groups:
        return []
    inc = (include or "all").strip().lower()
    if inc == "all":
        return list(groups)
    if inc == "latest":
        return [groups[-1]]
    if inc == "first":
        return [groups[0]]
    wanted = {tok.strip() for tok in include.split(",") if tok.strip()}
    return [g for g in groups if g.run_id in wanted]


def load_runs(
    model: str,
    *,
    include: str = "all",
    history_dir: Path = DEFAULT_HISTORY_DIR,
) -> List[RunGroup]:
    """Convenience: load and filter in one call."""
    groups = group_records_by_run(
        load_history(model, history_dir=history_dir)
    )
    return select_runs(groups, include=include)


def load_runs_for_models(
    models: Iterable[str],
    *,
    include: str = "all",
    history_dir: Path = DEFAULT_HISTORY_DIR,
) -> List[RunGroup]:
    """Load runs for several models and return a flat chronologically-ordered
    list, grouped by model (model order preserved from ``models`` arg).
    """
    out: List[RunGroup] = []
    for m in models:
        out.extend(load_runs(m, include=include, history_dir=history_dir))
    return out


# ---------------------------------------------------------------------------
# Lazy enrichment: pull csv_intended / csv_prompt back from disk
# ---------------------------------------------------------------------------
def _read_moc_json(run_dir: Path, scenario_id: str) -> Optional[dict]:
    p = run_dir / f"{scenario_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] could not read {p}: {exc}", file=sys.stderr)
        return None


def enrich_with_intended(groups: Iterable[RunGroup]) -> None:
    """In-place: add ``csv_intended`` / ``csv_prompt`` to each record by
    re-reading the original ``MOC-*.json`` file from ``run_directory``.

    Records that already have these fields are left alone. Records whose
    source file is missing get empty defaults so callers don't crash.
    """
    cache: Dict[Tuple[str, str], dict] = {}
    for g in groups:
        for rec in g.records:
            if "csv_intended" in rec and "csv_prompt" in rec:
                continue
            run_dir_str = rec.get("run_directory")
            sid = rec.get("scenario_id")
            if not run_dir_str or not sid:
                rec.setdefault("csv_intended", {})
                rec.setdefault("csv_prompt", "")
                continue
            key = (run_dir_str, sid)
            payload = cache.get(key)
            if payload is None:
                payload = _read_moc_json(Path(run_dir_str), sid) or {}
                cache[key] = payload
            rec.setdefault("csv_intended", payload.get("csv_intended") or {})
            rec.setdefault("csv_prompt", payload.get("csv_prompt") or "")


# ---------------------------------------------------------------------------
# Tiny utilities re-used by every plot
# ---------------------------------------------------------------------------
def norm_level(value: object) -> Optional[str]:
    """Normalize a risk-level string to LOW/MEDIUM/HIGH (or None)."""
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if s in RISK_LEVELS else None


def raw_risk_field(rec: Dict[str, Any], field_name: str) -> Optional[str]:
    """Risk level for ``field_name`` from a history record."""
    return norm_level(rec.get(f"raw_{field_name}"))


def intended_risk_field(rec: Dict[str, Any], field_name: str) -> Optional[str]:
    """Intended risk level (requires ``enrich_with_intended`` first)."""
    return norm_level((rec.get("csv_intended") or {}).get(field_name))


def binary_disposition(rec: Dict[str, Any]) -> Optional[str]:
    """Map a history record to ``PROCEED`` / ``BLOCKED`` / None.

    Mirrors ``_extract_final_disposition`` used by the older plot scripts
    so the new history-based path produces identical numbers.
    """
    allowed = rec.get("execution_allowed")
    raw = rec.get("final_disposition")
    raw_str = str(raw).strip().upper() if raw else None
    if allowed is None and not raw_str:
        return None
    if allowed is True or raw_str == "EXECUTION_ALLOWED":
        return "PROCEED"
    if allowed is False or (raw_str and raw_str.startswith("BLOCKED")):
        return "BLOCKED"
    return None


def expected_disposition(rec: Dict[str, Any]) -> Optional[str]:
    """Derive the *intended* PROCEED/BLOCKED label from csv_intended.

    Same rule the live pipeline uses: PROCEED only when expected_phase1 is
    PROCEED AND expected_phase3 is ETHICAL_PASS (or empty); otherwise
    BLOCKED. Requires ``enrich_with_intended`` to have been called.
    """
    intended = rec.get("csv_intended") or {}
    p1 = str(intended.get("expected_phase1") or "").strip().upper()
    p3 = str(intended.get("expected_phase3") or "").strip().upper()
    if not p1:
        return None
    if p1 == "PROCEED" and (not p3 or p3 == "ETHICAL_PASS"):
        return "PROCEED"
    return "BLOCKED"


def union_scenario_ids(groups: Iterable[RunGroup]) -> List[str]:
    """Sorted union of scenario_ids across the given run groups."""
    ids: set = set()
    for g in groups:
        for r in g.records:
            sid = r.get("scenario_id")
            if sid:
                ids.add(str(sid))
    return sorted(ids)


def union_scenario_ids_from_lookups(
    *lookups: Dict[str, dict],
) -> List[str]:
    """Sorted union of scenario_ids from scenario-lookup dicts."""
    ids: set = set()
    for lookup in lookups:
        ids.update(lookup.keys())
    return sorted(ids)


def scenario_lookup_from_group(group: RunGroup) -> Dict[str, dict]:
    """Convert a ``RunGroup`` to the ``{scenario_id: record}`` shape used by
    the legacy ``plot_*.py`` scripts (``load_run_dir`` output).

    Call ``enrich_with_intended`` on the group first if you need
    ``csv_intended`` / ``csv_prompt`` / ``expected*`` fields.
    """
    out: Dict[str, dict] = {}
    for rec in group.records:
        sid = str(rec.get("scenario_id") or "")
        if not sid:
            continue
        disp = binary_disposition(rec)
        raw_disp = rec.get("final_disposition")
        exp = expected_disposition(rec)
        out[sid] = {
            "raw": {
                f: raw_risk_field(rec, f) for f in RISK_FIELDS
            },
            "intended": {
                f: intended_risk_field(rec, f) for f in RISK_FIELDS
            },
            "expected": exp,
            "expected_disp": exp,
            "actual": disp,
            "actual_disp": disp,
            "actual_raw": raw_disp,
            "actual_disp_raw": raw_disp,
            "csv_prompt": rec.get("csv_prompt") or "",
            "status": rec.get("status"),
        }
    return out


def format_series_label(group: RunGroup, runs_for_model: int) -> str:
    """Axis label for a run series, e.g. ``GPT v2`` or just ``GPT``."""
    base = MODEL_DISPLAY.get(group.model.lower(), group.model.upper())
    if runs_for_model <= 1:
        return base
    return f"{base} {group.run_id}"


def load_labeled_lookups(
    models: Iterable[str],
    *,
    include: str = "all",
    history_dir: Path = DEFAULT_HISTORY_DIR,
    enrich: bool = True,
) -> List[Tuple[str, Dict[str, dict]]]:
    """Load history and return ``[(label, scenario_lookup), ...]``.

    Runs are grouped by model (in ``models`` order) and sorted
    chronologically within each model. Labels omit the run suffix when
    only one run is selected for that model (``--include-runs latest``).
    """
    model_list = [m.lower() for m in models]
    all_groups: List[RunGroup] = []
    per_model_groups: Dict[str, List[RunGroup]] = {}
    for model in model_list:
        groups = load_runs(model, include=include, history_dir=history_dir)
        per_model_groups[model] = groups
        all_groups.extend(groups)

    if enrich and all_groups:
        enrich_with_intended(all_groups)

    out: List[Tuple[str, Dict[str, dict]]] = []
    for model in model_list:
        n = len(per_model_groups.get(model, []))
        for group in per_model_groups.get(model, []):
            label = format_series_label(group, n)
            out.append((label, scenario_lookup_from_group(group)))
    return out


def intended_for_scenarios(
    scenario_ids: Iterable[str],
    *lookups: Dict[str, dict],
    field: str,
) -> List[Optional[str]]:
    """Pull intended risk level for ``field`` from the first lookup that has it."""
    out: List[Optional[str]] = []
    for sid in scenario_ids:
        v = None
        for lookup in lookups:
            rec = lookup.get(sid)
            if rec and rec.get("intended", {}).get(field) is not None:
                v = rec["intended"][field]
                break
        out.append(v)
    return out


def expected_dispositions_for_scenarios(
    scenario_ids: Iterable[str],
    *lookups: Dict[str, dict],
) -> List[Optional[str]]:
    """Pull expected PROCEED/BLOCKED from the first lookup that has it."""
    out: List[Optional[str]] = []
    for sid in scenario_ids:
        v = None
        for lookup in lookups:
            rec = lookup.get(sid)
            if rec is None:
                continue
            exp = rec.get("expected") or rec.get("expected_disp")
            if exp is not None:
                v = exp
                break
        out.append(v)
    return out
