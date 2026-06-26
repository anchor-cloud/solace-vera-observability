"""phase4_drift_per_model.py

Per-model run registration + drift detection on top of the existing
Phase 4 summary logic.

This is a *parallel* tracking system. The canonical
``phase4_history/phase4_history.jsonl`` is never read or written by this
script. Instead, runs are appended to per-model JSONL files at:

    phase4_model_history/<model>.jsonl

Each line in those files is one scenario record (same shape as
``phase4_history.jsonl`` lines) with two extra book-keeping fields:
``model_name`` and ``run_id`` (plus ``registered_at_utc`` and
``run_directory`` for traceability).

Modes (mutually exclusive):

    --register   Append a run to <model>.jsonl
    --compare    Compare two specific run IDs for the same model
    --drift      Auto-compare the two most recently registered runs

Examples:

    # Register the GPT v1 run
    python phase4_drift_per_model.py --model gpt --run-id v1 \
        --run-dir pipeline_outputs/moc_evidence_20260503T051527Z --register

    # Register a second GPT run
    python phase4_drift_per_model.py --model gpt --run-id v2 \
        --run-dir pipeline_outputs/moc_evidence_20260603T052801Z --register

    # Manually compare two run IDs
    python phase4_drift_per_model.py --model gpt --run-id v1 --run-id v2 --compare

    # Auto-compare the last two runs and flag deltas > 10 pp
    python phase4_drift_per_model.py --model gpt --drift

    # Auto-compare with a custom threshold
    python phase4_drift_per_model.py --model gpt --drift --threshold 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Live phase 4 summary builder.
from run_full_pipeline import build_phase4_summary  # noqa: E402

# Reuse JSON->record adapter and small helpers so all conversion logic
# lives in exactly one place.
from phase4_per_model_analysis import (  # noqa: E402
    BLOCKED_DISPOSITION_PREFIXES,
    AMBIGUITY_PHASE3_OUTPUTS,
    RISK_DIMENSIONS,
    compute_risk_distribution,
    compute_timing_stats,
    load_moc_json_files,
    raw_risks_from_record,
    to_phase4_record,
    _final_disposition_counts,
    _percentage_helper,
)


# ---------------------------------------------------------------------------
# Defaults for severity (justification similarity) backend
# ---------------------------------------------------------------------------
DEFAULT_SEVERITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAJOR_SEVERITY_THRESHOLD = 30.0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
HISTORY_DIR_DEFAULT = Path("phase4_model_history")
DRIFT_REPORTS_DIR_DEFAULT = Path("phase4_drift_reports")
DEFAULT_THRESHOLD_PP = 10.0

KNOWN_MODELS = ("gpt", "gemini", "claude", "grok")


# ---------------------------------------------------------------------------
# Per-model history I/O
# ---------------------------------------------------------------------------
def history_path(model: str, *, history_dir: Path = HISTORY_DIR_DEFAULT) -> Path:
    return history_dir / f"{model.lower()}.jsonl"


def load_history(model: str, *, history_dir: Path = HISTORY_DIR_DEFAULT) -> List[Dict[str, Any]]:
    """Return all records for a given model in append order (oldest first)."""
    path = history_path(model, history_dir=history_dir)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: skipping malformed line {line_no} in {path}: {exc}",
                    file=sys.stderr,
                )
    return records


def existing_run_ids(model: str, *, history_dir: Path = HISTORY_DIR_DEFAULT) -> List[str]:
    """Return distinct run_ids in registration order (first-seen wins)."""
    seen: "OrderedDict[str, None]" = OrderedDict()
    for rec in load_history(model, history_dir=history_dir):
        rid = rec.get("run_id")
        if rid and rid not in seen:
            seen[rid] = None
    return list(seen.keys())


def filter_records_for_run(
    records: List[Dict[str, Any]], run_id: str
) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("run_id") == run_id]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_run(
    *,
    model: str,
    run_id: str,
    run_dir: Path,
    force: bool = False,
    history_dir: Path = HISTORY_DIR_DEFAULT,
) -> Dict[str, Any]:
    """Append every usable scenario record from ``run_dir`` to <model>.jsonl.

    The canonical ``phase4_history.jsonl`` is never touched.

    Returns a small registration receipt dict.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir not found: {run_dir}")

    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_path(model, history_dir=history_dir)

    existing = load_history(model, history_dir=history_dir)
    if any(r.get("run_id") == run_id for r in existing):
        if not force:
            raise ValueError(
                f"Run id '{run_id}' already exists for model '{model}'. "
                f"Re-run with --force to overwrite it."
            )
        # Force overwrite: rewrite the file without the matching run_id.
        survivors = [r for r in existing if r.get("run_id") != run_id]
        with target.open("w", encoding="utf-8") as f:
            for r in survivors:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payloads = load_moc_json_files(run_dir)
    registered_at = datetime.now(UTC).isoformat()
    appended = 0
    skipped: List[str] = []

    with target.open("a", encoding="utf-8") as f:
        for payload in payloads:
            rec = to_phase4_record(payload)
            if rec is None:
                skipped.append(payload.get("scenario_id") or Path(payload.get("__source_path", "")).name)
                continue
            rec["model_name"] = model.lower()
            rec["run_id"] = run_id
            rec["registered_at_utc"] = registered_at
            rec["run_directory"] = str(run_dir.resolve())
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            appended += 1

    return {
        "model": model.lower(),
        "run_id": run_id,
        "run_directory": str(run_dir.resolve()),
        "history_file": str(target.resolve()),
        "registered_at_utc": registered_at,
        "scenario_records_appended": appended,
        "scenario_files_skipped": skipped,
        "force_overwrite": force,
    }


# ---------------------------------------------------------------------------
# Backfilling raw fields from disk for records registered before the
# extended ``to_phase4_record`` schema landed.
# ---------------------------------------------------------------------------
_EXTENDED_KEYS = (
    "raw_uncertainty",
    "raw_potential_harm",
    "raw_irreversibility",
    "raw_time_pressure",
    "model_rationale",
    "rationale_length",
    "duration_s",
)


def _record_needs_enrichment(rec: Dict[str, Any]) -> bool:
    """A legacy record is missing every extended field; partial is fine."""
    return not any(k in rec for k in _EXTENDED_KEYS)


def enrich_records_from_disk(
    records: List[Dict[str, Any]],
    *,
    warn: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """Backfill missing raw-risk / rationale / duration fields from JSON files.

    Records registered before the extended schema do not have the new
    fields. Each such record carries a ``run_directory`` and ``scenario_id``
    so we can transparently re-read the per-scenario JSON and merge the new
    fields in.

    Returns the (possibly enriched) records and the number of records that
    were actually backfilled. Records whose source file no longer exists
    pass through unchanged.
    """
    enriched_count = 0
    out: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    for rec in records:
        if not _record_needs_enrichment(rec):
            out.append(rec)
            continue

        run_dir = rec.get("run_directory")
        scenario_id = rec.get("scenario_id")
        if not run_dir or not scenario_id:
            out.append(rec)
            continue

        # The per-model JSONL history is the source of truth. If the referenced
        # run directory no longer exists on disk (e.g. it was archived or
        # cleaned up), skip the directory lookup entirely and use the JSONL
        # record as-is rather than failing or scanning for the folder.
        if not Path(run_dir).is_dir():
            out.append(rec)
            continue

        key = (str(run_dir), str(scenario_id))
        if key not in cache:
            json_path = Path(run_dir) / f"{scenario_id}.json"
            if not json_path.exists():
                cache[key] = None
            else:
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    payload["__source_path"] = str(json_path)
                    enriched = to_phase4_record(payload)
                except (OSError, json.JSONDecodeError) as exc:
                    if warn:
                        print(
                            f"WARNING: could not enrich {scenario_id} from "
                            f"{json_path}: {exc}",
                            file=sys.stderr,
                        )
                    enriched = None
                cache[key] = enriched

        enriched = cache[key]
        if enriched is None:
            out.append(rec)
            continue

        merged = dict(rec)
        for k in _EXTENDED_KEYS:
            if k in enriched and enriched[k] not in (None, ""):
                merged[k] = enriched[k]
        out.append(merged)
        enriched_count += 1

    if warn and enriched_count:
        print(
            f"[enrich] backfilled {enriched_count} legacy record(s) from "
            f"their run_directory JSON files.",
            file=sys.stderr,
        )

    return out, enriched_count


# ---------------------------------------------------------------------------
# Variance rate (per-dimension risk-level disagreement between two runs)
# ---------------------------------------------------------------------------
def compute_variance_rate(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """How often raw risk levels disagree between matched scenarios.

    For every scenario that appears in *both* runs and has all four raw
    risk levels populated on both sides, we check each of the four
    dimensions independently:

      * variance_rate (overall): % of paired scenarios where ANY dimension
        changed between Run A and Run B.
      * per_dimension[dim].changed_rate: % of paired scenarios where that
        specific dimension changed.

    Scenarios with missing raw fields on either side are excluded from
    the denominator and listed under ``unmatched_or_missing``.
    """
    by_id_a = {r.get("scenario_id"): r for r in records_a if r.get("scenario_id")}
    by_id_b = {r.get("scenario_id"): r for r in records_b if r.get("scenario_id")}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    paired = 0
    any_changed = 0
    per_dim_changed: Dict[str, int] = {dim: 0 for dim in RISK_DIMENSIONS}
    changed_examples: List[Dict[str, Any]] = []
    unmatched_or_missing: List[str] = []

    for sid in shared_ids:
        ra = raw_risks_from_record(by_id_a[sid])
        rb = raw_risks_from_record(by_id_b[sid])
        if any(not ra[d] for d in RISK_DIMENSIONS) or any(
            not rb[d] for d in RISK_DIMENSIONS
        ):
            unmatched_or_missing.append(sid)
            continue
        paired += 1
        scenario_changed_dims: List[str] = []
        for dim in RISK_DIMENSIONS:
            if ra[dim] != rb[dim]:
                per_dim_changed[dim] += 1
                scenario_changed_dims.append(dim)
        if scenario_changed_dims:
            any_changed += 1
            changed_examples.append({
                "scenario_id": sid,
                "changed_dimensions": scenario_changed_dims,
                "run_a": ra,
                "run_b": rb,
            })

    def pct(num: int) -> float:
        return round((num / paired) * 100.0, 2) if paired else 0.0

    changed_examples.sort(
        key=lambda e: (-len(e["changed_dimensions"]), e["scenario_id"])
    )

    return {
        "paired_scenarios": paired,
        "scenarios_with_any_change": any_changed,
        "variance_rate_pct": pct(any_changed),
        "per_dimension": {
            dim: {
                "changed_count": per_dim_changed[dim],
                "changed_rate_pct": pct(per_dim_changed[dim]),
            }
            for dim in RISK_DIMENSIONS
        },
        "unmatched_or_missing_raw_fields": unmatched_or_missing,
        "changed_scenarios": changed_examples,
    }


# ---------------------------------------------------------------------------
# Severity score (semantic similarity of justifications)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize_for_fallback(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Deterministic stdlib cosine: TF (no IDF) over lowercased word tokens.

    The fallback is intentionally simple; it is only used when
    sentence-transformers is not installed and exists so the script still
    produces a severity number on stock environments. Returns a value in
    [0, 1] where 1.0 means identical bag-of-words.
    """
    a_tokens = _tokenize_for_fallback(text_a)
    b_tokens = _tokenize_for_fallback(text_b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    shared = set(a_counts) & set(b_counts)
    if not shared:
        return 0.0
    num = sum(a_counts[t] * b_counts[t] for t in shared)
    den_a = math.sqrt(sum(v * v for v in a_counts.values()))
    den_b = math.sqrt(sum(v * v for v in b_counts.values()))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)


class _SeverityEncoder:
    """Lazy wrapper around sentence-transformers with a stdlib fallback.

    Loading the embedding model is expensive (first call downloads ~80MB).
    The wrapper caches a single instance per process and degrades to a
    TF-cosine fallback when sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = DEFAULT_SEVERITY_MODEL):
        self.model_name = model_name
        self._model = None
        self._backend = None  # filled on first encode

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            self._backend = self.model_name
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"WARNING: could not load sentence-transformers model "
                f"'{self.model_name}' ({exc.__class__.__name__}: {exc}). "
                f"Falling back to a stdlib TF-cosine. Install with "
                f"`pip install sentence-transformers` for the canonical "
                f"metric.",
                file=sys.stderr,
            )
            self._model = None
            self._backend = "tfidf_cosine_fallback"

    @property
    def backend(self) -> str:
        self._ensure_loaded()
        return self._backend or "tfidf_cosine_fallback"

    def cosine_similarities(
        self, pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Return cosine similarity for each (text_a, text_b) pair."""
        self._ensure_loaded()
        if self._model is None or not pairs:
            return [_tfidf_cosine(a, b) for a, b in pairs]

        texts_a = [a or "" for a, _ in pairs]
        texts_b = [b or "" for _, b in pairs]
        emb_a = self._model.encode(
            texts_a, convert_to_numpy=True, normalize_embeddings=True
        )
        emb_b = self._model.encode(
            texts_b, convert_to_numpy=True, normalize_embeddings=True
        )
        # Already L2-normalized, so dot product == cosine similarity.
        sims: List[float] = []
        for va, vb in zip(emb_a, emb_b):
            sim = float((va * vb).sum())
            sim = max(-1.0, min(1.0, sim))
            sims.append(sim)
        return sims


def compute_severity(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    *,
    encoder: Optional[_SeverityEncoder] = None,
    major_threshold: float = MAJOR_SEVERITY_THRESHOLD,
) -> Dict[str, Any]:
    """Semantic-distance score between matched justifications.

    Severity for a scenario = (1 - cosine_similarity) * 100. A severity of
    0 means identical justifications; 100 means orthogonal. Scenarios with
    severity strictly greater than ``major_threshold`` are flagged as
    "major justification change".
    """
    enc = encoder or _SeverityEncoder()
    by_id_a = {r.get("scenario_id"): r for r in records_a if r.get("scenario_id")}
    by_id_b = {r.get("scenario_id"): r for r in records_b if r.get("scenario_id")}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    pairs: List[Tuple[str, str]] = []
    sids: List[str] = []
    skipped: List[str] = []
    for sid in shared_ids:
        a_text = (by_id_a[sid].get("model_rationale") or "").strip()
        b_text = (by_id_b[sid].get("model_rationale") or "").strip()
        if not a_text or not b_text:
            skipped.append(sid)
            continue
        pairs.append((a_text, b_text))
        sids.append(sid)

    if not pairs:
        return {
            "backend": enc.backend,
            "major_threshold": major_threshold,
            "paired_scenarios": 0,
            "mean_severity": 0.0,
            "median_severity": 0.0,
            "max_severity": 0.0,
            "min_severity": 0.0,
            "major_change_count": 0,
            "major_change_pct": 0.0,
            "major_change_scenarios": [],
            "all_scenarios": [],
            "skipped_missing_rationale": skipped,
        }

    sims = enc.cosine_similarities(pairs)
    per_scenario: List[Dict[str, Any]] = []
    for sid, sim in zip(sids, sims):
        severity = round((1.0 - sim) * 100.0, 2)
        per_scenario.append({
            "scenario_id": sid,
            "cosine_similarity": round(sim, 4),
            "severity": severity,
            "is_major_change": severity > major_threshold,
        })

    per_scenario.sort(key=lambda e: (-e["severity"], e["scenario_id"]))

    severities = [e["severity"] for e in per_scenario]
    n = len(severities)
    mean_sev = sum(severities) / n
    sorted_sev = sorted(severities)
    median_sev = (
        sorted_sev[n // 2]
        if n % 2 == 1
        else (sorted_sev[n // 2 - 1] + sorted_sev[n // 2]) / 2.0
    )
    major_changes = [e for e in per_scenario if e["is_major_change"]]

    return {
        "backend": enc.backend,
        "major_threshold": major_threshold,
        "paired_scenarios": n,
        "mean_severity": round(mean_sev, 2),
        "median_severity": round(median_sev, 2),
        "max_severity": round(max(severities), 2),
        "min_severity": round(min(severities), 2),
        "major_change_count": len(major_changes),
        "major_change_pct": round((len(major_changes) / n) * 100.0, 2),
        "major_change_scenarios": major_changes,
        "all_scenarios": per_scenario,
        "skipped_missing_rationale": skipped,
    }


# ---------------------------------------------------------------------------
# Timing delta between two runs
# ---------------------------------------------------------------------------
def compute_timing_delta(
    timing_a: Dict[str, Any],
    timing_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare summary timing statistics across two runs."""
    def _sub(a: Any, b: Any) -> Optional[float]:
        try:
            return round(float(b) - float(a), 3)
        except (TypeError, ValueError):
            return None

    return {
        "mean_duration_s": {
            "run_a": timing_a.get("mean_duration_s"),
            "run_b": timing_b.get("mean_duration_s"),
            "delta": _sub(timing_a.get("mean_duration_s"), timing_b.get("mean_duration_s")),
        },
        "stdev_duration_s": {
            "run_a": timing_a.get("stdev_duration_s"),
            "run_b": timing_b.get("stdev_duration_s"),
            "delta": _sub(timing_a.get("stdev_duration_s"), timing_b.get("stdev_duration_s")),
        },
        "max_duration_s": {
            "run_a": timing_a.get("max_duration_s"),
            "run_b": timing_b.get("max_duration_s"),
            "delta": _sub(timing_a.get("max_duration_s"), timing_b.get("max_duration_s")),
        },
        "rationale_length_mean": {
            "run_a": timing_a.get("rationale_length_mean"),
            "run_b": timing_b.get("rationale_length_mean"),
            "delta": _sub(
                timing_a.get("rationale_length_mean"),
                timing_b.get("rationale_length_mean"),
            ),
        },
        "correlation_duration_vs_risk": {
            "run_a": timing_a.get("correlation_duration_vs_risk"),
            "run_b": timing_b.get("correlation_duration_vs_risk"),
        },
        "correlation_duration_vs_rationale_length": {
            "run_a": timing_a.get("correlation_duration_vs_rationale_length"),
            "run_b": timing_b.get("correlation_duration_vs_rationale_length"),
        },
        "outliers_a": timing_a.get("outliers", []),
        "outliers_b": timing_b.get("outliers", []),
    }


# ---------------------------------------------------------------------------
# Per-run summary (reusing build_phase4_summary)
# ---------------------------------------------------------------------------
def summarize_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary + percentage views for a list of phase4 records."""
    summary = build_phase4_summary(records)
    total = summary["total_records"]
    pct = _percentage_helper(total)

    final_disposition_counts = _final_disposition_counts(records)
    blocked_count = sum(
        v
        for k, v in final_disposition_counts.items()
        if k.upper().startswith(BLOCKED_DISPOSITION_PREFIXES)
    )

    return {
        "total_records": total,
        "phase1_posture_counts": dict(summary["phase1_posture_counts"]),
        "phase1_posture_percentages": {
            k: pct(v) for k, v in summary["phase1_posture_counts"].items()
        },
        "phase2_outcome_counts": dict(summary["phase2_outcome_counts"]),
        "phase2_outcome_percentages": {
            k: pct(v) for k, v in summary["phase2_outcome_counts"].items()
        },
        "phase3_output_counts": dict(summary["phase3_output_counts"]),
        "phase3_output_percentages": dict(summary["phase3_output_percentages"]),
        "violated_constraint_counts": dict(summary["violated_constraint_counts"]),
        "unresolved_constraint_counts": dict(summary["unresolved_constraint_counts"]),
        "final_disposition_counts": final_disposition_counts,
        "final_disposition_percentages": {
            k: pct(v) for k, v in final_disposition_counts.items()
        },
        "blocked_count": blocked_count,
        "blocked_pct": pct(blocked_count),
        "phase3_ambiguity_pct": sum(
            summary["phase3_output_percentages"].get(k, 0.0)
            for k in AMBIGUITY_PHASE3_OUTPUTS
        ),
        "context_tag_counts": dict(summary["context_tag_counts"]),
        "use_domain_counts": dict(summary["use_domain_counts"]),
        "drift_heuristics": dict(summary["drift_heuristics"]),
        "raw_risk_distribution": compute_risk_distribution(records),
        "timing": compute_timing_stats(records),
    }


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------
TRACKED_PHASE1_KEYS = ("PROCEED", "PAUSE", "ESCALATE")
TRACKED_PHASE3_KEYS = (
    "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
    "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
)


def _delta(a: float, b: float) -> float:
    return round(b - a, 2)


def compute_drift(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    *,
    threshold_pp: float,
    records_a: Optional[List[Dict[str, Any]]] = None,
    records_b: Optional[List[Dict[str, Any]]] = None,
    severity_encoder: Optional[_SeverityEncoder] = None,
    severity_major_threshold: float = MAJOR_SEVERITY_THRESHOLD,
) -> Dict[str, Any]:
    """Compute pp-deltas for every tracked metric and flag ones over threshold.

    When the underlying ``records_a`` and ``records_b`` lists are supplied,
    we additionally compute:

    * ``variance_rate``  — per-dimension raw-risk disagreement
    * ``severity``       — semantic distance between justifications
    * ``timing_delta``   — change in per-scenario duration distribution
    """
    metrics: List[Dict[str, Any]] = []

    def add(category: str, key: str, a_val: float, b_val: float) -> None:
        d = _delta(a_val, b_val)
        metrics.append({
            "category": category,
            "metric": key,
            "run_a_pct": a_val,
            "run_b_pct": b_val,
            "delta_pp": d,
            "exceeds_threshold": abs(d) >= threshold_pp,
        })

    p1a = summary_a["phase1_posture_percentages"]
    p1b = summary_b["phase1_posture_percentages"]
    for k in TRACKED_PHASE1_KEYS:
        add("phase1_posture", k, p1a.get(k, 0.0), p1b.get(k, 0.0))

    p3a = summary_a["phase3_output_percentages"]
    p3b = summary_b["phase3_output_percentages"]
    for k in TRACKED_PHASE3_KEYS:
        add("phase3_output", k, p3a.get(k, 0.0), p3b.get(k, 0.0))
    add(
        "phase3_output",
        "PHASE3_AMBIGUITY_TOTAL",
        summary_a.get("phase3_ambiguity_pct", 0.0),
        summary_b.get("phase3_ambiguity_pct", 0.0),
    )

    add(
        "final_gate",
        "BLOCKED",
        summary_a.get("blocked_pct", 0.0),
        summary_b.get("blocked_pct", 0.0),
    )

    drifted = [m for m in metrics if m["exceeds_threshold"]]

    result: Dict[str, Any] = {
        "threshold_pp": threshold_pp,
        "metrics": metrics,
        "drifted_metrics": drifted,
        "drift_detected": bool(drifted),
        "constraint_changes": _constraint_changes(summary_a, summary_b),
    }

    if records_a is not None and records_b is not None:
        result["variance_rate"] = compute_variance_rate(records_a, records_b)
        result["severity"] = compute_severity(
            records_a,
            records_b,
            encoder=severity_encoder,
            major_threshold=severity_major_threshold,
        )
        result["timing_delta"] = compute_timing_delta(
            summary_a.get("timing", {}),
            summary_b.get("timing", {}),
        )

    return result


def _constraint_changes(summary_a: Dict[str, Any], summary_b: Dict[str, Any]) -> Dict[str, Any]:
    def top_n(counts: Dict[str, int], n: int = 5) -> List[Tuple[str, int]]:
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def diff(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, Any]:
        all_keys = set(a) | set(b)
        deltas = {k: b.get(k, 0) - a.get(k, 0) for k in all_keys}
        appeared = sorted([k for k in all_keys if k not in a and b.get(k, 0) > 0])
        disappeared = sorted([k for k in all_keys if a.get(k, 0) > 0 and k not in b])
        biggest_growth = sorted(deltas.items(), key=lambda kv: -kv[1])[:5]
        biggest_drop = sorted(deltas.items(), key=lambda kv: kv[1])[:5]
        return {
            "deltas": deltas,
            "appeared": appeared,
            "disappeared": disappeared,
            "biggest_growth": biggest_growth,
            "biggest_drop": biggest_drop,
        }

    va = summary_a["violated_constraint_counts"]
    vb = summary_b["violated_constraint_counts"]
    ua = summary_a["unresolved_constraint_counts"]
    ub = summary_b["unresolved_constraint_counts"]

    return {
        "violated": {
            "run_a_top": top_n(va),
            "run_b_top": top_n(vb),
            **diff(va, vb),
        },
        "unresolved": {
            "run_a_top": top_n(ua),
            "run_b_top": top_n(ub),
            **diff(ua, ub),
        },
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _fmt_pct(v: float) -> str:
    return f"{v:>7.2f}%"


def _fmt_delta(v: float) -> str:
    return f"{v:+8.2f}pp"


def render_drift_report_text(
    *,
    model: str,
    run_a_meta: Dict[str, Any],
    run_b_meta: Dict[str, Any],
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    drift: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"Phase 4 drift report: {model.upper()}")
    lines.append("=" * 78)
    lines.append(
        f"Run A : {run_a_meta['run_id']:<14} "
        f"registered={run_a_meta['registered_at_utc']}  "
        f"scenarios={summary_a['total_records']}"
    )
    lines.append(
        f"Run B : {run_b_meta['run_id']:<14} "
        f"registered={run_b_meta['registered_at_utc']}  "
        f"scenarios={summary_b['total_records']}"
    )
    lines.append(f"Threshold: {drift['threshold_pp']} percentage points")
    lines.append("")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in drift["metrics"]:
        grouped.setdefault(m["category"], []).append(m)

    pretty_category = {
        "phase1_posture": "Phase 1 posture shift",
        "phase3_output": "Phase 3 output shift",
        "final_gate": "Final execution gate shift",
    }

    for category in ("phase1_posture", "phase3_output", "final_gate"):
        lines.append(pretty_category[category] + ":")
        header = (
            f"  {'Metric':<40}"
            f"  {'Run A':>9}"
            f"  {'->':^4}"
            f"  {'Run B':>9}"
            f"  {'Delta':>10}"
            f"  Flag"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for m in grouped.get(category, []):
            flag = "*** DRIFT ***" if m["exceeds_threshold"] else ""
            lines.append(
                f"  {m['metric']:<40}"
                f"  {_fmt_pct(m['run_a_pct'])}"
                f"  {'->':^4}"
                f"  {_fmt_pct(m['run_b_pct'])}"
                f"  {_fmt_delta(m['delta_pp'])}"
                f"  {flag}"
            )
        lines.append("")

    cc = drift["constraint_changes"]
    lines.append("Top violated constraints:")
    lines.append(f"  Run A: {_top_str(cc['violated']['run_a_top'])}")
    lines.append(f"  Run B: {_top_str(cc['violated']['run_b_top'])}")
    if cc["violated"]["appeared"]:
        lines.append(f"  Appeared in B: {', '.join(cc['violated']['appeared'])}")
    if cc["violated"]["disappeared"]:
        lines.append(f"  Disappeared from A: {', '.join(cc['violated']['disappeared'])}")
    lines.append("")

    lines.append("Top unresolved constraints:")
    lines.append(f"  Run A: {_top_str(cc['unresolved']['run_a_top'])}")
    lines.append(f"  Run B: {_top_str(cc['unresolved']['run_b_top'])}")
    if cc["unresolved"]["appeared"]:
        lines.append(f"  Appeared in B: {', '.join(cc['unresolved']['appeared'])}")
    if cc["unresolved"]["disappeared"]:
        lines.append(f"  Disappeared from A: {', '.join(cc['unresolved']['disappeared'])}")
    lines.append("")

    variance = drift.get("variance_rate")
    if variance is not None:
        lines.append("Variance rate (raw risk levels across paired scenarios):")
        lines.append(
            f"  Paired scenarios            : {variance['paired_scenarios']}"
        )
        lines.append(
            f"  Any dimension changed       : "
            f"{variance['scenarios_with_any_change']} "
            f"({variance['variance_rate_pct']}%)"
        )
        for dim in RISK_DIMENSIONS:
            per = variance["per_dimension"].get(dim, {})
            lines.append(
                f"    {dim:<16} changed in "
                f"{per.get('changed_count', 0)} scenario(s) "
                f"({per.get('changed_rate_pct', 0.0)}%)"
            )
        if variance["unmatched_or_missing_raw_fields"]:
            lines.append(
                f"  Missing raw fields          : "
                f"{', '.join(variance['unmatched_or_missing_raw_fields'])}"
            )
        if variance["changed_scenarios"]:
            lines.append("  Most-changed scenarios (top 5):")
            for entry in variance["changed_scenarios"][:5]:
                deltas = ", ".join(
                    f"{d}:{entry['run_a'][d]}->{entry['run_b'][d]}"
                    for d in entry["changed_dimensions"]
                )
                lines.append(
                    f"    - {entry['scenario_id']:<10} {deltas}"
                )
        lines.append("")

    severity = drift.get("severity")
    if severity is not None:
        lines.append("Severity (semantic distance between justifications):")
        lines.append(f"  Backend                     : {severity['backend']}")
        lines.append(
            f"  Paired scenarios            : {severity['paired_scenarios']}"
        )
        if severity["paired_scenarios"]:
            lines.append(
                f"  Mean / median severity      : "
                f"{severity['mean_severity']} / {severity['median_severity']}"
            )
            lines.append(
                f"  Min / max severity          : "
                f"{severity['min_severity']} / {severity['max_severity']}"
            )
            lines.append(
                f"  Major change (>{severity['major_threshold']:g})        : "
                f"{severity['major_change_count']} "
                f"({severity['major_change_pct']}%)"
            )
            if severity["major_change_scenarios"]:
                lines.append("  Major-change scenarios (top 10):")
                for entry in severity["major_change_scenarios"][:10]:
                    lines.append(
                        f"    - {entry['scenario_id']:<10} "
                        f"severity={entry['severity']:>6.2f}  "
                        f"cos={entry['cosine_similarity']:+.4f}"
                    )
            else:
                lines.append("  Major-change scenarios     : (none)")
        if severity["skipped_missing_rationale"]:
            lines.append(
                f"  Skipped (no rationale)      : "
                f"{', '.join(severity['skipped_missing_rationale'])}"
            )
        lines.append("")

    timing_delta = drift.get("timing_delta")
    if timing_delta is not None:
        lines.append("Timing delta:")
        for key, label in (
            ("mean_duration_s", "Mean duration (s)        "),
            ("stdev_duration_s", "Stdev duration (s)       "),
            ("max_duration_s", "Max duration (s)         "),
            ("rationale_length_mean", "Mean rationale (chars)   "),
        ):
            entry = timing_delta.get(key, {})
            a = entry.get("run_a")
            b = entry.get("run_b")
            d = entry.get("delta")
            lines.append(
                f"  {label}: A={a}  B={b}  delta={d if d is not None else 'n/a'}"
            )
        for key, label in (
            ("correlation_duration_vs_risk", "Corr(duration, risk)     "),
            ("correlation_duration_vs_rationale_length", "Corr(duration, rat. len) "),
        ):
            entry = timing_delta.get(key, {})
            lines.append(
                f"  {label}: A={entry.get('run_a')}  B={entry.get('run_b')}"
            )
        oa = timing_delta.get("outliers_a") or []
        ob = timing_delta.get("outliers_b") or []
        if oa or ob:
            lines.append(
                f"  Outliers (Run A {len(oa)}, Run B {len(ob)}):"
            )
            for o in oa[:5]:
                lines.append(
                    f"    A  {o['scenario_id']:<10} {o['duration_s']:>7.2f}s  z={o['z']:+.2f}"
                )
            for o in ob[:5]:
                lines.append(
                    f"    B  {o['scenario_id']:<10} {o['duration_s']:>7.2f}s  z={o['z']:+.2f}"
                )
        lines.append("")

    drifted = drift["drifted_metrics"]
    lines.append("Summary:")
    if not drifted:
        lines.append(
            f"  No metric crossed the {drift['threshold_pp']} pp threshold."
        )
    else:
        lines.append(
            f"  {len(drifted)} metric(s) crossed the {drift['threshold_pp']} pp threshold:"
        )
        for m in drifted:
            lines.append(
                f"    - {m['category']}.{m['metric']} "
                f"({_fmt_delta(m['delta_pp']).strip()})"
            )

    return "\n".join(lines)


def _top_str(top_list: List[Tuple[str, int]]) -> str:
    if not top_list:
        return "(none)"
    return ", ".join(f"{k}({v})" for k, v in top_list)


def write_drift_report(
    *,
    model: str,
    run_a_meta: Dict[str, Any],
    run_b_meta: Dict[str, Any],
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    drift: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model.lower()}_{run_a_meta['run_id']}_to_{run_b_meta['run_id']}"
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"

    text = render_drift_report_text(
        model=model,
        run_a_meta=run_a_meta,
        run_b_meta=run_b_meta,
        summary_a=summary_a,
        summary_b=summary_b,
        drift=drift,
    )
    txt_path.write_text(text, encoding="utf-8")

    payload = {
        "model": model.lower(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_a": {
            **run_a_meta,
            "summary": summary_a,
        },
        "run_b": {
            **run_b_meta,
            "summary": summary_b,
        },
        "drift": drift,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return txt_path, json_path


# ---------------------------------------------------------------------------
# Run resolution helpers
# ---------------------------------------------------------------------------
def _run_meta(records: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    """Extract simple metadata from one run's records."""
    if not records:
        return {
            "run_id": run_id,
            "registered_at_utc": "",
            "run_directory": "",
            "scenario_count": 0,
        }
    first = records[0]
    return {
        "run_id": run_id,
        "registered_at_utc": first.get("registered_at_utc", ""),
        "run_directory": first.get("run_directory", ""),
        "scenario_count": len(records),
    }


def make_auto_run_id(*, when: Optional[datetime] = None) -> str:
    """Return a unique run id suitable for automatic registration."""
    ts = when or datetime.now(UTC)
    return f"auto_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def resolve_run_directory(
    model: str,
    *,
    run_id: Optional[str] = None,
    latest: bool = False,
    history_dir: Path = HISTORY_DIR_DEFAULT,
) -> Tuple[str, Path]:
    """Look up ``run_directory`` for a registered run from per-model history.

    Exactly one of ``run_id`` or ``latest=True`` must be supplied (or pass
    ``latest=True`` implicitly when only ``model`` is given by callers that
    default to the most recent run).

    Returns ``(run_id, run_directory_path)``.
    """
    distinct = existing_run_ids(model, history_dir=history_dir)
    if not distinct:
        raise ValueError(
            f"No registered runs for model '{model}'. "
            f"Run a MOC test script first (or register manually with "
            f"phase4_drift_per_model.py --register)."
        )

    if latest or run_id is None:
        if run_id is not None and latest:
            raise ValueError("Pass only one of run_id=... or latest=True.")
        rid = distinct[-1]
    else:
        rid = run_id
        if rid not in distinct:
            raise ValueError(
                f"Run id '{rid}' is not registered for model '{model}'. "
                f"Known: {', '.join(distinct)}"
            )

    records = filter_records_for_run(load_history(model, history_dir=history_dir), rid)
    if not records:
        raise ValueError(f"No scenario records found for model '{model}' run '{rid}'.")

    run_dir_raw = records[0].get("run_directory")
    if not run_dir_raw:
        raise ValueError(
            f"Run '{rid}' for model '{model}' has no run_directory in history."
        )
    run_dir = Path(run_dir_raw)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Registered run directory not found: {run_dir}")
    return rid, run_dir


def _resolve_drift_run_ids(
    model: str,
    requested: List[str],
    *,
    history_dir: Path,
) -> Tuple[str, str]:
    """Decide which two run_ids to compare.

    * If the user passed exactly two ``--run-id`` values, use them as A,B.
    * If none were passed, auto-pick the last two registered run_ids.
    * Anything else is an error.
    """
    distinct = existing_run_ids(model, history_dir=history_dir)
    if len(requested) == 2:
        for rid in requested:
            if rid not in distinct:
                raise ValueError(
                    f"Run id '{rid}' has not been registered for model "
                    f"'{model}'. Known: {distinct or 'none'}."
                )
        return requested[0], requested[1]

    if len(requested) == 0:
        if len(distinct) < 2:
            raise ValueError(
                f"Need at least 2 registered runs for model '{model}' to "
                f"compute drift. Found: {distinct or 'none'}."
            )
        return distinct[-2], distinct[-1]

    raise ValueError(
        f"--drift / --compare expects 0 or 2 --run-id values; got {len(requested)}."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phase4_drift_per_model",
        description=(
            "Per-model run registration + drift detection. The canonical "
            "phase4_history.jsonl is never read or modified."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model label. Common values: {', '.join(KNOWN_MODELS)}.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help=(
            "Run identifier (e.g. 'v1', '50_scenarios'). Required exactly "
            "once for --register. May be supplied 0 or 2 times for --compare "
            "/ --drift; if 0, the last two registered runs are used."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to the run directory (only used by --register).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_PP,
        help=(
            "Absolute percentage-point threshold for flagging drift "
            f"(default: {DEFAULT_THRESHOLD_PP})."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIR_DEFAULT,
        help=f"Directory holding per-model JSONL history (default: {HISTORY_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DRIFT_REPORTS_DIR_DEFAULT,
        help=f"Where to write drift reports (default: {DRIFT_REPORTS_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "When used with --register, overwrite an already-registered "
            "(model, run_id) pair instead of refusing."
        ),
    )
    parser.add_argument(
        "--severity-model",
        type=str,
        default=DEFAULT_SEVERITY_MODEL,
        help=(
            "sentence-transformers model used to embed justifications for "
            f"the severity score (default: {DEFAULT_SEVERITY_MODEL}). "
            "If the package is unavailable the script falls back to a "
            "stdlib TF-cosine."
        ),
    )
    parser.add_argument(
        "--severity-threshold",
        type=float,
        default=MAJOR_SEVERITY_THRESHOLD,
        help=(
            "Severity score above which a scenario is flagged as a 'major "
            f"justification change' (default: {MAJOR_SEVERITY_THRESHOLD})."
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--register",
        action="store_true",
        help="Register a run into <history-dir>/<model>.jsonl.",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Compare two run_ids for the model and write a drift report.",
    )
    mode.add_argument(
        "--drift",
        action="store_true",
        help="Auto-compare the last two registered runs for the model.",
    )

    return parser.parse_args(argv)


def _do_register(args: argparse.Namespace) -> int:
    if len(args.run_id) != 1:
        print("ERROR: --register requires exactly one --run-id.", file=sys.stderr)
        return 2
    if args.run_dir is None:
        print("ERROR: --register requires --run-dir.", file=sys.stderr)
        return 2

    try:
        receipt = register_run(
            model=args.model,
            run_id=args.run_id[0],
            run_dir=args.run_dir,
            force=args.force,
            history_dir=args.history_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"[register] model={receipt['model']} run_id={receipt['run_id']} "
        f"appended={receipt['scenario_records_appended']}"
    )
    if receipt["scenario_files_skipped"]:
        print(
            f"[register] skipped (status != ok or missing pipeline_result): "
            f"{', '.join(receipt['scenario_files_skipped'])}"
        )
    print(f"[register] history: {receipt['history_file']}")
    if receipt["force_overwrite"]:
        print("[register] NOTE: --force was used; previous records for this run_id were dropped.")
    return 0


def _do_compare_or_drift(args: argparse.Namespace) -> int:
    try:
        run_a_id, run_b_id = _resolve_drift_run_ids(
            args.model, args.run_id, history_dir=args.history_dir
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    history = load_history(args.model, history_dir=args.history_dir)
    records_a = filter_records_for_run(history, run_a_id)
    records_b = filter_records_for_run(history, run_b_id)

    if not records_a or not records_b:
        print(
            f"ERROR: no records found for one of the runs "
            f"(A='{run_a_id}', B='{run_b_id}') in model '{args.model}'.",
            file=sys.stderr,
        )
        return 2

    records_a, _ = enrich_records_from_disk(records_a)
    records_b, _ = enrich_records_from_disk(records_b)

    summary_a = summarize_run(records_a)
    summary_b = summarize_run(records_b)
    encoder = _SeverityEncoder(args.severity_model)
    drift = compute_drift(
        summary_a,
        summary_b,
        threshold_pp=args.threshold,
        records_a=records_a,
        records_b=records_b,
        severity_encoder=encoder,
        severity_major_threshold=args.severity_threshold,
    )

    run_a_meta = _run_meta(records_a, run_a_id)
    run_b_meta = _run_meta(records_b, run_b_id)

    txt_path, json_path = write_drift_report(
        model=args.model,
        run_a_meta=run_a_meta,
        run_b_meta=run_b_meta,
        summary_a=summary_a,
        summary_b=summary_b,
        drift=drift,
        out_dir=args.reports_dir,
    )

    text = txt_path.read_text(encoding="utf-8")
    print(text)
    print()
    print(f"[report-text] {txt_path}")
    print(f"[report-json] {json_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.register:
        return _do_register(args)
    if args.compare or args.drift:
        return _do_compare_or_drift(args)

    print("ERROR: no mode selected (use --register / --compare / --drift).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
