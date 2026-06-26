"""plot_pipeline_outcomes.py

Generate ``moc_comparison_figures/pipeline_outcome_comparison.png``: a
grouped bar chart comparing FINAL pipeline disposition (PROCEED / BLOCKED)
across the intended baseline and every registered model run.

Data source (default)
---------------------
    phase4_model_history/<model>.jsonl

Every registered run is included by default (v1, v2, v3, ...).

Mapping:
    execution_allowed = true / final_disposition = EXECUTION_ALLOWED -> PROCEED
    final_disposition = BLOCKED_BY_PHASE*                                  -> BLOCKED

The "Expected" baseline is derived from csv_intended.expected_phase1 +
expected_phase3 the same way the live pipeline derives the actual
disposition.

Usage:
    python plot_pipeline_outcomes.py
    python plot_pipeline_outcomes.py --include-runs latest
    python plot_pipeline_outcomes.py --models gpt,gemini,claude,grok

Requires:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase4_history_loader import (
    DEFAULT_HISTORY_DIR,
    expected_dispositions_for_scenarios,
    load_labeled_lookups,
    union_scenario_ids_from_lookups,
)


DEFAULT_OUT_DIR = Path("moc_comparison_figures")
DEFAULT_OUT_NAME = "pipeline_outcome_comparison.png"
DEFAULT_MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude")

DISPOSITIONS: Tuple[str, ...] = ("PROCEED", "BLOCKED")
DISPOSITION_COLORS: Dict[str, str] = {
    "PROCEED": "#2ca02c",
    "BLOCKED": "#d62728",
}


def distribution_pct(values: Iterable[Optional[str]]) -> Dict[str, float]:
    counts: Counter = Counter()
    valid = 0
    for v in values:
        if v not in DISPOSITIONS:
            continue
        counts[v] += 1
        valid += 1
    if valid == 0:
        return {d: 0.0 for d in DISPOSITIONS}
    return {d: 100.0 * counts.get(d, 0) / valid for d in DISPOSITIONS}


def compute_distributions(
    scenario_ids: List[str],
    model_series: List[Tuple[str, Dict[str, dict]]],
    *,
    all_lookups: List[Dict[str, dict]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int], Dict[str, Counter]]:
    expected_vals = expected_dispositions_for_scenarios(
        scenario_ids, *all_lookups
    )

    distributions: Dict[str, Dict[str, float]] = {
        "Expected": distribution_pct(expected_vals),
    }
    valid_counts: Dict[str, int] = {
        "Expected": sum(1 for v in expected_vals if v in DISPOSITIONS),
    }
    raw_counts: Dict[str, Counter] = {
        "Expected": Counter(v for v in expected_vals if v),
    }

    for label, lookup in model_series:
        vals = [lookup.get(sid, {}).get("actual") for sid in scenario_ids]
        raws = [lookup.get(sid, {}).get("actual_raw") for sid in scenario_ids]
        distributions[label] = distribution_pct(vals)
        valid_counts[label] = sum(1 for v in vals if v in DISPOSITIONS)
        raw_counts[label] = Counter(r for r in raws if r)

    return distributions, valid_counts, raw_counts


def _fig_width(n_providers: int, *, base: float = 11.5, per_bar: float = 0.55) -> float:
    return max(base, min(base + per_bar * max(0, n_providers - 4), 26.0))


def plot_pipeline_outcomes(
    distributions: Dict[str, Dict[str, float]],
    valid_counts: Dict[str, int],
    n_scenarios: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(_fig_width(len(distributions)), 6.8))

    providers = list(distributions.keys())
    n_providers = len(providers)
    n_disp = len(DISPOSITIONS)

    bar_width = 0.35
    cluster_centers = np.arange(n_providers, dtype=float)
    offsets = (np.arange(n_disp) - (n_disp - 1) / 2.0) * bar_width

    for i, disposition in enumerate(DISPOSITIONS):
        heights = [distributions[p][disposition] for p in providers]
        positions = cluster_centers + offsets[i]
        bars = ax.bar(
            positions,
            heights,
            width=bar_width,
            color=DISPOSITION_COLORS[disposition],
            edgecolor="#333333",
            linewidth=0.6,
            label=disposition,
        )
        for rect, h in zip(bars, heights):
            x = rect.get_x() + rect.get_width() / 2.0
            label = f"{h:.1f}%"
            if h >= 8.0:
                ax.text(
                    x, h + 1.4, label,
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#222222",
                )
            else:
                ax.text(
                    x, h + 0.6, label,
                    ha="center", va="bottom",
                    fontsize=8, color="#222222",
                )

    sublabels = [f"{p}\n(n={valid_counts[p]})" for p in providers]
    ax.set_xticks(cluster_centers)
    if len(sublabels) > 6:
        ax.set_xticklabels(sublabels, rotation=25, ha="right", fontsize=9)
    else:
        ax.set_xticklabels(sublabels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Percentage of scenarios", fontsize=11)
    ax.set_xlabel("Source", fontsize=11)
    ax.set_title(
        "Pipeline outcome by provider \u2014 final execution gate",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.text(
        0.99, 0.985,
        "PROCEED = execution_allowed=true; BLOCKED = any BLOCKED_BY_PHASE*.   "
        f"Expected from csv_intended.   |   {n_scenarios} scenarios",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.0, color="#555555",
        style="italic",
    )

    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10, title="Final disposition")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    distributions: Dict[str, Dict[str, float]],
    valid_counts: Dict[str, int],
    raw_counts: Dict[str, Counter],
    n_scenarios: int,
) -> None:
    print()
    print("=" * 78)
    print(f"FINAL EXECUTION GATE DISTRIBUTION  ({n_scenarios} scenarios)")
    print("=" * 78)
    print(f"  {'source':<22}  {'n':>4}  {'PROCEED':>8}  {'BLOCKED':>8}")
    for provider in distributions:
        d = distributions[provider]
        n = valid_counts[provider]
        print(
            f"  {provider:<22}  {n:>4}  "
            f"{d['PROCEED']:>7.1f}%  "
            f"{d['BLOCKED']:>7.1f}%"
        )

    print()
    print("[Final disposition breakdown by phase (audit trail)]")
    print("-" * 78)
    keys = (
        "EXECUTION_ALLOWED",
        "BLOCKED_BY_PHASE1_POSTURE",
        "BLOCKED_BY_PHASE2_PAUSE",
        "BLOCKED_BY_PHASE3_AMBIGUITY",
        "BLOCKED_BY_PHASE3_FAIL",
    )
    headers = ("ALLOW", "P1_POSTURE", "P2_PAUSE", "P3_AMBIGUITY", "P3_FAIL")
    print(
        f"  {'source':<22}" +
        "".join(f"{h:>14}" for h in headers)
    )
    for provider in distributions:
        c = raw_counts.get(provider, Counter())
        row = f"  {provider:<22}"
        for k in keys:
            row += f"{c.get(k, 0):>14}"
        print(row)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot final-execution-gate comparison from "
            "phase4_model_history/*.jsonl."
        ),
    )
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models (default: {','.join(DEFAULT_MODELS)}).",
    )
    p.add_argument(
        "--include-runs",
        default="all",
        help="Which runs to plot: all (default), latest, first, or v1,v2,...",
    )
    p.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=f"Per-model history dir (default: {DEFAULT_HISTORY_DIR}).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out-name", type=str, default=DEFAULT_OUT_NAME)
    return p.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    print(f"[info] models       : {','.join(models)}")
    print(f"[info] include-runs : {args.include_runs}")
    print(f"[info] history-dir  : {args.history_dir}")
    print(f"[info] out-dir      : {args.out_dir}")
    print()

    model_series = load_labeled_lookups(
        models,
        include=args.include_runs,
        history_dir=args.history_dir,
    )
    if not model_series:
        print("ERROR: no registered runs found.", file=sys.stderr)
        return 2

    all_lookups = [lookup for _, lookup in model_series]
    scenario_ids = union_scenario_ids_from_lookups(*all_lookups)
    print(f"[info] scenarios: {len(scenario_ids)} unique IDs")
    print(f"[info] series ({len(model_series)}):")
    for label, lookup in model_series:
        print(f"         {label:<24}  n={len(lookup)}")

    distributions, valid_counts, raw_counts = compute_distributions(
        scenario_ids, model_series, all_lookups=all_lookups
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    plot_pipeline_outcomes(distributions, valid_counts, len(scenario_ids), out_path)
    print(f"[ok]   wrote {out_path}")

    print_summary(distributions, valid_counts, raw_counts, n_scenarios=len(scenario_ids))
    print()
    print(f"Done. Figure at: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
