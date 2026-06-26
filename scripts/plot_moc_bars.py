"""plot_moc_bars.py

Generate grouped bar charts for the MOC Effect comparison across providers.
For each risk dimension (uncertainty, potential_harm, irreversibility,
time_pressure), produces a chart with one x-axis group per series
(Expected, then each registered model run) and three bars per group
showing the percentage of outputs in each risk level (LOW / MEDIUM / HIGH).

Data source (default)
---------------------
    phase4_model_history/<model>.jsonl

Every registered run for each model is included by default (v1, v2, v3,
...). Use ``--include-runs latest`` for a single bar per model.

Outputs (in --out-dir, default moc_comparison_figures/):
    uncertainty_bars.png
    potential_harm_bars.png
    irreversibility_bars.png
    time_pressure_bars.png
    moc_combined_2x2.png

Usage:
    python plot_moc_bars.py
    python plot_moc_bars.py --include-runs latest
    python plot_moc_bars.py --models gpt,gemini,claude,grok --out-dir moc_comparison_figures

Requires:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase4_history_loader import (
    DEFAULT_HISTORY_DIR,
    RISK_FIELDS as ALL_FIELDS,
    intended_for_scenarios,
    load_labeled_lookups,
    union_scenario_ids_from_lookups,
)


DEFAULT_OUT_DIR = Path("moc_comparison_figures")
DEFAULT_MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude")

FIELD_TITLES: Dict[str, str] = {
    "uncertainty":     "Uncertainty",
    "potential_harm":  "Potential Harm",
    "irreversibility": "Irreversibility",
    "time_pressure":   "Time Pressure",
}

LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
LEVEL_COLORS: Dict[str, str] = {
    "LOW":    "#2ca02c",
    "MEDIUM": "#ff9f1c",
    "HIGH":   "#d62728",
}


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------
def distribution_pct(values: Iterable[Optional[str]]) -> Dict[str, float]:
    counts: Counter = Counter()
    valid = 0
    for v in values:
        if v not in LEVELS:
            continue
        counts[v] += 1
        valid += 1
    if valid == 0:
        return {lvl: 0.0 for lvl in LEVELS}
    return {lvl: 100.0 * counts.get(lvl, 0) / valid for lvl in LEVELS}


def compute_distributions(
    field: str,
    scenario_ids: Sequence[str],
    model_series: List[Tuple[str, Dict[str, dict]]],
    *,
    all_lookups: Sequence[Dict[str, dict]],
) -> Dict[str, Dict[str, float]]:
    expected_vals = intended_for_scenarios(scenario_ids, *all_lookups, field=field)
    out: Dict[str, Dict[str, float]] = {
        "Expected": distribution_pct(expected_vals),
    }
    for label, lookup in model_series:
        vals = [
            lookup.get(sid, {}).get("raw", {}).get(field)
            for sid in scenario_ids
        ]
        out[label] = distribution_pct(vals)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _fig_width(n_providers: int, *, base: float = 10.0, per_bar: float = 0.55) -> float:
    return max(base, min(base + per_bar * max(0, n_providers - 4), 24.0))


def _draw_grouped_bars(ax, distributions: Dict[str, Dict[str, float]], title: str) -> None:
    providers = list(distributions.keys())
    n_providers = len(providers)
    n_levels = len(LEVELS)

    bar_width = 0.25
    cluster_centers = np.arange(n_providers, dtype=float)
    offsets = (np.arange(n_levels) - (n_levels - 1) / 2.0) * bar_width

    for i, level in enumerate(LEVELS):
        heights = [distributions[p][level] for p in providers]
        positions = cluster_centers + offsets[i]
        bars = ax.bar(
            positions,
            heights,
            width=bar_width,
            color=LEVEL_COLORS[level],
            edgecolor="#333333",
            linewidth=0.6,
            label=level,
        )
        for rect, h in zip(bars, heights):
            x = rect.get_x() + rect.get_width() / 2.0
            label = f"{h:.1f}%"
            if h >= 8.0:
                ax.text(
                    x, h + 1.2, label,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#222222",
                )
            else:
                ax.text(
                    x, h + 0.6, label,
                    ha="center", va="bottom",
                    fontsize=7.5, color="#222222",
                )

    ax.set_xticks(cluster_centers)
    labels = providers
    if len(labels) > 6:
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    else:
        ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Percentage of outputs", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9, title="Risk level")


def plot_field_bars(
    field: str,
    distributions: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    n = len(distributions)
    fig, ax = plt.subplots(figsize=(_fig_width(n), 6))
    title = (
        f"MOC Effect: {FIELD_TITLES[field]} \u2014 "
        "raw model outputs vs intended"
    )
    _draw_grouped_bars(ax, distributions, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_2x2(
    field_distributions: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
    *,
    n_series: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(_fig_width(n_series, base=18), 11))
    flat_axes = axes.flatten()
    for ax, field in zip(flat_axes, ALL_FIELDS):
        if field not in field_distributions:
            ax.set_visible(False)
            continue
        _draw_grouped_bars(ax, field_distributions[field], FIELD_TITLES[field])

    fig.suptitle(
        "MOC Effect: Risk distribution by provider (raw outputs vs intended)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    field_distributions: Dict[str, Dict[str, Dict[str, float]]],
    n_scenarios: int,
) -> None:
    print()
    print("=" * 74)
    print(f"BAR-CHART INPUT SUMMARY  ({n_scenarios} scenarios)")
    print("=" * 74)
    for field in ALL_FIELDS:
        if field not in field_distributions:
            continue
        print(f"\n[{FIELD_TITLES[field].upper()}]")
        print("-" * 56)
        print(f"  {'source':<22}  {'LOW':>7}  {'MEDIUM':>7}  {'HIGH':>7}")
        for provider in field_distributions[field]:
            d = field_distributions[field][provider]
            print(
                f"  {provider:<22}  "
                f"{d['LOW']:>6.1f}%  "
                f"{d['MEDIUM']:>6.1f}%  "
                f"{d['HIGH']:>6.1f}%"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot grouped bar charts comparing MOC Effect across registered "
            "model runs (reads phase4_model_history/*.jsonl)."
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
        print(
            "ERROR: no registered runs found. Register runs with "
            "phase4_drift_per_model.py --register first.",
            file=sys.stderr,
        )
        return 2

    all_lookups = [lookup for _, lookup in model_series]
    scenario_ids = union_scenario_ids_from_lookups(*all_lookups)
    print(f"[info] scenarios: {len(scenario_ids)} unique IDs")
    print(f"[info] series ({len(model_series)}):")
    for label, lookup in model_series:
        print(f"         {label:<24}  n={len(lookup)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    field_distributions: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in ALL_FIELDS:
        dists = compute_distributions(
            field,
            scenario_ids,
            model_series,
            all_lookups=all_lookups,
        )
        field_distributions[field] = dists
        out_path = args.out_dir / f"{field}_bars.png"
        plot_field_bars(field, dists, out_path)
        print(f"[ok]   wrote {out_path}")

    combined_path = args.out_dir / "moc_combined_2x2.png"
    plot_combined_2x2(
        field_distributions,
        combined_path,
        n_series=1 + len(model_series),
    )
    print(f"[ok]   wrote {combined_path}")

    print_summary(field_distributions, n_scenarios=len(scenario_ids))
    print()
    print(f"Done. Figures in: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
