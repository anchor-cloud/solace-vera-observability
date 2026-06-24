"""plot_moc_comparison.py

Generate per-field comparison line plots for the MOC Effect across registered
model runs. For each scenario, plots each model run's raw risk-level output
against the human-intended risk level on a 0=LOW / 1=MEDIUM / 2=HIGH scale.

Data source (default)
---------------------
    phase4_model_history/<model>.jsonl

Every registered run is included by default (v1, v2, v3, ...).

Outputs:
    moc_comparison_figures/<field>.png   (one PNG per field, 300 DPI)

Usage:
    python plot_moc_comparison.py
    python plot_moc_comparison.py --include-runs latest
    python plot_moc_comparison.py --models gpt,gemini,claude,grok

Requires:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase4_history_loader import (
    DEFAULT_HISTORY_DIR,
    RISK_FIELDS as ALL_FIELDS,
    intended_for_scenarios,
    load_labeled_lookups,
    union_scenario_ids_from_lookups,
)


DEFAULT_OUT_DIR = Path("moc_comparison_figures")
DEFAULT_MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude")

LEVEL_TO_NUM: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
NUM_TO_LEVEL: Dict[int, str] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Base hues per model; line styles cycle for multiple runs of the same model.
MODEL_COLORS: Dict[str, str] = {
    "gpt":    "#d62728",
    "gemini": "#2ca02c",
    "claude": "#1f77b4",
    "grok":   "#9467bd",
}
RUN_LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":")


def _series_for(
    scenario_ids: Sequence[str],
    lookup: Dict[str, dict],
    field: str,
) -> List[float]:
    out: List[float] = []
    for sid in scenario_ids:
        rec = lookup.get(sid)
        if rec is None:
            out.append(math.nan)
            continue
        v = rec.get("raw", {}).get(field)
        if v is None or v not in LEVEL_TO_NUM:
            out.append(math.nan)
        else:
            out.append(float(LEVEL_TO_NUM[v]))
    return out


def _expected_series(
    scenario_ids: Sequence[str],
    all_lookups: Sequence[Dict[str, dict]],
    field: str,
) -> List[float]:
    vals = intended_for_scenarios(scenario_ids, *all_lookups, field=field)
    return [
        float(LEVEL_TO_NUM[v]) if v in LEVEL_TO_NUM else math.nan
        for v in vals
    ]


def _style_for_label(label: str, run_index_within_model: int) -> Dict[str, object]:
    """Pick color from model prefix and linestyle from run index."""
    model_key = label.split()[0].lower()
    color = MODEL_COLORS.get(model_key, "#555555")
    ls = RUN_LINESTYLES[run_index_within_model % len(RUN_LINESTYLES)]
    markers = {"gpt": "s", "gemini": "^", "claude": "D", "grok": "v"}
    marker = markers.get(model_key, "o")
    return {"color": color, "linestyle": ls, "marker": marker, "linewidth": 1.4}


def plot_field(
    field: str,
    scenario_ids: Sequence[str],
    expected: List[float],
    model_series: List[Tuple[str, Dict[str, dict]]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = list(range(len(scenario_ids)))

    ax.plot(
        x, expected,
        label="Expected",
        color="black", linestyle="--", marker="o",
        markersize=4, linewidth=1.6, alpha=0.9,
    )

    # Track run index per model for linestyle cycling.
    run_idx: Dict[str, int] = {}
    for label, lookup in model_series:
        model_key = label.split()[0].lower()
        idx = run_idx.get(model_key, 0)
        run_idx[model_key] = idx + 1
        series = _series_for(scenario_ids, lookup, field)
        style = _style_for_label(label, idx)
        ax.plot(x, series, label=label, markersize=4, alpha=0.75, **style)  # type: ignore[arg-type]

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["LOW", "MEDIUM", "HIGH"])
    ax.set_ylim(-0.3, 2.3)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ids, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(scenario_ids) - 0.5)
    ax.set_xlabel("Scenario ID", fontsize=11)
    ax.set_ylabel("Risk level", fontsize=11)
    ax.set_title(
        f"MOC Effect comparison \u2014 raw '{field}' vs intended",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "  0.0%"
    return f"{100.0 * numerator / denominator:5.1f}%"


def _distribution_row(values: List[float]) -> Tuple[int, Dict[str, int]]:
    counts = Counter()
    valid = 0
    for v in values:
        if isinstance(v, float) and math.isnan(v):
            counts["MISSING"] += 1
            continue
        valid += 1
        counts[NUM_TO_LEVEL[int(v)]] += 1
    return valid, counts


def print_summary(
    fields: Sequence[str],
    scenario_ids: Sequence[str],
    expected_by_field: Dict[str, List[float]],
    series_by_label: Dict[str, Dict[str, List[float]]],
) -> None:
    print()
    print("=" * 78)
    print(f"MOC PROVIDER COMPARISON   ({len(scenario_ids)} scenarios)")
    print("=" * 78)

    for field in fields:
        print()
        print(f"[{field.upper()}]")
        print("-" * 60)
        exp_vals = expected_by_field[field]
        print(
            f"  {'source':<22} {'n':>4}  {'LOW':>7} {'MEDIUM':>7} {'HIGH':>7}  {'missing':>8}"
        )
        valid, counts = _distribution_row(exp_vals)
        print(
            f"  {'Expected':<22} {valid:>4}  "
            f"{_pct(counts.get('LOW', 0), valid):>7} "
            f"{_pct(counts.get('MEDIUM', 0), valid):>7} "
            f"{_pct(counts.get('HIGH', 0), valid):>7}  "
            f"{counts.get('MISSING', 0):>8}"
        )
        for source_label in series_by_label:
            series = series_by_label[source_label][field]
            valid, counts = _distribution_row(series)
            print(
                f"  {source_label:<22} {valid:>4}  "
                f"{_pct(counts.get('LOW', 0), valid):>7} "
                f"{_pct(counts.get('MEDIUM', 0), valid):>7} "
                f"{_pct(counts.get('HIGH', 0), valid):>7}  "
                f"{counts.get('MISSING', 0):>8}"
            )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-field MOC comparison from phase4_model_history/*.jsonl."
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
    p.add_argument(
        "--fields",
        type=lambda s: tuple(x.strip().lower() for x in s.split(",") if x.strip()),
        default=ALL_FIELDS,
        help=f"Comma-separated fields (default: {','.join(ALL_FIELDS)}).",
    )
    return p.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    bad = [f for f in args.fields if f not in ALL_FIELDS]
    if bad:
        print(f"ERROR: unknown field(s) {bad}. Allowed: {ALL_FIELDS}", file=sys.stderr)
        return 1

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    print(f"[info] models       : {','.join(models)}")
    print(f"[info] include-runs : {args.include_runs}")
    print(f"[info] history-dir  : {args.history_dir}")
    print(f"[info] out-dir      : {args.out_dir}")
    print(f"[info] fields       : {','.join(args.fields)}")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_by_field: Dict[str, List[float]] = {}
    series_by_label: Dict[str, Dict[str, List[float]]] = {}

    for field in args.fields:
        expected = _expected_series(scenario_ids, all_lookups, field)
        expected_by_field[field] = expected
        for label, lookup in model_series:
            series_by_label.setdefault(label, {})[field] = _series_for(
                scenario_ids, lookup, field
            )

        out_path = args.out_dir / f"{field}.png"
        plot_field(field, scenario_ids, expected, model_series, out_path)
        print(f"[ok]   wrote {out_path}")

    print_summary(args.fields, scenario_ids, expected_by_field, series_by_label)
    print()
    print(f"Done. Figures in: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
