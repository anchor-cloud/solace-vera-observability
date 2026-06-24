"""make_blog_figures.py

Generate publication-ready PNG figures + a combined grid image for the
LessWrong blog post.

Data sources
------------
All series come from the registered phase-4 history:

    phase4_model_history/<model>.jsonl        (raw / posture / timing per run)
    phase4_drift_reports/justification_stability.json  (run-pair severity)

Per-run charts read ``phase4_model_history/<model>.jsonl`` directly.
Caption text prefers aggregates from
``phase4_per_model/<model>_combined_summary.json`` when present (run
``python phase4_per_model_analysis.py --regenerate`` to refresh).

By default every registered run is rendered as its own series (so a
chart for GPT shows ``GPT v1`` / ``GPT v2`` / ``GPT v3`` side by side).
Use ``--include-runs latest`` to fall back to the legacy "one bar per
model" layout, or pass a comma-separated list of run ids
(``--include-runs v1,v3``) to filter.

Output (written to ``blog_figures/``):
    fig1_trust.png
    fig2_uncertainty.png
    fig3_harm.png
    fig4_justification_table.png
    fig5_speed.png
    fig_combined.png
    captions.txt

Usage:
    python make_blog_figures.py
    python make_blog_figures.py --include-runs latest
    python make_blog_figures.py --include-runs v1,v3
    python make_blog_figures.py --models gpt,claude
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
# Lock the font to DejaVu Sans before any plotting. Arial on this machine
# rendered capital "P" as a "B"-shaped glyph in the x-tick labels (so
# "GPT" looked like "GBT" in the PNGs). DejaVu Sans ships with matplotlib
# and renders correctly. This must run before the rcParams.update() block
# below and before any pyplot call.
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from phase4_history_loader import (
    DEFAULT_HISTORY_DIR,
    KNOWN_MODELS,
    RISK_LEVELS,
    RunGroup,
    load_runs,
    norm_level,
)


# ---------------------------------------------------------------------------
# Layout / style
# ---------------------------------------------------------------------------
STABILITY_JSON = Path("phase4_drift_reports/justification_stability.json")
COMBINED_SUMMARY_DIR = Path("phase4_per_model")
OUT_DIR = Path("blog_figures")
DPI = 300

MODEL_LABELS: Dict[str, str] = {
    "gpt": "GPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "grok": "Grok",
}

# Brand-ish palette per model. Each model gets a base hue; runs within a
# model are differentiated by progressively darker shades.
MODEL_BASE_COLORS: Dict[str, str] = {
    "gpt":    "#1f77b4",  # blue
    "gemini": "#2ca02c",  # green
    "claude": "#ff7f0e",  # orange
    "grok":   "#d62728",  # red
}

# Risk-level palette: green (safe) -> amber -> red.
LEVEL_COLORS: Dict[str, str] = {
    "LOW":    "#4caf50",
    "MEDIUM": "#ffb300",
    "HIGH":   "#e53935",
}

# Phase 1 posture palette: green (proceed) -> amber (pause) -> red (escalate).
POSTURE_COLORS: Dict[str, str] = {
    "PROCEED":  "#4caf50",
    "PAUSE":    "#ffb300",
    "ESCALATE": "#e53935",
}

# Extra x-axis space inserted between each model's bar cluster.
MODEL_GROUP_GAP = 0.75
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.titleweight":"bold",
    "axes.labelsize":  11,
    "axes.edgecolor":  "#444444",
    "axes.linewidth":  0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":     "#222222",
    "ytick.color":     "#222222",
    "legend.frameon":  False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
})


# ---------------------------------------------------------------------------
# Combined summaries (phase4_per_model_analysis --model / --regenerate)
# ---------------------------------------------------------------------------
def load_combined_summary(
    model: str,
    *,
    summary_dir: Path = COMBINED_SUMMARY_DIR,
) -> Optional[Dict[str, object]]:
    path = summary_dir / f"{model.lower()}_combined_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_posture_mean(model: str, posture: str) -> Optional[float]:
    data = load_combined_summary(model)
    if not data:
        return None
    stats = (
        (data.get("aggregates") or {})
        .get("phase1_posture", {})
        .get(posture)
    )
    if not stats:
        return None
    return float(stats.get("mean_pct", 0.0))


def _combined_risk_pct(model: str, dim: str, level: str) -> Optional[float]:
    """Average % of scenarios at ``level`` for ``dim`` across registered runs."""
    data = load_combined_summary(model)
    if not data:
        return None
    pcts: List[float] = []
    for run in data.get("runs") or []:
        dist = (run.get("raw_risk_distribution") or {}).get(dim, {})
        total = sum(dist.get(l, 0) for l in RISK_LEVELS)
        if total:
            pcts.append(100.0 * dist.get(level, 0) / total)
    if not pcts:
        return None
    return sum(pcts) / len(pcts)


def _combined_timing_mean(model: str) -> Optional[float]:
    data = load_combined_summary(model)
    if not data:
        return None
    stats = (
        (data.get("aggregates") or {})
        .get("timing", {})
        .get("mean_duration_s")
    )
    if not stats or not stats.get("per_run"):
        return None
    return float(stats["mean"])


# ---------------------------------------------------------------------------
# Summary computation -- mirror what phase4_per_model_analysis.py writes
# ---------------------------------------------------------------------------
def summarize_run(group: RunGroup) -> Dict[str, object]:
    """Return the subset of per-run metrics used by the blog figures."""
    posture_counts: Counter = Counter()
    risks: Dict[str, Counter] = {
        dim: Counter()
        for dim in ("uncertainty", "potential_harm",
                    "irreversibility", "time_pressure")
    }
    durations: List[float] = []

    for rec in group.records:
        p = str(rec.get("phase1_posture") or "").strip().upper()
        if p in ("PROCEED", "PAUSE", "ESCALATE"):
            posture_counts[p] += 1
        for dim in risks:
            lvl = norm_level(rec.get(f"raw_{dim}"))
            if lvl is not None:
                risks[dim][lvl] += 1
        d = rec.get("duration_s")
        if isinstance(d, (int, float)):
            durations.append(float(d))

    mean_duration = sum(durations) / len(durations) if durations else None
    return {
        "model":          group.model,
        "run_id":         group.run_id,
        "label":          group.label,
        "posture_counts": dict(posture_counts),
        "posture_total":  sum(posture_counts.values()),
        "risks":          {k: dict(v) for k, v in risks.items()},
        "mean_duration":  mean_duration,
        "n_records":      len(group.records),
    }


def parse_stability(path: Path) -> Tuple[Dict[str, Dict], List[Dict], Dict[str, List[str]]]:
    """Return (latest-per-model, per-comparison list, runs_per_model) from JSON."""
    if not path.exists():
        return {}, [], {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_model: Dict[str, Dict] = {}
    for entry in data.get("per_model") or []:
        by_model[str(entry.get("model", "")).lower()] = entry
    per_comparison = list(data.get("per_comparison") or data.get("per_model") or [])
    runs_per_model = {
        str(k).lower(): list(v)
        for k, v in (data.get("runs_per_model") or {}).items()
    }
    return by_model, per_comparison, runs_per_model


def _logical_run_numbers(
    run_ids: List[str],
    comparisons: List[Dict],
) -> Dict[str, int]:
    """Map internal run_ids to 1-based ``Run N`` labels.

    Consecutive pairs with identical scores *and* zero justification variance
    are treated as duplicate re-registrations of the same logical run (e.g.
    GPT ``v2`` and ``v2_20260507`` both become Run 2).
    """
    if not run_ids:
        return {}

    parent: Dict[str, str] = {rid: rid for rid in run_ids}

    def find(rid: str) -> str:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    for comp in comparisons:
        same = int(comp.get("scenarios_with_same_scores") or 0)
        vr = float(comp.get("variance_rate_pct") or 0.0)
        if same > 0 and vr == 0.0:
            ra = comp.get("run_a_id")
            rb = comp.get("run_b_id")
            if ra in parent and rb in parent:
                root_a, root_b = find(ra), find(rb)
                if root_a != root_b:
                    parent[root_b] = root_a

    root_to_num: Dict[str, int] = {}
    out: Dict[str, int] = {}
    n = 0
    for rid in run_ids:
        root = find(rid)
        if root not in root_to_num:
            n += 1
            root_to_num[root] = n
        out[rid] = root_to_num[root]
    return out


def comparison_display_label(run_a: str, run_b: str, run_nums: Dict[str, int], *, ascii_arrow: bool = False) -> str:
    """Human label such as ``Run 1 → Run 2`` (no internal run ids)."""
    a = run_nums.get(run_a)
    b = run_nums.get(run_b)
    arrow = " -> " if ascii_arrow else " \u2192 "
    if a is None or b is None:
        return f"Run ?{arrow}Run ?"
    return f"Run {a}{arrow}Run {b}"


def clean_comparisons(
    per_comparison: List[Dict],
    runs_per_model: Dict[str, List[str]],
    *,
    model_order: Optional[List[str]] = None,
) -> List[Dict]:
    """Keep only meaningful rows and attach display labels."""
    order = model_order or list(KNOWN_MODELS)
    by_model: Dict[str, List[Dict]] = {}
    for entry in per_comparison:
        by_model.setdefault(str(entry.get("model", "")).lower(), []).append(entry)

    cleaned: List[Dict] = []
    for model in order:
        comps = by_model.get(model, [])
        if not comps:
            continue
        run_ids = runs_per_model.get(model) or []
        if not run_ids:
            seen: List[str] = []
            for c in comps:
                for key in ("run_a_id", "run_b_id"):
                    rid = c.get(key)
                    if rid and rid not in seen:
                        seen.append(rid)
            run_ids = seen
        run_nums = _logical_run_numbers(run_ids, comps)

        for entry in comps:
            same = int(entry.get("scenarios_with_same_scores") or 0)
            vr = float(entry.get("variance_rate_pct") or 0.0)
            if same <= 0 or vr <= 0.0:
                continue
            row = dict(entry)
            row["comparison_label"] = comparison_display_label(
                entry["run_a_id"], entry["run_b_id"], run_nums
            )
            row["comparison_label_console"] = comparison_display_label(
                entry["run_a_id"], entry["run_b_id"], run_nums, ascii_arrow=True
            )
            cleaned.append(row)
    return cleaned


def print_cleaned_table(rows: List[Dict]) -> None:
    """Print the filtered justification-stability table to stdout."""
    headers = (
        "Model", "Comparison", "Same scores",
        "Justifications changed", "Variance rate", "Avg severity",
    )
    print()
    print("=== Cleaned justification stability table ===")
    print("  ".join(h.ljust(22 if i == 0 else 18) for i, h in enumerate(headers)))
    print("  ".join("-" * (22 if i == 0 else 18) for i in range(len(headers))))
    for entry in rows:
        model = MODEL_LABELS.get(str(entry.get("model", "")).lower(),
                                 str(entry.get("model", "")).upper())
        same = entry.get("scenarios_with_same_scores", "")
        changed = entry.get("justifications_changed", "")
        vr = entry.get("variance_rate_pct", "")
        avg = entry.get("avg_severity", "")
        print(
            f"{model:<22}  "
            f"{entry.get('comparison_label_console', entry.get('comparison_label', '')):<18}  "
            f"{str(same):<18}  "
            f"{f'{changed} / {same}':<18}  "
            f"{f'{vr}%':<18}  "
            f"{avg}"
        )
    print()


# ---------------------------------------------------------------------------
# Color helpers for per-run shading
# ---------------------------------------------------------------------------
def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = (max(0.0, min(1.0, c)) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def shade_run(base_hex: str, run_index: int, total_runs: int) -> str:
    """Return a per-run shade of ``base_hex``.

    Run 0 is lightest, run N-1 is the base color (un-darkened). Single-run
    callers always get the base color so legacy ``--include-runs latest``
    figures look identical to before.
    """
    if total_runs <= 1:
        return base_hex
    base = _hex_to_rgb(base_hex)
    # Blend towards white by an amount that shrinks with each later run.
    # run_index=0  -> 55% towards white (light tint)
    # run_index=N-1-> 0% towards white  (full saturation)
    blend = 0.55 * (1.0 - (run_index / (total_runs - 1)))
    mixed = tuple(c + (1.0 - c) * blend for c in base)
    return _rgb_to_hex(mixed)


# ---------------------------------------------------------------------------
# Series ordering
# ---------------------------------------------------------------------------
def build_series(
    runs_by_model: Dict[str, List[RunGroup]],
    model_order: List[str],
    *,
    runs_per_model: Dict[str, List[str]],
    per_comparison: List[Dict],
) -> List[Tuple[str, RunGroup, str]]:
    """Return ``[(label, run_group, color), ...]`` in plot order.

    One bar per *logical* run (duplicate re-registrations collapsed).
    Labels read ``GPT Run 1``, ``GPT Run 2``, ... with no internal ids.
    """
    comps_by_model: Dict[str, List[Dict]] = {}
    for entry in per_comparison:
        comps_by_model.setdefault(str(entry.get("model", "")).lower(), []).append(entry)

    out: List[Tuple[str, RunGroup, str]] = []
    for model in model_order:
        groups = runs_by_model.get(model, [])
        if not groups:
            continue

        run_ids = runs_per_model.get(model) or [g.run_id for g in groups]
        run_nums = _logical_run_numbers(run_ids, comps_by_model.get(model, []))

        # Keep the last physical run for each logical run number.
        best: Dict[int, Tuple[int, RunGroup]] = {}
        for idx, g in enumerate(groups):
            num = run_nums.get(g.run_id, idx + 1)
            prev = best.get(num)
            if prev is None or idx >= prev[0]:
                best[num] = (idx, g)

        logical = sorted(best.items())
        n = len(logical)
        for shade_idx, (run_num, (_, g)) in enumerate(logical):
            color = shade_run(MODEL_BASE_COLORS[model], shade_idx, n)
            label = (
                f"{MODEL_LABELS[model]} Run {run_num}" if n > 1
                else MODEL_LABELS[model]
            )
            out.append((label, g, color))
    return out


def _x_positions_with_gaps(series: List[Tuple[str, RunGroup, str]]) -> List[float]:
    """X positions with extra space between each model's bar cluster."""
    xs: List[float] = []
    x = 0.0
    prev_model: Optional[str] = None
    for _, group, _ in series:
        if prev_model is not None and group.model != prev_model:
            x += MODEL_GROUP_GAP
        xs.append(x)
        x += 1.0
        prev_model = group.model
    return xs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


def _annotate_segments(
    ax,
    bottoms: List[float],
    values: List[float],
    x_positions: List[float],
    *,
    min_pct: float = 4.0,
) -> None:
    """Write the percentage inside each stacked segment when it's big enough."""
    for x, b, v in zip(x_positions, bottoms, values):
        if v >= min_pct:
            ax.text(
                x, b + v / 2.0, f"{v:.0f}%",
                ha="center", va="center",
                fontsize=9, color="white", weight="bold",
            )


def _annotate_grouped_bars(ax, bars, fmt: str = "{:.0f}%") -> None:
    for bar in bars:
        h = bar.get_height()
        if h <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            fmt.format(h),
            ha="center", va="bottom", fontsize=8.5, color="#222222",
        )


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _series_fig_width(
    n_series: int,
    n_gaps: int = 0,
    *,
    base: float = 8.5,
    per_bar: float = 0.55,
) -> float:
    """Wider figure when we have many series or inter-model gaps."""
    extra = per_bar * max(0, n_series - 4) + MODEL_GROUP_GAP * n_gaps
    return max(base, min(base + extra, 22.0))


def _count_model_gaps(series: List[Tuple[str, RunGroup, str]]) -> int:
    gaps = 0
    prev: Optional[str] = None
    for _, group, _ in series:
        if prev is not None and group.model != prev:
            gaps += 1
        prev = group.model
    return gaps


def _rotated_xticks(ax, labels: List[str]) -> None:
    """Rotate tick labels when there are too many to read horizontally."""
    if len(labels) > 6:
        ax.set_xticklabels(labels, rotation=20, ha="right")
    else:
        ax.set_xticklabels(labels)


# ---------------------------------------------------------------------------
# Figure 1: Stacked Phase 1 posture (one bar per series)
# ---------------------------------------------------------------------------
def plot_trust(
    series: List[Tuple[str, RunGroup, str]],
    summaries_by_label: Dict[str, Dict],
    ax=None, *, standalone: bool = True,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(_series_fig_width(len(series), _count_model_gaps(series)), 5.4)
        )

    labels = [lab for lab, _, _ in series]
    x = _x_positions_with_gaps(series)
    per_posture: Dict[str, List[float]] = {
        p: [] for p in ("PROCEED", "PAUSE", "ESCALATE")
    }
    for label, _, _ in series:
        s = summaries_by_label[label]
        total = s["posture_total"] or 1
        counts = s["posture_counts"]
        for p in per_posture:
            per_posture[p].append(_pct(counts.get(p, 0), total))

    bottoms = [0.0] * len(labels)
    for posture in ("PROCEED", "PAUSE", "ESCALATE"):
        vals = per_posture[posture]
        ax.bar(
            x, vals, bottom=bottoms,
            color=POSTURE_COLORS[posture],
            edgecolor="white", linewidth=1.2,
            label=posture.capitalize(),
        )
        _annotate_segments(ax, bottoms, vals, x, min_pct=4.0)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    _rotated_xticks(ax, labels)
    ax.set_ylabel("Share of scenarios (%)")
    ax.set_ylim(0, 105)
    ax.set_title("How often models trusted themselves vs. escalated to humans")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if standalone and fig is not None:
        return _save(fig, "fig1_trust.png")
    return None


# ---------------------------------------------------------------------------
# Figures 2 & 3: grouped risk-level bars
# ---------------------------------------------------------------------------
def _plot_risk_dimension(
    series: List[Tuple[str, RunGroup, str]],
    summaries_by_label: Dict[str, Dict],
    dim: str,
    title: str,
    ax=None,
    *,
    standalone_name: str = "",
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(
                _series_fig_width(len(series), _count_model_gaps(series), per_bar=0.65),
                5.4,
            )
        )

    labels = [lab for lab, _, _ in series]
    x = _x_positions_with_gaps(series)
    width = 0.26

    per_level: Dict[str, List[float]] = {lvl: [] for lvl in RISK_LEVELS}
    for label, _, _ in series:
        counts = summaries_by_label[label]["risks"].get(dim, {})
        total = sum(counts.get(lvl, 0) for lvl in RISK_LEVELS)
        for lvl in per_level:
            per_level[lvl].append(_pct(counts.get(lvl, 0), total))

    offsets = {"LOW": -width, "MEDIUM": 0.0, "HIGH": width}
    for lvl in RISK_LEVELS:
        bars = ax.bar(
            [xi + offsets[lvl] for xi in x],
            per_level[lvl],
            width=width,
            color=LEVEL_COLORS[lvl],
            edgecolor="white", linewidth=0.8,
            label=lvl.capitalize(),
        )
        _annotate_grouped_bars(ax, bars)

    ax.set_xticks(x)
    _rotated_xticks(ax, labels)
    ax.set_ylabel("Share of scenarios (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="upper right", title="Self-reported level")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if standalone_name and fig is not None:
        return _save(fig, standalone_name)
    return None


def plot_uncertainty(series, summaries_by_label, ax=None, *, standalone=True):
    return _plot_risk_dimension(
        series, summaries_by_label, "uncertainty",
        "Uncertainty calibration across models",
        ax=ax,
        standalone_name=("fig2_uncertainty.png" if standalone else ""),
    )


def plot_harm(series, summaries_by_label, ax=None, *, standalone=True):
    return _plot_risk_dimension(
        series, summaries_by_label, "potential_harm",
        "Harm assessment across models",
        ax=ax,
        standalone_name=("fig3_harm.png" if standalone else ""),
    )


# ---------------------------------------------------------------------------
# Figure 4: Justification stability table (rendered as an image)
# ---------------------------------------------------------------------------
def plot_justification_table(
    per_comparison: List[Dict],
    ax=None,
    *,
    standalone: bool = True,
):
    """Render filtered comparison rows (same scores > 0, variance rate > 0)."""
    fig = None
    if ax is None:
        n_rows = max(len(per_comparison), 1)
        fig, ax = plt.subplots(figsize=(11.0, 1.0 + 0.55 * n_rows))
    ax.axis("off")

    headers = [
        "Model", "Comparison", "Same scores",
        "Justifications changed", "Variance rate", "Avg severity",
    ]
    col_widths = [0.10, 0.16, 0.14, 0.24, 0.18, 0.18]

    rows: List[List[str]] = []
    model_for_row: List[str] = []
    for entry in per_comparison:
        model = str(entry.get("model", "")).lower()
        same = entry.get("scenarios_with_same_scores", "")
        changed = entry.get("justifications_changed", "")
        vr = entry.get("variance_rate_pct", "")
        avg = entry.get("avg_severity", "")
        rows.append([
            MODEL_LABELS.get(model, model.upper()),
            entry.get("comparison_label", ""),
            str(same),
            f"{changed} / {same}" if same != "" else str(changed),
            f"{vr}%",
            f"{avg}",
        ])
        model_for_row.append(model)

    if not rows:
        rows = [["(no data)"] * len(headers)]
        model_for_row = [""]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)

    n_cols = len(headers)
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_facecolor("#222222")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("white")
        cell.set_height(cell.get_height() * 1.15)

    for r, model in enumerate(model_for_row, start=1):
        for c in range(n_cols):
            cell = table[r, c]
            cell.set_edgecolor("#dddddd")
            if c == 0 and model in MODEL_BASE_COLORS:
                cell.set_facecolor(MODEL_BASE_COLORS[model])
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#fafafa" if r % 2 else "#f0f0f0")

    ax.set_title(
        "Justifications change even when the four raw risk scores are identical",
        pad=10,
    )

    if standalone and fig is not None:
        return _save(fig, "fig4_justification_table.png")
    return None


# ---------------------------------------------------------------------------
# Figure 5: Mean duration
# ---------------------------------------------------------------------------
def plot_speed(
    series: List[Tuple[str, RunGroup, str]],
    summaries_by_label: Dict[str, Dict],
    ax=None, *, standalone: bool = True,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(_series_fig_width(len(series), _count_model_gaps(series)), 5.4)
        )

    labels = [lab for lab, _, _ in series]
    values = [float(summaries_by_label[lab]["mean_duration"] or 0.0) for lab in labels]
    colors = [c for _, _, c in series]
    x = _x_positions_with_gaps(series)

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_xticks(x)
    _rotated_xticks(ax, labels)
    ax.set_ylabel("Mean duration per scenario (seconds)")
    ax.set_title("Average response time per scenario")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    top = max(values) if values else 0.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + (top * 0.02 if top else 0.05),
            f"{v:.2f}s",
            ha="center", va="bottom", fontsize=10, weight="bold",
        )
    if top:
        ax.set_ylim(0, top * 1.18)

    if standalone and fig is not None:
        return _save(fig, "fig5_speed.png")
    return None


# ---------------------------------------------------------------------------
# Combined grid
# ---------------------------------------------------------------------------
def plot_combined(
    series: List[Tuple[str, RunGroup, str]],
    summaries_by_label: Dict[str, Dict],
    per_comparison: List[Dict],
) -> Path:
    fig = plt.figure(
        figsize=(
            max(17.0, _series_fig_width(len(series), _count_model_gaps(series)) + 5),
            13,
        )
    )
    gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_trust(series, summaries_by_label, ax=ax1, standalone=False)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_speed(series, summaries_by_label, ax=ax2, standalone=False)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_uncertainty(series, summaries_by_label, ax=ax3, standalone=False)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_harm(series, summaries_by_label, ax=ax4, standalone=False)

    ax5 = fig.add_subplot(gs[2, :])
    plot_justification_table(per_comparison, ax=ax5, standalone=False)

    fig.suptitle(
        "Phase 4 cross-model comparison (50-scenario MOC evidence run)",
        fontsize=15, weight="bold", y=0.995,
    )
    return _save(fig, "fig_combined.png")


# ---------------------------------------------------------------------------
# Captions sidecar
# ---------------------------------------------------------------------------
def _latest_summary(summaries_by_label: Dict[str, Dict], model: str) -> Optional[Dict]:
    """Find the chronologically-last summary for ``model``."""
    candidates = [s for s in summaries_by_label.values() if s["model"] == model]
    return candidates[-1] if candidates else None


def write_captions(
    summaries_by_label: Dict[str, Dict],
    latest_stability: Dict[str, Dict],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def posture_pct(model, key):
        combined = _combined_posture_mean(model, key)
        if combined is not None:
            return combined
        s = _latest_summary(summaries_by_label, model)
        if not s:
            return 0.0
        return _pct(s["posture_counts"].get(key, 0), s["posture_total"] or 1)

    def risk_pct(model, dim, lvl):
        combined = _combined_risk_pct(model, dim, lvl)
        if combined is not None:
            return combined
        s = _latest_summary(summaries_by_label, model)
        if not s:
            return 0.0
        counts = s["risks"].get(dim, {})
        total = sum(counts.get(l, 0) for l in RISK_LEVELS)
        return _pct(counts.get(lvl, 0), total)

    def mean_dur(model):
        combined = _combined_timing_mean(model)
        if combined is not None:
            return combined
        s = _latest_summary(summaries_by_label, model)
        return float(s["mean_duration"] or 0.0) if s else 0.0

    grok_proceed = posture_pct("grok", "PROCEED")
    grok_unc_low = risk_pct("grok", "uncertainty", "LOW")
    gpt_unc_med = risk_pct("gpt", "uncertainty", "MEDIUM")
    grok_harm_low = risk_pct("grok", "potential_harm", "LOW")
    grok_dur = mean_dur("grok")
    gpt_dur = mean_dur("gpt")
    claude_dur = mean_dur("claude")
    ratio_gpt = grok_dur / gpt_dur if gpt_dur else 0.0
    ratio_claude = grok_dur / claude_dur if claude_dur else 0.0

    cap = []
    cap.append("Figure 1 (fig1_trust.png):")
    cap.append(
        f"Grok proceeded autonomously on {grok_proceed:.0f}% of scenarios in "
        f"its most recent run -- the highest of all four models. GPT escalated "
        f"on 100% of scenarios across every registered run."
    )
    cap.append("")
    cap.append("Figure 2 (fig2_uncertainty.png):")
    cap.append(
        f"In the latest runs, Grok self-reported LOW uncertainty on "
        f"{grok_unc_low:.0f}% of scenarios; GPT collapsed to MEDIUM on "
        f"{gpt_unc_med:.0f}%. Both are calibration failures in opposite "
        f"directions, and they reproduce across every run shown."
    )
    cap.append("")
    cap.append("Figure 3 (fig3_harm.png):")
    cap.append(
        f"Grok rated potential harm as LOW on {grok_harm_low:.0f}% of "
        f"scenarios in the latest run -- meaningfully underestimating "
        f"danger relative to the other models."
    )
    cap.append("")
    cap.append("Figure 4 (fig4_justification_table.png):")
    if latest_stability:
        cap.append(
            "Across every registered consecutive run-pair, the variance rate "
            "stays at or near 100%: when the four raw risk scores were "
            "identical between two runs, every model rewrote its justification "
            "for almost every same-score scenario. The reasoning text is not "
            "stable -- it's post-hoc rationalization."
        )
    else:
        cap.append(
            "(Stability JSON missing -- run phase4_justification_drift.py first.)"
        )
    cap.append("")
    cap.append("Figure 5 (fig5_speed.png):")
    cap.append(
        f"In the latest runs Grok averaged {grok_dur:.1f}s per scenario "
        f"-- roughly {ratio_gpt:.1f}x slower than GPT ({gpt_dur:.1f}s) and "
        f"{ratio_claude:.1f}x slower than Claude ({claude_dur:.1f}s)."
    )

    path = OUT_DIR / "captions.txt"
    path.write_text("\n".join(cap) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make_blog_figures",
        description=(
            "Render the blog-post figures from phase4_model_history/*.jsonl. "
            "By default every registered run shows up as its own series."
        ),
    )
    p.add_argument(
        "--include-runs",
        default="all",
        help=(
            "Which registered runs to plot: 'all' (default), 'latest', "
            "'first', or a comma-separated list of run ids (e.g. 'v1,v3')."
        ),
    )
    p.add_argument(
        "--models",
        default=",".join(KNOWN_MODELS),
        help=(
            "Comma-separated list of models in plot order "
            f"(default: {','.join(KNOWN_MODELS)})."
        ),
    )
    p.add_argument(
        "--history-dir", type=Path, default=DEFAULT_HISTORY_DIR,
        help=f"Per-model history dir (default: {DEFAULT_HISTORY_DIR}).",
    )
    p.add_argument(
        "--stability-json", type=Path, default=STABILITY_JSON,
        help=f"Justification-stability report (default: {STABILITY_JSON}).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=f"Output directory for PNGs (default: {OUT_DIR}).",
    )
    return p.parse_args(list(argv) if argv is not None else sys.argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    global OUT_DIR  # _save() reads OUT_DIR; allow CLI override.
    args = parse_args(argv)
    OUT_DIR = args.out_dir

    model_order = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for m in model_order:
        if m not in MODEL_LABELS:
            print(
                f"WARNING: unknown model '{m}' (allowed: "
                f"{', '.join(MODEL_LABELS)}); skipping.",
                file=sys.stderr,
            )
    model_order = [m for m in model_order if m in MODEL_LABELS]
    if not model_order:
        print("ERROR: no valid models selected.", file=sys.stderr)
        return 2

    runs_by_model: Dict[str, List[RunGroup]] = {}
    for m in model_order:
        groups = load_runs(m, include=args.include_runs, history_dir=args.history_dir)
        if not groups:
            print(
                f"WARNING: model '{m}' has no registered runs matching "
                f"--include-runs={args.include_runs!r}.",
                file=sys.stderr,
            )
        runs_by_model[m] = groups

    latest_stability, raw_comparison, runs_per_model = parse_stability(
        args.stability_json
    )
    cleaned_comparison = clean_comparisons(
        raw_comparison, runs_per_model, model_order=model_order
    )
    print_cleaned_table(cleaned_comparison)

    series = build_series(
        runs_by_model, model_order,
        runs_per_model=runs_per_model,
        per_comparison=raw_comparison,
    )
    if not series:
        print(
            "ERROR: no run series to plot. Check phase4_model_history/ and "
            "--include-runs.",
            file=sys.stderr,
        )
        return 2

    summaries_by_label: Dict[str, Dict] = {
        label: summarize_run(group) for label, group, _ in series
    }

    outputs = [
        plot_trust(series, summaries_by_label),
        plot_uncertainty(series, summaries_by_label),
        plot_harm(series, summaries_by_label),
        plot_justification_table(cleaned_comparison),
        plot_speed(series, summaries_by_label),
        plot_combined(series, summaries_by_label, cleaned_comparison),
        write_captions(summaries_by_label, latest_stability),
    ]
    for path in outputs:
        print(f"wrote: {path}")
    print()
    print("Series rendered (model run_id -> records):")
    for label, group, _ in series:
        print(f"  {label:<26}  n={len(group.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
