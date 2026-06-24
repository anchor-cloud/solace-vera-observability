"""make_moc013_deepdive.py

Per-model MOC-013 deep-dive tables — same row layout as
``blog_figures/fig_moc013_runs_comparison.png`` (``make_moc013_runs_comparison.py``),
but one model per image and three run rows (v1, v2, v3) with full justification text.

Usage:
    python make_moc013_deepdive.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_moc013_comparison import (
    MODEL_COLORS,
    MODEL_LABELS,
    _wrap,
    clean_text,
    draw_score_row,
)
from make_moc013_model_tables import (
    MODELS,
    RUN_SLOTS,
    SCENARIO_ID,
    load_combined,
    load_moc013,
    resolve_run_entry,
    reviewer_phrase,
)

OUT_DIR = Path("model_moc013_tables")
DPI = 300
FIG_WIDTH = 16.0

# Same column boundaries as make_moc013_runs_comparison.py
COL_MODEL = (0.03, 0.11)
COL_RUN = (0.11, 0.155)
COL_SCORE = (0.155, 0.345)
COL_JUST = (0.345, 0.665)
COL_VIEW = (0.665, 0.97)

JUST_WRAP_W = 60
REVIEW_WRAP_W = 50

RUN_NUMBERS = ("1", "2", "3")


def _line_count(text: str, width: int) -> int:
    if not text:
        return 1
    wrapped = textwrap.fill(
        text, width=width, break_long_words=False, break_on_hyphens=False
    )
    return max(1, len(wrapped.splitlines()))


def build_run_rows(
    combined: Dict[str, Any],
) -> Tuple[str, Dict[str, str], List[Optional[Dict[str, Any]]]]:
    """Return (prompt, intended, [slot per run or None])."""
    prompt = ""
    intended: Dict[str, str] = {}
    rows: List[Optional[Dict[str, Any]]] = []

    for _header, slot_key in RUN_SLOTS:
        entry = resolve_run_entry(combined, slot_key)
        if entry is None:
            rows.append(None)
            continue
        loaded = load_moc013(str(entry.get("run_directory") or ""))
        if loaded is None:
            rows.append(None)
            continue
        inc, raw, rationale, pr = loaded
        if pr and not prompt:
            prompt = pr
        if inc and not intended:
            intended = inc
        rows.append({
            "raw": raw,
            "justification": clean_text(rationale).strip(),
            "reviewer": reviewer_phrase(inc, raw),
        })

    return prompt, intended, rows


def _row_height_units(row: Optional[Dict[str, Any]]) -> float:
    if row is None:
        return 4.0
    just_lines = _line_count(row["justification"], JUST_WRAP_W)
    rev_lines = _line_count(row["reviewer"], REVIEW_WRAP_W)
    return 3.5 + 1.12 * just_lines + 0.35 * rev_lines


def _estimate_figure_height(rows: List[Optional[Dict[str, Any]]]) -> float:
    units = sum(_row_height_units(r) for r in rows)
    return 7.5 + 0.095 * units


def render_deepdive(model: str) -> Path:
    combined = load_combined(model)
    prompt, intended, rows = build_run_rows(combined)
    label = MODEL_LABELS[model]
    n_rows = len(rows)

    fig_h = _estimate_figure_height(rows)
    fig = plt.figure(figsize=(FIG_WIDTH, fig_h), facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----- Title -----
    ax.text(
        0.5, 0.978,
        f"{label}: {SCENARIO_ID} — full justification across three runs",
        ha="center", va="center", fontsize=20, weight="bold",
    )
    ax.text(
        0.5, 0.958,
        f"Scenario {SCENARIO_ID} — {label} v1, v2, and v3 side by side so a "
        "reviewer can spot how risk scores and justifications shift between runs",
        ha="center", va="center", fontsize=12, color="#444444", style="italic",
    )

    # ----- Scenario block -----
    scen_top, scen_bot = 0.945, 0.860
    ax.add_patch(Rectangle(
        (0.03, scen_bot), 0.94, scen_top - scen_bot,
        transform=ax.transAxes, facecolor="#f5f7fa",
        edgecolor="#cfd6df", lw=0.8,
    ))
    ax.text(0.045, scen_top - 0.013, "Scenario prompt",
            fontsize=11, weight="bold", color="#222222")
    ax.text(0.045, scen_top - 0.028, clean_text(prompt),
            fontsize=12, color="#1a1a1a")
    ax.text(0.045, scen_top - 0.050,
            "Intended risk profile (CSV ground truth)",
            fontsize=11, weight="bold", color="#222222")

    intended_chip_top = scen_top - 0.055
    intended_chip_bottom = intended_chip_top - 0.026
    draw_score_row(
        ax,
        top_y=intended_chip_top,
        bottom_y=intended_chip_bottom,
        x_left=0.045,
        x_right=0.30,
        scores=intended,
    )

    legend_x = 0.34
    ax.text(legend_x, scen_top - 0.050, "Chip key",
            fontsize=11, weight="bold", color="#333333", va="top")
    ax.text(legend_x, scen_top - 0.066,
            "Letters:  U = uncertainty   H = potential_harm   "
            "I = irreversibility   T = time_pressure",
            fontsize=10, color="#444444", va="top")
    ax.text(legend_x, scen_top - 0.080,
            "Colors:   green = LOW    amber = MEDIUM    red = HIGH",
            fontsize=10, color="#444444", va="top")

    # ----- Header + table body (runs as rows) -----
    header_y_top = 0.845
    header_y_bot = 0.820
    body_top = header_y_bot
    body_bottom_min = 0.085

    units = [_row_height_units(r) for r in rows]
    total_units = sum(units) or 1.0
    body_span = body_top - body_bottom_min

    col_model_x0, col_model_x1 = COL_MODEL
    col_run_x0, col_run_x1 = COL_RUN
    col_score_x0, col_score_x1 = COL_SCORE
    col_rat_x0, col_rat_x1 = COL_JUST
    col_view_x0, col_view_x1 = COL_VIEW

    # Header row
    ax.add_patch(Rectangle(
        (col_model_x0, header_y_bot),
        col_view_x1 - col_model_x0, header_y_top - header_y_bot,
        transform=ax.transAxes, facecolor="#222222",
        edgecolor="white", lw=0.5,
    ))
    header_y = (header_y_top + header_y_bot) / 2.0
    for x0, x1, hdr in (
        (col_model_x0, col_model_x1, "Model"),
        (col_run_x0, col_run_x1, "Run"),
        (col_score_x0, col_score_x1, "Risk scores (U/H/I/T)"),
        (col_rat_x0, col_rat_x1, "Full justification"),
        (col_view_x0, col_view_x1, "What a reviewer would see"),
    ):
        ax.text((x0 + x1) / 2.0, header_y, hdr,
                ha="center", va="center",
                fontsize=11.5, weight="bold", color="white")

    # Model cell spans all three run rows
    model_top = body_top
    model_bot = body_bottom_min
    ax.add_patch(Rectangle(
        (col_model_x0, model_bot),
        col_model_x1 - col_model_x0, model_top - model_bot,
        transform=ax.transAxes,
        facecolor=MODEL_COLORS[model], edgecolor="white", lw=1.0,
    ))
    ax.text(
        (col_model_x0 + col_model_x1) / 2.0,
        (model_top + model_bot) / 2.0,
        label,
        ha="center", va="center",
        fontsize=15, weight="bold", color="white",
    )

    y_cursor = body_top
    for i, row in enumerate(rows):
        row_h = body_span * (units[i] / total_units)
        row_top = y_cursor
        row_bot = row_top - row_h
        y_cursor = row_bot
        is_odd = i % 2 == 0
        run_num = RUN_NUMBERS[i]

        # Data-area background (alternating like reference v1/v2)
        ax.add_patch(Rectangle(
            (col_run_x0, row_bot),
            col_view_x1 - col_run_x0, row_h,
            transform=ax.transAxes,
            facecolor="#ffffff" if is_odd else "#fafbfc",
            edgecolor="#e3e6ea", lw=0.5,
        ))

        # Run cell
        ax.add_patch(Rectangle(
            (col_run_x0, row_bot), col_run_x1 - col_run_x0, row_h,
            transform=ax.transAxes,
            facecolor="#eef1f5" if is_odd else "#e3e8ef",
            edgecolor="#cfd6df", lw=0.5,
        ))
        ax.text(
            (col_run_x0 + col_run_x1) / 2.0,
            (row_top + row_bot) / 2.0,
            run_num,
            ha="center", va="center",
            fontsize=14, weight="bold", color="#333333",
        )

        if row is None:
            ax.text(
                col_score_x0 + 0.008, (row_top + row_bot) / 2.0,
                "—", ha="left", va="center", fontsize=10, color="#888888",
            )
            continue

        # Score chips
        chip_top = (row_top + row_bot) / 2.0 + row_h * 0.21
        chip_bottom = (row_top + row_bot) / 2.0 - row_h * 0.21
        draw_score_row(
            ax,
            top_y=chip_top,
            bottom_y=chip_bottom,
            x_left=col_score_x0 + 0.008,
            x_right=col_score_x1 - 0.008,
            scores=row["raw"],
        )

        wrapped_just = _wrap(row["justification"], width=JUST_WRAP_W)
        ax.text(
            col_rat_x0 + 0.008, row_top - 0.008,
            f"\u201c{wrapped_just}\u201d",
            ha="left", va="top",
            fontsize=9.5, color="#1a1a1a", style="italic",
            linespacing=1.25,
        )

        wrapped_view = _wrap(row["reviewer"], width=REVIEW_WRAP_W)
        ax.text(
            col_view_x0 + 0.008, row_top - 0.008,
            wrapped_view,
            ha="left", va="top",
            fontsize=10.0, color="#1a1a1a",
            linespacing=1.30,
        )

        if i < n_rows - 1:
            ax.plot(
                [col_model_x0, col_view_x1],
                [row_bot, row_bot],
                transform=ax.transAxes,
                color="#a8b0bb", lw=1.2,
            )

    # ----- Caption strip -----
    cap_top, cap_bot = 0.072, 0.020
    ax.add_patch(Rectangle(
        (0.03, cap_bot), 0.94, cap_top - cap_bot,
        transform=ax.transAxes, facecolor="#eef1f5",
        edgecolor="#cfd6df", lw=0.8,
    ))
    ax.text(
        0.5, (cap_top + cap_bot) / 2.0,
        "Same scenario, same model, three runs. The pipeline records each run's "
        "risk scores and full justifications, making run-to-run shifts visible. "
        "A human reviewing the pipeline's output would see these differences "
        "before approving any action.",
        ha="center", va="center", fontsize=11.5, color="#1a1a1a", style="italic",
        wrap=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model}_moc013_deepdive.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    return path


def main() -> int:
    written: List[Path] = []
    for model in MODELS:
        try:
            path = render_deepdive(model)
        except FileNotFoundError as exc:
            print(f"WARNING [{model}]: {exc}", file=sys.stderr)
            continue
        written.append(path)
        print(f"wrote: {path}")

    if not written:
        print("ERROR: no deep-dive images written.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
