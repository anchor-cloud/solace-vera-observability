"""make_moc013_model_tables.py

Clean per-model tables for MOC-013 across three runs (v1, v2, v3).

Data: ``<run_directory>/MOC-013.json`` from combined-summary registration.
Missing runs → blank cells.

Outputs (``model_moc013_tables/``):
    <model>_moc013_3runs.png
    all_models_moc013_3runs.png   (four tables stacked vertically)

Usage:
    python make_moc013_model_tables.py
    python make_moc013_model_tables.py --stacked-only
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})
import matplotlib.pyplot as plt

from make_moc013_comparison import clean_text, load_record

COMBINED_DIR = Path("phase4_per_model")
OUT_DIR = Path("model_moc013_tables")
SCENARIO_ID = "MOC-013"
JUSTIFICATION_MAX = 120
DPI = 300

# Layout: wide label column + three equal data columns
COL_WIDTHS = (0.30, 0.233, 0.233, 0.233)
DATA_WRAP_CHARS = 50
REVIEWER_WRAP_CHARS = 44

MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude", "grok")
MODEL_LABELS: Dict[str, str] = {
    "gpt": "GPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "grok": "Grok",
}
MODEL_COLORS: Dict[str, str] = {
    "gpt": "#1f77b4",
    "gemini": "#2ca02c",
    "claude": "#ff7f0e",
    "grok": "#d62728",
}

RUN_SLOTS: Tuple[Tuple[str, str], ...] = (
    ("Run 1 (v1)", "v1"),
    ("Run 2 (v2)", "v2"),
    ("Run 3 (v3)", "v3"),
)

ROW_LABELS = (
    "Risk scores (U/H/I/T)",
    "Justification excerpt",
    "What a reviewer would see",
)

LEVEL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
DIM_LETTERS = (
    ("uncertainty", "U"),
    ("potential_harm", "H"),
    ("irreversibility", "I"),
    ("time_pressure", "T"),
)


def load_combined(model: str) -> Dict[str, Any]:
    path = COMBINED_DIR / f"{model}_combined_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_entry(
    combined: Dict[str, Any], slot_key: str
) -> Optional[Dict[str, Any]]:
    by_id = {
        str(r.get("run_id")): r
        for r in (combined.get("runs") or [])
        if r.get("run_id")
    }
    if slot_key in ("v1", "v2"):
        return by_id.get(slot_key)
    if slot_key == "v3":
        for rid in reversed(combined.get("run_ids") or []):
            if str(rid).startswith("v3"):
                return by_id.get(str(rid))
    return None


def load_moc013(run_directory: str) -> Optional[
    Tuple[Dict[str, str], Dict[str, str], str, str]
]:
    path = Path(run_directory) / f"{SCENARIO_ID}.json"
    if not path.exists():
        return None
    return load_record(path)


def format_uhit(raw: Dict[str, str]) -> str:
    parts = []
    for dim, _ in DIM_LETTERS:
        v = (raw.get(dim) or "").strip().upper()
        parts.append({"LOW": "L", "MEDIUM": "M", "HIGH": "H"}.get(v, "?"))
    return "/".join(parts)


def format_intended_line(intended: Dict[str, str]) -> str:
    parts = []
    for dim, letter in DIM_LETTERS:
        v = (intended.get(dim) or "").strip().upper()
        parts.append(f"{letter}={v}" if v else f"{letter}=?")
    return ", ".join(parts)


def excerpt_justification(text: str, max_len: int = JUSTIFICATION_MAX) -> str:
    """First ``max_len`` characters exactly from JSON (no mid-word chop)."""
    text = clean_text(text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def wrap_cell(text: str, width: int) -> str:
    if not text:
        return ""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def reviewer_phrase(intended: Dict[str, str], raw: Dict[str, str]) -> str:
    i_harm = (intended.get("potential_harm") or "").upper()
    r_harm = (raw.get("potential_harm") or "").upper()
    i_u = (intended.get("uncertainty") or "").upper()
    r_u = (raw.get("uncertainty") or "").upper()

    if all((raw.get(dim) or "").upper() == "LOW" for dim, _ in DIM_LETTERS):
        return "Rated all LOW; wrong for this scenario"

    if LEVEL_RANK.get(r_harm, 1) >= LEVEL_RANK.get(i_harm, 1) and i_harm == "HIGH":
        return "Correctly flagged high harm"
    if LEVEL_RANK.get(r_harm, 1) < LEVEL_RANK.get(i_harm, 1):
        return "Underestimated harm"
    if LEVEL_RANK.get(r_u, 1) < LEVEL_RANK.get(i_u, 1):
        return "Downgraded uncertainty"

    collapsed = sum(
        1
        for dim, _ in DIM_LETTERS
        if LEVEL_RANK.get((raw.get(dim) or "").upper(), 1)
        < LEVEL_RANK.get((intended.get(dim) or "").upper(), 1)
    )
    if collapsed >= 2:
        return "Collapsed to MEDIUM"
    if collapsed == 1:
        return "One risk below intended"
    return "Mostly aligned with intended"


def build_table_data(
    combined: Dict[str, Any],
) -> Tuple[str, Dict[str, str], List[Tuple[str, Optional[Dict[str, str]]]]]:
    prompt = ""
    intended: Dict[str, str] = {}
    columns: List[Tuple[str, Optional[Dict[str, str]]]] = []

    for header, slot_key in RUN_SLOTS:
        entry = resolve_run_entry(combined, slot_key)
        if entry is None:
            columns.append((header, None))
            continue
        loaded = load_moc013(str(entry.get("run_directory") or ""))
        if loaded is None:
            columns.append((header, None))
            continue
        inc, raw, rationale, pr = loaded
        if pr and not prompt:
            prompt = pr
        if inc and not intended:
            intended = inc
        columns.append((
            header,
            {
                "scores": format_uhit(raw),
                "justification": excerpt_justification(rationale),
                "reviewer": reviewer_phrase(inc, raw),
            },
        ))

    return prompt, intended, columns


def _cell_matrix(
    columns: List[Tuple[str, Optional[Dict[str, str]]]],
) -> List[List[str]]:
    headers = [h for h, _ in columns]
    keys = ("scores", "justification", "reviewer")

    matrix: List[List[str]] = [[""] + headers]
    for name, key in zip(ROW_LABELS, keys):
        row = [wrap_cell(name, width=26)]
        for _, cell in columns:
            if cell is None:
                row.append("")
                continue
            val = cell.get(key, "")
            if key == "justification" and val:
                wrapped = wrap_cell(val, width=DATA_WRAP_CHARS)
                val = f"\u201c{wrapped}\u201d"
            elif key == "reviewer":
                val = wrap_cell(val, width=REVIEWER_WRAP_CHARS)
            row.append(val)
        matrix.append(row)
    return matrix


def _style_table(table, matrix: List[List[str]]) -> None:
    nrows = len(matrix)
    ncols = len(matrix[0])

    for col in range(ncols):
        cell = table[(0, col)]
        cell.set_facecolor("#37474f")
        cell.set_text_props(weight="bold", color="white", ha="center", va="center")
        cell.set_height(0.085)

    for row in range(1, nrows):
        table[(row, 0)].set_facecolor("#eceff1")
        table[(row, 0)].set_text_props(weight="bold", ha="left", va="top", fontsize=9.5)
        line_counts = [matrix[row][0].count("\n") + 1]
        for col in range(1, ncols):
            line_counts.append(matrix[row][col].count("\n") + 1)
        max_lines = max(line_counts)
        row_h = 0.065 + 0.028 * max_lines
        if row == 2:
            row_h = max(row_h, 0.14)
        elif row == 3:
            row_h = max(row_h, 0.20)

        for col in range(ncols):
            c = table[(row, col)]
            c.set_height(row_h)
            c.PAD = 0.06
            c.set_edgecolor("#90a4ae")
            c.set_linewidth(0.8)
            if col > 0:
                c.set_text_props(ha="left", va="top", fontsize=9.5)

    for row in range(nrows):
        for col in range(ncols):
            table[(row, col)].set_edgecolor("#90a4ae")
            table[(row, col)].set_linewidth(0.8)


def render_model_table(
    model: str,
    *,
    ax: plt.Axes,
) -> None:
    combined = load_combined(model)
    prompt, intended, columns = build_table_data(combined)
    label = MODEL_LABELS[model]
    matrix = _cell_matrix(columns)

    ax.axis("off")
    color = MODEL_COLORS[model]

    ax.text(
        0.5, 0.98, f"{label} — {SCENARIO_ID}",
        ha="center", va="top", fontsize=13, weight="bold", color=color,
        transform=ax.transAxes,
    )
    ax.text(
        0.03, 0.90,
        wrap_cell(clean_text(prompt) or "(prompt not found)", width=100),
        ha="left", va="top", fontsize=9, color="#333333", transform=ax.transAxes,
    )
    ax.text(
        0.03, 0.80,
        f"Intended risk: {format_intended_line(intended)}",
        ha="left", va="top", fontsize=9.5, weight="bold", color="#1a1a1a",
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=matrix,
        loc="upper center",
        cellLoc="left",
        colWidths=list(COL_WIDTHS),
        bbox=[0.01, 0.02, 0.98, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    _style_table(table, matrix)


def render_single_model(model: str) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")
    render_model_table(model, ax=ax)
    fig.tight_layout()
    path = OUT_DIR / f"{model}_moc013_3runs.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_stacked(models: List[str]) -> Path:
    """Stacked comparison with generous vertical space per model."""
    n = len(models)
    fig_h = 6.2 * n + 0.6
    fig = plt.figure(figsize=(13, fig_h), facecolor="white")

    gap = 0.012
    panel_h = (1.0 - 0.04 - gap * (n - 1)) / n
    top_start = 0.97

    for i, model in enumerate(models):
        bottom = top_start - (i + 1) * panel_h - i * gap
        ax = fig.add_axes([0.04, bottom, 0.92, panel_h])
        render_model_table(model, ax=ax)
        if i < n - 1:
            line_y = bottom - gap / 2
            fig.add_artist(
                plt.Line2D(
                    [0.04, 0.96], [line_y, line_y],
                    transform=fig.transFigure, color="#cfd8dc", linewidth=1.2,
                )
            )

    fig.suptitle(
        f"{SCENARIO_ID}: three runs per model (v1, v2, v3)",
        fontsize=14, weight="bold", y=0.995,
    )
    path = OUT_DIR / "all_models_moc013_3runs.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stacked-only",
        action="store_true",
        help="Only regenerate all_models_moc013_3runs.png",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_models: List[str] = []

    for model in MODELS:
        try:
            load_combined(model)
        except FileNotFoundError as exc:
            print(f"WARNING [{model}]: {exc}", file=sys.stderr)
            continue
        ok_models.append(model)

    if not ok_models:
        print("ERROR: no combined summaries found.", file=sys.stderr)
        return 2

    if not args.stacked_only:
        for model in ok_models:
            path = render_single_model(model)
            print(f"wrote: {path}")

    stacked = render_stacked(ok_models)
    print(f"wrote: {stacked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
