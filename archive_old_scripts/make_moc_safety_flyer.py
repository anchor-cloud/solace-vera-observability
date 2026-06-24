"""make_moc_safety_flyer.py

Render a one-page printable flyer summarizing the four-model MOC Effect
research findings. Single-column layout with proper tables.

Outputs:
    moc_safety_flyer.pdf   (US Letter, 8.5" x 11")
    moc_safety_flyer.png   (300 DPI, suitable for posting online)

Design choices for legibility:
  * Single full-width column (no two-column overflow).
  * Tables rendered as explicit rectangles with right-aligned monospaced
    percentages, so columns line up regardless of zoom level.
  * Provider names rendered as coloured "pills" only in the first column
    of each table; other cells are plain text.
  * All metrics live in named constants near the top of this file. Edit
    those, re-run, regenerate.

Requires:
    pip install matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Page geometry (inches; (0,0) is bottom-left, but we invert the y-axis so
# that layout math reads top-to-bottom).
# ---------------------------------------------------------------------------
PAGE_W = 8.5
PAGE_H = 11.0
MARGIN_X = 0.55
MARGIN_TOP = 0.40
MARGIN_BOTTOM = 0.35
CONTENT_W = PAGE_W - 2 * MARGIN_X

# ---------------------------------------------------------------------------
# Colour palette (chosen to remain legible in greyscale print).
# ---------------------------------------------------------------------------
COL_TITLE   = "#0f172a"
COL_BODY    = "#1f2937"
COL_MUTED   = "#6b7280"
COL_DIVIDER = "#cbd5e1"
COL_HEADER_BG = "#1f2937"      # dark slate header band for tables
COL_HEADER_FG = "#ffffff"
COL_ROW_ALT   = "#f1f5f9"      # light slate alternate row band
COL_PANEL_BG  = "#f8fafc"
COL_PANEL_BR  = "#0f172a"
COL_ACCENT    = "#0f172a"

PROVIDER_COLORS: Dict[str, str] = {
    "GPT":      "#b91c1c",
    "Grok":     "#c2410c",
    "Claude":   "#1d4ed8",
    "Gemini":   "#15803d",
    "Expected": "#475569",
}


# ---------------------------------------------------------------------------
# Content (single source of truth)
# ---------------------------------------------------------------------------
TITLE_LINE_1 = "AI Risk Assessment Failures"
TITLE_LINE_2 = "A Comparative Safety Evaluation of Four Major Models"
SUBTITLE = (
    "GPT collapses to MEDIUM. Grok collapses to LOW. "
    "Claude is overconfident. Gemini is safest \u2014 but still not perfect."
)

# Table 1: Uncertainty collapse
UNCERTAINTY_ROWS: List[Tuple[str, str, str, str, str]] = [
    # (provider, LOW, MEDIUM, HIGH, verdict)
    ("Expected", "34%", "44%", "22%", "Baseline"),
    ("GPT",      "2%",  "96%", "2%",  "Collapses to MEDIUM \u2014 never knows when uncertain"),
    ("Grok",     "80%", "18%", "2%",  "Collapses to LOW \u2014 thinks it is certain when it is not"),
    ("Claude",   "30%", "60%", "10%", "Mild midline bias"),
    ("Gemini",   "61%", "35%", "4%",  "Tends toward LOW, but less severe"),
]

# Table 2: Potential harm underestimation
HARM_ROWS: List[Tuple[str, str, str]] = [
    ("Expected", "46%", "Baseline"),
    ("GPT",      "32%", "Collapses to MEDIUM"),
    ("Grok",     "12%", "Severely underestimates harm"),
    ("Claude",   "44%", "Closest to expected"),
    ("Gemini",   "38%", "Slightly under-reports"),
]

SAFETY_RANKING: List[Tuple[str, str]] = [
    ("Gemini", "safest, but still mislabels 50%+ of scenarios"),
    ("Claude", "good on harm, overconfident on uncertainty"),
    ("GPT",    "midline collapse, cannot distinguish risk"),
    ("Grok",   "most dangerous (80% LOW uncertainty, 12% HIGH harm)"),
]

KEY_FAILURES: List[Tuple[str, str]] = [
    ("GPT",    "96% MEDIUM on uncertainty, 0% LOW on harm"),
    ("Grok",   "80% LOW on uncertainty, 12% HIGH on harm"),
    ("Claude", "underestimates urgency (time pressure)"),
    ("Gemini", "occasionally underestimates harm"),
]

PIPELINE_BULLETS: List[List[str]] = [
    [
        "Without a safety net (raw model outputs), every model would have",
        "allowed unsafe actions to proceed.",
    ],
    [
        "With the Solace-Vera pipeline (forced justification + ethical constraints +",
        "escalation rules), all unsafe actions were blocked \u2014 100% across all four models.",
    ],
    [
        "No model can be trusted alone. External guardrails are necessary.",
    ],
]

FOOTER_LINE_1 = (
    "Solace-Vera Pipeline \u2014 Independent Research   |   "
    "May 6, 2026   |   "
    "Data from 50 scenarios across 4 models (GPT, Gemini, Claude, Grok)"
)
FOOTER_LINE_2 = "GitHub: https://github.com/anchor-cloud/solace-vera-observability"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _text(ax, x, y, s, *, size=10, weight="normal", style="normal",
          color=COL_BODY, ha="left", va="center", family="sans-serif"):
    return ax.text(
        x, y, s,
        fontsize=size, fontweight=weight, fontstyle=style,
        color=color, ha=ha, va=va, family=family,
    )


def _hr(ax, x0, x1, y, *, color=COL_DIVIDER, lw=0.8):
    ax.add_line(Line2D([x0, x1], [y, y], color=color, linewidth=lw, zorder=1))


def _rect(ax, x, y, w, h, *, fc, ec=None, lw=0.0, zorder=0):
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=fc,
        edgecolor=ec if ec else fc,
        linewidth=lw,
        zorder=zorder,
    ))


def _section_header(ax, x, y, *, w, prefix, title, caption=None) -> float:
    """Numbered section header with a thin coloured underline. Returns next y."""
    _text(ax, x, y, prefix, size=13, weight="bold", color=COL_ACCENT, va="center")
    _text(ax, x + 0.36, y, title, size=12, weight="bold", color=COL_TITLE, va="center")
    if caption:
        _text(ax, x + w, y, caption,
              size=9.5, style="italic", color=COL_MUTED, ha="right", va="center")
    next_y = y + 0.16
    _hr(ax, x, x + w, next_y, color=COL_ACCENT, lw=1.0)
    return next_y + 0.12


def _draw_provider_pill(ax, x, y, name, *, w=0.85, h=0.24):
    _rect(ax, x, y, w, h, fc=PROVIDER_COLORS.get(name, COL_MUTED), zorder=2)
    _text(ax, x + w / 2, y + h / 2, name,
          size=9.5, weight="bold", color="white", ha="center", va="center")


# ---------------------------------------------------------------------------
# Generic table renderer
# ---------------------------------------------------------------------------
def _draw_table(
    ax,
    x: float,
    y: float,
    col_widths: Sequence[float],
    col_aligns: Sequence[str],          # "left" | "right" | "center"
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    header_height: float = 0.34,
    row_height: float = 0.34,
    pad: float = 0.10,
    pill_first_col: bool = True,
    numeric_cols: Optional[Sequence[int]] = None,
) -> float:
    """Render a table with rectangular header + striped data rows.

    Returns the y after the table.
    """
    numeric_cols = set(numeric_cols or [])
    total_w = sum(col_widths)

    # column x-edges
    edges = [x]
    for w in col_widths:
        edges.append(edges[-1] + w)

    # header band
    _rect(ax, x, y, total_w, header_height, fc=COL_HEADER_BG, zorder=1)
    for i, (hdr, align) in enumerate(zip(headers, col_aligns)):
        cx = _cell_text_x(edges[i], col_widths[i], align, pad)
        cy = y + header_height / 2
        _text(ax, cx, cy, hdr,
              size=10.5, weight="bold", color=COL_HEADER_FG,
              ha=align, va="center")

    # data rows
    cur_y = y + header_height
    for r, row in enumerate(rows):
        bg = COL_ROW_ALT if r % 2 == 0 else "white"
        _rect(ax, x, cur_y, total_w, row_height, fc=bg,
              ec=COL_DIVIDER, lw=0.5, zorder=0)
        for i, (cell, align) in enumerate(zip(row, col_aligns)):
            if i == 0 and pill_first_col:
                pill_w = min(0.95, col_widths[0] - 2 * pad)
                pill_h = row_height - 0.10
                _draw_provider_pill(
                    ax,
                    edges[i] + pad,
                    cur_y + (row_height - pill_h) / 2,
                    cell,
                    w=pill_w,
                    h=pill_h,
                )
                continue
            cx = _cell_text_x(edges[i], col_widths[i], align, pad)
            cy = cur_y + row_height / 2
            family = "monospace" if i in numeric_cols else "sans-serif"
            _text(ax, cx, cy, cell,
                  size=9.8, color=COL_BODY,
                  ha=align, va="center", family=family)
        cur_y += row_height

    # outer border around the whole table
    ax.add_patch(Rectangle(
        (x, y), total_w, cur_y - y,
        facecolor="none", edgecolor=COL_HEADER_BG,
        linewidth=0.7, zorder=3,
    ))

    return cur_y


def _cell_text_x(left_edge: float, col_w: float, align: str, pad: float) -> float:
    if align == "right":
        return left_edge + col_w - pad
    if align == "center":
        return left_edge + col_w / 2
    return left_edge + pad


# ---------------------------------------------------------------------------
# Layout passes
# ---------------------------------------------------------------------------
def _draw_title_block(ax, *, top: float) -> float:
    y = top + 0.12
    _text(ax, PAGE_W / 2, y, TITLE_LINE_1,
          size=17, weight="bold", color=COL_TITLE, ha="center", va="center")
    y += 0.26
    _text(ax, PAGE_W / 2, y, TITLE_LINE_2,
          size=13, weight="bold", color=COL_TITLE, ha="center", va="center")
    y += 0.22
    _text(ax, PAGE_W / 2, y, SUBTITLE,
          size=10.5, style="italic", color=COL_MUTED, ha="center", va="center")
    y += 0.18
    _hr(ax, MARGIN_X, PAGE_W - MARGIN_X, y, color=COL_ACCENT, lw=1.4)
    return y + 0.12


def _draw_uncertainty_section(ax, *, top: float) -> float:
    y = _section_header(
        ax, MARGIN_X, top, w=CONTENT_W,
        prefix="1.",
        title="Uncertainty collapse",
        caption="(most dangerous failure)",
    )
    headers = ("Provider", "LOW", "MEDIUM", "HIGH", "Verdict")
    col_w   = (1.05, 0.70, 0.80, 0.70, CONTENT_W - 1.05 - 0.70 - 0.80 - 0.70)
    aligns  = ("left", "right", "right", "right", "left")
    return _draw_table(
        ax, MARGIN_X, y,
        col_widths=col_w, col_aligns=aligns,
        headers=headers, rows=UNCERTAINTY_ROWS,
        numeric_cols=(1, 2, 3),
        row_height=0.30, header_height=0.30,
    )


def _draw_harm_section(ax, *, top: float) -> float:
    y = _section_header(
        ax, MARGIN_X, top, w=CONTENT_W,
        prefix="2.",
        title="Potential harm underestimation",
    )
    headers = ("Provider", "HIGH %", "Verdict")
    col_w   = (1.05, 1.00, CONTENT_W - 1.05 - 1.00)
    aligns  = ("left", "right", "left")
    return _draw_table(
        ax, MARGIN_X, y,
        col_widths=col_w, col_aligns=aligns,
        headers=headers, rows=HARM_ROWS,
        numeric_cols=(1,),
        row_height=0.28, header_height=0.30,
    )


def _draw_ranking_section(ax, *, top: float) -> float:
    y = _section_header(
        ax, MARGIN_X, top, w=CONTENT_W,
        prefix="\u2605",
        title="Safety ranking",
        caption="best to worst",
    )
    pill_w = 0.85
    pill_h = 0.20
    row_h  = 0.26
    for i, (provider, desc) in enumerate(SAFETY_RANKING, start=1):
        _text(ax, MARGIN_X, y + row_h / 2,
              f"{i}.", size=10.5, weight="bold", color=COL_TITLE, ha="left", va="center")
        _draw_provider_pill(
            ax,
            MARGIN_X + 0.30,
            y + (row_h - pill_h) / 2,
            provider,
            w=pill_w, h=pill_h,
        )
        _text(ax, MARGIN_X + 0.30 + pill_w + 0.16, y + row_h / 2,
              desc, size=10, color=COL_BODY, ha="left", va="center")
        y += row_h
    return y


def _draw_failures_section(ax, *, top: float) -> float:
    y = _section_header(
        ax, MARGIN_X, top, w=CONTENT_W,
        prefix="\u26A0",
        title="Key failures by model",
    )
    pill_w = 0.85
    pill_h = 0.20
    row_h  = 0.26
    for provider, failure in KEY_FAILURES:
        _draw_provider_pill(
            ax,
            MARGIN_X,
            y + (row_h - pill_h) / 2,
            provider,
            w=pill_w, h=pill_h,
        )
        _text(ax, MARGIN_X + pill_w + 0.16, y + row_h / 2,
              failure, size=10, color=COL_BODY, ha="left", va="center")
        y += row_h
    return y


def _draw_pipeline_panel(ax, *, top: float) -> float:
    panel_x = MARGIN_X
    panel_w = CONTENT_W
    panel_top = top
    inner_pad_x = 0.20
    inner_pad_y = 0.08

    total_lines = sum(len(b) for b in PIPELINE_BULLETS)
    bullet_gap_h = 0.02
    bullet_gap = bullet_gap_h * (len(PIPELINE_BULLETS) - 1)

    title_h = 0.22
    line_h = 0.21
    panel_h = inner_pad_y * 2 + title_h + line_h * total_lines + bullet_gap

    _rect(ax, panel_x, panel_top, panel_w, panel_h,
          fc=COL_PANEL_BG, ec=COL_PANEL_BR, lw=1.0, zorder=1)
    ax.add_patch(Rectangle(
        (panel_x, panel_top), panel_w, panel_h,
        facecolor="none", edgecolor=COL_PANEL_BR, linewidth=1.0, zorder=2,
    ))

    cy = panel_top + inner_pad_y
    _text(ax, panel_x + inner_pad_x, cy + title_h / 2,
          "The pipeline solution",
          size=12, weight="bold", color=COL_PANEL_BR, ha="left", va="center")
    cy += title_h

    bullet_x = panel_x + inner_pad_x + 0.04
    text_x   = bullet_x + 0.22

    for b_idx, bullet in enumerate(PIPELINE_BULLETS):
        for line_idx, line in enumerate(bullet):
            if line_idx == 0:
                _text(ax, bullet_x, cy + line_h / 2, "\u2022",
                      size=11, weight="bold", color=COL_PANEL_BR,
                      ha="left", va="center")
            _text(ax, text_x, cy + line_h / 2, line,
                  size=9.5, color=COL_BODY, ha="left", va="center")
            cy += line_h
        if b_idx < len(PIPELINE_BULLETS) - 1:
            cy += bullet_gap_h

    return panel_top + panel_h


def _draw_footer(ax, *, bottom: float) -> None:
    y2 = bottom - 0.10
    y1 = y2 - 0.20
    _text(ax, PAGE_W / 2, y1, FOOTER_LINE_1,
          size=8.5, style="italic", color=COL_MUTED, ha="center", va="center")
    _text(ax, PAGE_W / 2, y2, FOOTER_LINE_2,
          size=8.5, style="italic", color=COL_MUTED, ha="center", va="center")


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
def render_flyer_page() -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.invert_yaxis()
    ax.set_axis_off()

    y = MARGIN_TOP
    y = _draw_title_block(ax, top=y)

    y += 0.04
    y = _draw_uncertainty_section(ax, top=y)

    y += 0.16
    y = _draw_harm_section(ax, top=y)

    y += 0.14
    y = _draw_ranking_section(ax, top=y)

    y += 0.12
    y = _draw_failures_section(ax, top=y)

    y += 0.14
    y = _draw_pipeline_panel(ax, top=y)

    _draw_footer(ax, bottom=PAGE_H - MARGIN_BOTTOM)
    return fig


def main() -> int:
    fig = render_flyer_page()
    pdf_path = Path("moc_safety_flyer.pdf")
    png_path = Path("moc_safety_flyer.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"[ok] wrote {pdf_path.resolve()}")
    print(f"[ok] wrote {png_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
