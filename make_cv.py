"""make_cv.py

Render Ashlie Mulnix's one-page CV (``Ashlie_Mulnix_CV.pdf``) using
matplotlib's PDF backend. Hyperlinks are embedded via ``text(..., url=...)``
so a PDF viewer renders them as clickable links.

The content is laid out by hand on a US-Letter canvas. Re-run after any
edit to ``CONTENT`` below to regenerate the file.

Output:
    Ashlie_Mulnix_CV.pdf  (in the repo root)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt


OUT_PATH = Path("Ashlie_Mulnix_CV.pdf")


# ---------------------------------------------------------------------------
# Content (mirrors the source-of-truth text exactly)
# ---------------------------------------------------------------------------
NAME      = "Ashlie Mulnix"
TAGLINE   = "Independent AI Safety Researcher"

SELECTED_WORK: List[str] = [
    "Built Solace-Vera, a 4-phase pre-action auditing pipeline "
    "(GitHub link below)",
    "Tested GPT, Gemini, Claude, Grok on 50 risk scenarios, two runs each",
    "Discovered that justifications change 100% of the time when risk "
    "scores are identical",
    "Published definition of MOC Effect on Zenodo",
    "Reported initial finding (MOC Effect) to OpenAI (out of scope). "
    "At a UC Berkeley researcher's suggestion, extended tests to Gemini, "
    "Claude, and Grok. Gained cross-model insights.",
]

BACKGROUND: List[str] = [
    "Self-taught, no CS degree",
    "Small business employee (florist)",
    "Two-year degree in Anthropology",
    "Two-year degree in Social Sciences",
]

LINKS: List[Tuple[str, str, str]] = [
    # (label, display_text, url)
    ("GitHub",
     "https://github.com/anchor-cloud/solace-vera-observability",
     "https://github.com/anchor-cloud/solace-vera-observability"),
    ("Zenodo",
     "https://zenodo.org/records/19957469",
     "https://zenodo.org/records/19957469"),
]

EXAMPLE_EVIDENCE_DESC = (
    "Example evidence: Comparison of four models on a single "
    "high-risk scenario"
)
EXAMPLE_EVIDENCE_URL = (
    "https://github.com/anchor-cloud/solace-vera-observability/blob/"
    "main/blog_figures/fig_model_comparison_moc013.png"
)
# Display the long evidence URL split over two lines so it fits within the
# right margin. Each fragment becomes its own clickable annotation pointing
# at the same target URL above.
EXAMPLE_EVIDENCE_URL_LINES = (
    "https://github.com/anchor-cloud/solace-vera-observability/",
    "blob/main/blog_figures/fig_model_comparison_moc013.png",
)

LOOKING_FOR = (
    "Guidance on where to publish these findings, how to find collaborators "
    "to turn the scenario set into a benchmark, and whether to apply for a "
    "fellowship or research role."
)


# ---------------------------------------------------------------------------
# Layout constants (figure-fraction coordinates on an 8.5x11" canvas)
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 8.5, 11.0  # US Letter

MARGIN_L      = 0.085
MARGIN_R      = 0.915
BULLET_X      = MARGIN_L + 0.003
TEXT_INDENT_X = MARGIN_L + 0.022

WRAP_BODY     = 92   # chars per line for indented body text @ 10.5pt
WRAP_LOOKING  = 96   # chars per line for the closing paragraph

COLOR_TITLE       = "#0e1320"
COLOR_TAGLINE     = "#3a3f4d"
COLOR_SECTION     = "#0e1320"
COLOR_BODY        = "#1a1a1a"
COLOR_BULLET      = "#5b6470"
COLOR_LINK        = "#1f4eb6"

FS_NAME    = 22
FS_TAGLINE = 13
FS_SECTION = 13
FS_BODY    = 10.5
FS_BULLET  = 12  # bullet glyph slightly larger so it reads cleanly

LINE_H            = 0.0185  # figure-fraction per line of body text
SECTION_GAP       = 0.028
AFTER_SECTION_HD  = 0.024
BETWEEN_BULLETS   = 0.003


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_section_heading(ax, y: float, text: str) -> float:
    ax.text(
        MARGIN_L, y, text,
        fontsize=FS_SECTION, weight="bold", color=COLOR_SECTION, va="top",
    )
    # Subtle horizontal rule beneath the heading
    rule_y = y - 0.012
    ax.plot(
        [MARGIN_L, MARGIN_R], [rule_y, rule_y],
        transform=ax.transAxes,
        color="#d9dde3", lw=0.7, solid_capstyle="butt",
    )
    return y - AFTER_SECTION_HD


def draw_bullet(ax, y: float, text: str, *, wrap_w: int = WRAP_BODY) -> float:
    wrapped = textwrap.fill(text, width=wrap_w, break_long_words=False)
    n_lines = wrapped.count("\n") + 1
    ax.text(
        BULLET_X, y, "\u2022",  # • bullet
        fontsize=FS_BULLET, color=COLOR_BULLET, va="top",
    )
    ax.text(
        TEXT_INDENT_X, y, wrapped,
        fontsize=FS_BODY, color=COLOR_BODY, va="top",
        linespacing=1.30,
    )
    return y - (LINE_H * n_lines + BETWEEN_BULLETS)


def draw_link_bullet(ax, y: float, label: str, url: str) -> float:
    """Bullet with 'Label: <clickable URL>'."""
    ax.text(
        BULLET_X, y, "\u2022",
        fontsize=FS_BULLET, color=COLOR_BULLET, va="top",
    )
    label_str = f"{label}: "
    ax.text(
        TEXT_INDENT_X, y, label_str,
        fontsize=FS_BODY, color=COLOR_BODY, va="top",
    )
    # Approximate label width in figure-fraction; just enough offset.
    label_offset = 0.012 * len(label_str)
    ax.text(
        TEXT_INDENT_X + label_offset, y, url,
        fontsize=FS_BODY, color=COLOR_LINK, va="top",
        url=url,
    )
    return y - (LINE_H + BETWEEN_BULLETS)


def draw_paragraph(ax, y: float, text: str, *, wrap_w: int = WRAP_LOOKING) -> float:
    wrapped = textwrap.fill(text, width=wrap_w, break_long_words=False)
    n_lines = wrapped.count("\n") + 1
    ax.text(
        MARGIN_L, y, wrapped,
        fontsize=FS_BODY, color=COLOR_BODY, va="top", linespacing=1.35,
    )
    return y - LINE_H * n_lines


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def main() -> int:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----- Header -----
    y = 0.94
    ax.text(MARGIN_L, y, NAME,
            fontsize=FS_NAME, weight="bold", color=COLOR_TITLE, va="top")
    y -= 0.040
    ax.text(MARGIN_L, y, TAGLINE,
            fontsize=FS_TAGLINE, color=COLOR_TAGLINE,
            style="italic", va="top")
    y -= SECTION_GAP + 0.010

    # ----- Selected work -----
    y = draw_section_heading(ax, y, "Selected work")
    for item in SELECTED_WORK:
        y = draw_bullet(ax, y, item)
    y -= SECTION_GAP

    # ----- Background -----
    y = draw_section_heading(ax, y, "Background")
    for item in BACKGROUND:
        y = draw_bullet(ax, y, item)
    y -= SECTION_GAP

    # ----- Links (with hyperlinks) -----
    y = draw_section_heading(ax, y, "Links")
    for label, display, url in LINKS:
        y = draw_link_bullet(ax, y, label, url)

    # Example-evidence link: description line, then the URL wrapped over two
    # lines (both clickable, both pointing at the same target).
    ax.text(BULLET_X, y, "\u2022",
            fontsize=FS_BULLET, color=COLOR_BULLET, va="top")
    ax.text(TEXT_INDENT_X, y, EXAMPLE_EVIDENCE_DESC,
            fontsize=FS_BODY, color=COLOR_BODY, va="top")
    y -= LINE_H
    for frag in EXAMPLE_EVIDENCE_URL_LINES:
        ax.text(TEXT_INDENT_X, y, frag,
                fontsize=FS_BODY - 0.5, color=COLOR_LINK, va="top",
                url=EXAMPLE_EVIDENCE_URL)
        y -= LINE_H
    y -= SECTION_GAP

    # ----- What I am looking for -----
    y = draw_section_heading(ax, y, "What I am looking for")
    y = draw_paragraph(ax, y, LOOKING_FOR)

    fig.savefig(OUT_PATH, format="pdf")
    plt.close(fig)
    print(f"wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
