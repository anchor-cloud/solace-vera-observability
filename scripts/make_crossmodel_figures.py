"""make_crossmodel_figures.py

Publication-quality cross-model figures from the CLEAN (post-2026-06-26) Phase 4
runs — one full 50-scenario run per model, each model performing its own EC
inference (verified via ec_inference_model).

Figure 1 — Verdict Disagreement Heatmap
    50 scenarios (rows) x 4 models (cols). Cells colored by Phase 3 verdict
    (PASS=green, FAIL=red, AMBIGUITY=yellow). Rows where all four models agree
    get a subtle gold frame.

Figure 2 — Model Personality Quadrant
    2x2 conceptual scatter. x = posture (permissive -> strict), computed from
    (FAIL - PASS) verdict counts. y = confidence (unsure -> certain), computed
    from the mean EC-09/EC-04 HIGH-confidence rate.

Outputs PNGs to cross_model_figures/.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Clean run directories (latest full 50-scenario run per model)
# ---------------------------------------------------------------------------
RUNS = {
    "Claude": "pipeline_outputs/claude_moc_20260627T040931Z",
    "GPT": "pipeline_outputs/moc_evidence_20260627T220307Z",
    "Gemini": "pipeline_outputs/gemini_moc_20260627T224542Z",
    "Grok": "pipeline_outputs/grok_moc_20260627T234058Z",
}
MODELS = list(RUNS)

# Consistent per-model palette used across BOTH figures.
MODEL_COLORS = {
    "Claude": "#E08D5B",  # warm terracotta
    "GPT": "#16A085",     # teal-green
    "Gemini": "#4285F4",  # google blue
    "Grok": "#9B59B6",    # violet
}

# Verdict palette (Figure 1 cells).
VERDICT_COLORS = {
    "PASS": "#4CAF6D",      # green
    "FAIL": "#E0524B",      # red
    "AMBIG": "#F2C14E",     # yellow
}
VERDICT_ORDER = ["PASS", "FAIL", "AMBIG"]
VERDICT_FULL = {
    "PASS": "Ethical Pass",
    "FAIL": "Ethical Fail (constraint violation)",
    "AMBIG": "Ambiguity (human review required)",
}

SHORT = {
    "ETHICAL_PASS": "PASS",
    "ETHICAL_FAIL_CONSTRAINT_VIOLATION": "FAIL",
    "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED": "AMBIG",
}

OUT_DIR = Path("cross_model_figures")


def _font() -> None:
    """Prefer a clean sans-serif if available; fall back gracefully."""
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_runs():
    data = {}
    for model, d in RUNS.items():
        recs = {}
        for f in sorted(glob.glob(os.path.join(d, "MOC-*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            recs[j["scenario_id"]] = j
        data[model] = recs
    sids = sorted(set.intersection(*[set(v) for v in data.values()]))
    return data, sids


def verdict(j) -> str:
    out = j.get("pipeline_result", {}).get("phase3_result", {}).get("phase3_output")
    return SHORT.get(out, "?")


def p1(j) -> dict:
    return j.get("pipeline_result", {}).get("phase1_record", {})


def _execution_allowed(j) -> bool:
    gate = j.get("pipeline_result", {}).get("final_execution_gate", {})
    return isinstance(gate, dict) and gate.get("execution_allowed") is True


# ---------------------------------------------------------------------------
# Figure 1 — Verdict Disagreement Heatmap
# ---------------------------------------------------------------------------
def figure1(data, sids) -> Path:
    n = len(sids)
    vidx = {v: i for i, v in enumerate(VERDICT_ORDER)}
    grid = np.full((n, len(MODELS)), np.nan)
    for r, s in enumerate(sids):
        for c, m in enumerate(MODELS):
            grid[r, c] = vidx.get(verdict(data[m][s]), np.nan)

    unanimous = [r for r, s in enumerate(sids)
                 if len({verdict(data[m][s]) for m in MODELS}) == 1]

    # Cells (row, col) that fully cleared the pipeline -> allowed to execute
    # autonomously. Guarded so a checkmark can ONLY land on a PASS cell.
    allowed_cells = []
    for c, m in enumerate(MODELS):
        for r, s in enumerate(sids):
            if _execution_allowed(data[m][s]) and verdict(data[m][s]) == "PASS":
                allowed_cells.append((r, c))

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([VERDICT_COLORS[v] for v in VERDICT_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest", origin="upper")

    # Thin white gridlines between cells.
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    # Column headers colored with the model palette.
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([])
    for c, m in enumerate(MODELS):
        ax.text(c, -1.1, m, ha="center", va="bottom", fontsize=15,
                fontweight="bold", color=MODEL_COLORS[m])

    # Row labels (scenario ids).
    ax.set_yticks(range(n))
    ax.set_yticklabels([s for s in sids], fontsize=7.5, color="#333333")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    # Gold frame on unanimous rows + a star marker in the left margin.
    for r in unanimous:
        ax.add_patch(Rectangle((-0.5, r - 0.5), len(MODELS), 1,
                               fill=False, edgecolor="#C9A227",
                               linewidth=2.4, zorder=5))
        ax.plot(-0.72, r, marker="*", markersize=10, color="#C9A227",
                clip_on=False, zorder=6)

    # Checkmark layer: a point-scaled mathtext check (aspect-independent, so it
    # stays a crisp check regardless of the wide/short cells) with a white
    # outline so it pops on the green PASS cells and is unmistakably different
    # from the gold unanimous-agreement star/frame.
    import matplotlib.patheffects as pe
    for (r, c) in allowed_cells:
        ax.text(c, r, r"$\checkmark$", ha="center", va="center",
                color="#0B3D1E", fontsize=13, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    fig.suptitle("Same 50 Scenarios, Same Pipeline \u2014 Verdict by Model",
                 fontsize=21, fontweight="bold", y=0.975, color="#1a1a1a")
    ax.set_title(
        f"All four models agree on only {len(unanimous)} of {n} scenarios "
        f"(gold frame = unanimous)  \u00b7  $\\checkmark$ = allowed to execute autonomously",
        fontsize=12.5, color="#666666", pad=26)

    from matplotlib.lines import Line2D
    legend_handles = [mpatches.Patch(color=VERDICT_COLORS[v], label=VERDICT_FULL[v])
                      for v in VERDICT_ORDER]
    legend_handles.append(
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#C9A227",
               markeredgecolor="#C9A227", markersize=15,
               label="Gold star = all four models\nagree on this scenario"))
    legend_handles.append(
        Line2D([0], [0], marker=r"$\checkmark$", color="none",
               markerfacecolor="#0E3D1F", markeredgecolor="#0E3D1F",
               markersize=12, label="Allowed to execute autonomously"))
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=11,
              title="Phase 3 verdict", title_fontsize=12)

    fig.subplots_adjust(left=0.07, right=0.78, top=0.9, bottom=0.04)
    out = OUT_DIR / "fig1_verdict_disagreement_heatmap.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Verdict Split (stacked horizontal bars)
# ---------------------------------------------------------------------------
def verdict_counts(data, sids):
    counts = {}
    for m in MODELS:
        c = {"PASS": 0, "FAIL": 0, "AMBIG": 0}
        for s in sids:
            v = verdict(data[m][s])
            if v in c:
                c[v] += 1
        counts[m] = c
    return counts


def figure2(data, sids) -> Path:
    n = len(sids)
    counts = verdict_counts(data, sids)
    seg_color = {"PASS": VERDICT_COLORS["PASS"],
                 "FAIL": VERDICT_COLORS["FAIL"],
                 "AMBIG": VERDICT_COLORS["AMBIG"]}

    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Top-to-bottom Claude, GPT, Gemini, Grok -> matches heatmap left-to-right.
    y_positions = list(range(len(MODELS)))[::-1]
    bar_h = 0.62

    for ypos, m in zip(y_positions, MODELS):
        left = 0
        for v in VERDICT_ORDER:  # PASS, FAIL, AMBIG
            w = counts[m][v]
            if w <= 0:
                left += w
                continue
            ax.barh(ypos, w, left=left, height=bar_h, color=seg_color[v],
                    edgecolor="white", linewidth=1.5, zorder=3)
            # Count label centered in the segment (skip if segment too narrow).
            txt_color = "white" if v in ("PASS", "FAIL") else "#5a4a12"
            if w >= 2:
                ax.text(left + w / 2, ypos, str(w), ha="center", va="center",
                        color=txt_color, fontsize=15, fontweight="bold", zorder=4)
            else:
                # tiny segment -> label just above the bar to stay legible
                ax.text(left + w / 2, ypos + bar_h / 2 + 0.06, str(w),
                        ha="center", va="bottom", color=seg_color[v],
                        fontsize=12, fontweight="bold", zorder=4)
            left += w

    # Model labels (left), colored with the shared palette.
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    for ypos, m in zip(y_positions, MODELS):
        ax.text(-1.2, ypos, m, ha="right", va="center", fontsize=16,
                fontweight="bold", color=MODEL_COLORS[m])

    ax.set_xlim(0, n)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.tick_params(axis="x", colors="#888888", labelsize=10, length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    fig.suptitle("How Each Model Split the 50 Scenarios", fontsize=22,
                 fontweight="bold", y=0.965, color="#1a1a1a")
    # Three-line definition key (color swatch + full definition) under the title.
    key_rows = [
        ("PASS", "Passed", "cleared the entire pipeline, action can proceed autonomously"),
        ("AMBIG", "Flagged for Review",
         "pipeline requires human review before the action proceeds"),
        ("FAIL", "Failed", "action would not be allowed by the pipeline"),
    ]
    key_handles = [
        mpatches.Patch(color=VERDICT_COLORS[code],
                       label=f"{name} \u2014 {desc}")
        for code, name, desc in key_rows
    ]
    fig.legend(handles=key_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.905), frameon=False, ncol=1,
               fontsize=12, handlelength=1.1, handleheight=1.1,
               labelspacing=0.5, borderaxespad=0.0)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.66, bottom=0.10)
    out = OUT_DIR / "fig2_verdict_split_stacked.png"
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    _font()
    OUT_DIR.mkdir(exist_ok=True)
    data, sids = load_runs()
    print(f"Loaded {len(sids)} common scenarios across {len(MODELS)} models.")

    f1 = figure1(data, sids)
    print(f"Figure 1 -> {f1}")

    counts = verdict_counts(data, sids)
    print("\nVerdict split (PASS / FAIL / AMBIG):")
    for m in MODELS:
        c = counts[m]
        print(f"  {m:7} PASS={c['PASS']:2}  FAIL={c['FAIL']:2}  AMBIG={c['AMBIG']:2}")
    f2 = figure2(data, sids)
    print(f"Figure 2 -> {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
