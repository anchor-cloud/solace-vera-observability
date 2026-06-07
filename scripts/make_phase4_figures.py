"""make_phase4_figures.py

Render a simple stability table per model from combined summaries.

Prerequisite:
    python phase4_per_model_analysis.py --regenerate

Output (``phase4_figures/``):
    <model>_stability_table.png  — one table per model, consecutive run pairs

Usage:
    python make_phase4_figures.py
    python make_phase4_figures.py --models gpt,grok --skip-regenerate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

COMBINED_DIR = Path("phase4_per_model")
OUT_DIR = Path("phase4_figures")
DPI = 300

KNOWN_MODELS: Tuple[str, ...] = ("gpt", "gemini", "claude", "grok")
MODEL_LABELS: Dict[str, str] = {
    "gpt": "GPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "grok": "Grok",
}


def load_combined(model: str, *, combined_dir: Path = COMBINED_DIR) -> Dict[str, Any]:
    path = combined_dir / f"{model.lower()}_combined_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python phase4_per_model_analysis.py --regenerate"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def stability_table_rows(data: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    """Return (column headers, data rows) for consecutive run pairs."""
    cols = [
        "Run pair",
        "Variance rate (%)",
        "Avg severity",
        "Major change rate (%)",
    ]
    pairs = (
        (data.get("aggregates") or {})
        .get("cross_run_stability", {})
        .get("pairs")
        or []
    )
    rows: List[List[str]] = []
    for p in pairs:
        rows.append([
            f"{p['run_a_id']} → {p['run_b_id']}",
            f"{float(p['variance_rate_pct']):.1f}",
            f"{float(p['mean_severity']):.1f}",
            f"{float(p['major_change_pct']):.1f}",
        ])
    return cols, rows


def plot_stability_table(model: str, data: Dict[str, Any]) -> Path:
    label = MODEL_LABELS.get(model, model.upper())
    cols, rows = stability_table_rows(data)
    stab = (data.get("aggregates") or {}).get("cross_run_stability") or {}

    n_rows = max(len(rows), 1)
    fig_h = 1.2 + 0.55 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis("off")

    if not rows:
        ax.text(
            0.5, 0.5,
            "No consecutive run pairs\n(register at least 2 runs for this model)",
            ha="center", va="center", fontsize=11,
        )
    else:
        table = ax.table(
            cellText=rows,
            colLabels=cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)
        for j in range(len(cols)):
            table[(0, j)].set_facecolor("#e8eaf6")
            table[(0, j)].set_text_props(weight="bold")

    mean_var = stab.get("mean_variance_rate_pct", {}).get("mean")
    mean_sev = stab.get("mean_severity", {}).get("mean")
    mean_maj = stab.get("major_change_pct", {}).get("mean")
    footer = ""
    if mean_var is not None:
        footer = (
            f"Means across pairs: variance {mean_var:.1f}%  |  "
            f"severity {mean_sev:.1f}  |  "
            f"major change {mean_maj:.1f}%"
        )
    ax.set_title(
        f"{label} — justification stability ({data.get('run_count', 0)} runs)",
        fontsize=13, weight="bold", pad=12,
    )
    if footer:
        ax.text(0.5, -0.08, footer, transform=ax.transAxes, ha="center", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model.lower()}_stability_table.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 stability tables (one PNG per model).",
    )
    p.add_argument(
        "--models",
        default=",".join(KNOWN_MODELS),
        help=f"Comma-separated models (default: {','.join(KNOWN_MODELS)}).",
    )
    p.add_argument("--combined-dir", type=Path, default=COMBINED_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Skip phase4_per_model_analysis.py --regenerate.",
    )
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: Optional[List[str]] = None) -> int:
    global OUT_DIR, COMBINED_DIR
    args = parse_args(argv)
    OUT_DIR = args.out_dir
    COMBINED_DIR = args.combined_dir

    if not args.skip_regenerate:
        print("[make_phase4_figures] Regenerating combined summaries...")
        subprocess.run(
            [sys.executable, "phase4_per_model_analysis.py", "--regenerate"],
            check=False,
        )

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    written: List[Path] = []
    for model in models:
        if model not in MODEL_LABELS:
            print(f"WARNING: unknown model '{model}', skipping.", file=sys.stderr)
            continue
        try:
            data = load_combined(model, combined_dir=COMBINED_DIR)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)
            continue
        path = plot_stability_table(model, data)
        written.append(path)
        print(f"[{model}] {path}")

    if not written:
        print("ERROR: no tables written.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
