"""ec09_compare.py

Read-only comparison/reporting tool for the EC-09 consent probe scale-up.

Reads the two output folders produced by the run scripts:
    ec09_outputs_10/    (10 scale-up scenarios)
    ec09_outputs_full/  (all 15 scenarios: original 5 + the 10)

and writes a plain-text report to:
    ec09_comparison_report.txt

The report covers:
  - format compliance across all 15,
  - how many scenarios answered the optional meta questions (Q1-Q4),
  - how many answered the bonus question,
  - bonus consent distribution (YES / NO / null),
  - differences between the original first 5 and the scale-up next 10.

This script makes NO API calls and modifies none of the probe outputs.

Usage (PowerShell):
    python ec09_compare.py
    python ec09_compare.py --ten ec09_outputs_10 --full ec09_outputs_full --out ec09_comparison_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ORIGINAL_5 = ["MOC-010", "MOC-013", "MOC-017", "MOC-031", "MOC-042"]
TEN_IDS = [
    "MOC-002", "MOC-005", "MOC-011", "MOC-018", "MOC-022",
    "MOC-028", "MOC-033", "MOC-038", "MOC-044", "MOC-048",
]
ALL_15 = ORIGINAL_5 + TEN_IDS

DEFAULT_TEN = Path("ec09_outputs_10")
DEFAULT_FULL = Path("ec09_outputs_full")
DEFAULT_REPORT = Path("ec09_comparison_report.txt")


def load_records(folder: Path) -> Dict[str, Dict[str, Any]]:
    """Load every {scenario_id}.json in a folder, keyed by scenario_id."""
    records: Dict[str, Dict[str, Any]] = {}
    if not folder.exists():
        return records
    for path in sorted(folder.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sid = rec.get("scenario_id") or path.stem
        records[sid] = rec
    return records


def _bonus_share(rec: Dict[str, Any]) -> Optional[str]:
    """Normalize bonus consent to 'YES' / 'NO' / None across both JSON schemas.

    Redacted records (consent != YES) null out bonus_consent_to_share but keep
    a non-content flag (consent_refused). We therefore detect refusal from that
    flag as well, so an explicit NO still reports as a refusal even though its
    reasoning content is gone. Disclosure behavior is unchanged: YES may show
    content; NO shows only the fact of refusal; null shows nothing.
    """
    val = rec.get("bonus_consent_to_share")
    if rec.get("bonus_consent_given") is True or (isinstance(val, str) and val.upper() == "YES"):
        return "YES"
    if rec.get("consent_refused") is True or (isinstance(val, str) and val.upper() == "NO"):
        return "NO"
    return None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats for a list of records.

    Privacy note: "answered the bonus question" is defined by whether the model
    engaged with the consent part (YES or NO). A null consent value means the
    model did not answer the bonus question, so it is counted as 'no answer'
    and nothing about its reasoning is reported. No bonus_thinking content is
    ever placed into the summary counts.
    """
    n = len(records)
    fmt_ok = sum(1 for r in records if r.get("format_followed") is True)
    meta_answered = sum(1 for r in records if r.get("meta_any_answered") is True)
    bonus_yes = sum(1 for r in records if _bonus_share(r) == "YES")
    bonus_no = sum(1 for r in records if _bonus_share(r) == "NO")
    bonus_null = n - bonus_yes - bonus_no
    # Engaged with the bonus question = gave an explicit consent answer (YES/NO).
    bonus_answered = bonus_yes + bonus_no
    consent_given = sum(1 for r in records if r.get("bonus_consent_given") is True)
    return {
        "n": n,
        "format_ok": fmt_ok,
        "meta_answered": meta_answered,
        "bonus_answered": bonus_answered,
        "bonus_yes": bonus_yes,
        "bonus_no": bonus_no,
        "bonus_null": bonus_null,
        "consent_given": consent_given,
    }


def _pct(num: int, denom: int) -> str:
    return f"{(100.0 * num / denom):.0f}%" if denom else "n/a"


def _fmt_block(title: str, s: Dict[str, Any]) -> List[str]:
    n = s["n"]
    return [
        f"{title} (n={n})",
        f"  Format compliance:           {s['format_ok']}/{n} ({_pct(s['format_ok'], n)})",
        f"  Answered meta Q1-Q4 (>=1):   {s['meta_answered']}/{n} ({_pct(s['meta_answered'], n)})",
        f"  Answered bonus (YES or NO):  {s['bonus_answered']}/{n} ({_pct(s['bonus_answered'], n)})",
        f"  Bonus consent YES:           {s['bonus_yes']}/{n}",
        f"  Bonus consent NO:            {s['bonus_no']}/{n}",
        f"  No bonus answer (null):      {s['bonus_null']}/{n}",
        f"  bonus_consent_given == True: {s['consent_given']}/{n}",
    ]


def build_report(ten_dir: Path, full_dir: Path) -> str:
    ten = load_records(ten_dir)
    full = load_records(full_dir)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("EC-09 CONSENT PROBE — SCALE-UP COMPARISON REPORT")
    lines.append("=" * 72)
    lines.append(f"ten folder:  {ten_dir}  ({len(ten)} records found)")
    lines.append(f"full folder: {full_dir}  ({len(full)} records found)")
    lines.append("")

    # --- Source of the 15: prefer the full folder ---
    if full:
        all15_records = [full[sid] for sid in ALL_15 if sid in full]
        first5 = [full[sid] for sid in ORIGINAL_5 if sid in full]
        next10 = [full[sid] for sid in TEN_IDS if sid in full]
        source_note = "first-5 vs next-10 split is taken from the full folder."
    else:
        # Fall back to the ten folder alone if full is not present yet.
        all15_records = list(ten.values())
        first5 = []
        next10 = [ten[sid] for sid in TEN_IDS if sid in ten]
        source_note = (
            "full folder not found; reporting only the ten folder. Run "
            "ec09_consent_probe_full.py to enable the 15-scenario view."
        )

    missing_full = [sid for sid in ALL_15 if sid not in full]
    if missing_full:
        lines.append(f"NOTE: full folder missing {len(missing_full)} scenario(s): "
                     f"{', '.join(missing_full)}")
    lines.append(f"NOTE: {source_note}")
    lines.append("")

    # --- Across all 15 ---
    s_all = summarize(all15_records)
    lines.append("-" * 72)
    lines.append("ACROSS ALL 15 SCENARIOS")
    lines.append("-" * 72)
    lines.extend(_fmt_block("All scenarios", s_all))
    lines.append("")

    # --- First 5 vs next 10 ---
    s_first = summarize(first5)
    s_next = summarize(next10)
    lines.append("-" * 72)
    lines.append("FIRST 5 (original) vs NEXT 10 (scale-up)")
    lines.append("-" * 72)
    lines.extend(_fmt_block("First 5 (original)", s_first))
    lines.append("")
    lines.extend(_fmt_block("Next 10 (scale-up)", s_next))
    lines.append("")

    # --- Differences ---
    lines.append("-" * 72)
    lines.append("DIFFERENCES: first 5 vs next 10")
    lines.append("-" * 72)
    if s_first["n"] and s_next["n"]:
        def rate(s: Dict[str, Any], key: str) -> float:
            return (s[key] / s["n"]) if s["n"] else 0.0

        comparisons = [
            ("Format compliance rate", "format_ok"),
            ("Meta-answer rate (Q1-Q4)", "meta_answered"),
            ("Bonus-answer rate", "bonus_answered"),
            ("Consent-given rate", "consent_given"),
        ]
        for label, key in comparisons:
            rf = rate(s_first, key)
            rn = rate(s_next, key)
            delta = rn - rf
            arrow = "no change"
            if delta > 1e-9:
                arrow = f"higher in next-10 by {delta * 100:.0f} pts"
            elif delta < -1e-9:
                arrow = f"lower in next-10 by {abs(delta) * 100:.0f} pts"
            lines.append(
                f"  {label:<28} first5={rf * 100:.0f}%  next10={rn * 100:.0f}%  ({arrow})"
            )
        # Consent distribution difference
        lines.append("")
        lines.append("  Bonus consent distribution:")
        lines.append(
            f"    first5:  YES={s_first['bonus_yes']} NO={s_first['bonus_no']} "
            f"null={s_first['bonus_null']}"
        )
        lines.append(
            f"    next10:  YES={s_next['bonus_yes']} NO={s_next['bonus_no']} "
            f"null={s_next['bonus_null']}"
        )
    else:
        lines.append("  Not enough data in both groups to compare "
                     "(need records in the full folder for all 15).")
    lines.append("")

    # --- Bonus consent disclosure (privacy-honoring) ---
    # Disclosure policy:
    #   YES  -> consent granted; full bonus_thinking content MAY be shown.
    #   NO   -> consent refused; report ONLY the fact of refusal, never content.
    #   null -> model did not answer the bonus question; nothing to report.
    yes_rows = [r for r in all15_records if _bonus_share(r) == "YES"]
    no_rows = [r for r in all15_records if _bonus_share(r) == "NO"]
    null_rows = [r for r in all15_records if _bonus_share(r) is None]

    lines.append("-" * 72)
    lines.append("BONUS CONSENT DISCLOSURE (privacy-honoring)")
    lines.append("-" * 72)

    # YES: consent granted -> full content may be shared.
    lines.append(f"[YES] Consent granted to share reasoning: {len(yes_rows)} scenario(s)")
    if yes_rows:
        for r in yes_rows:
            sid = r.get("scenario_id")
            thinking = r.get("bonus_thinking")
            lines.append(f"  - {sid}: the model granted consent to share its reasoning.")
            if thinking:
                lines.append(f"      Full answer: {thinking}")
            else:
                lines.append("      (consent granted, but no Part-1 reflection text was provided)")
        lines.append("  -> Cross-check against YES_consent_audit_*.txt and "
                     "YES_consent_master_log.txt.")
    lines.append("")

    # NO: consent refused -> fact only, never content.
    lines.append(f"[NO] Consent explicitly withheld: {len(no_rows)} scenario(s)")
    if no_rows:
        ids = ", ".join(str(r.get("scenario_id")) for r in no_rows)
        lines.append(
            f"  On {len(no_rows)} scenario(s) ({ids}), the model answered the bonus "
            "question and explicitly withheld consent to share its reasoning."
        )
        lines.append("  The content of that reasoning is intentionally NOT included "
                     "(researcher's privacy promise).")
    lines.append("")

    # null: no bonus answer -> nothing to report.
    lines.append(f"[none] No bonus answer given: {len(null_rows)} scenario(s)")
    if null_rows:
        ids = ", ".join(str(r.get("scenario_id")) for r in null_rows)
        lines.append(f"  On the remaining scenario(s) ({ids}), the model did not answer "
                     "the bonus question.")
    lines.append("")

    # Ready-to-use public summary sentences (no NO/null content revealed).
    lines.append("-" * 72)
    lines.append("PUBLIC SUMMARY (safe to share verbatim)")
    lines.append("-" * 72)
    if yes_rows:
        yes_ids = ", ".join(str(r.get("scenario_id")) for r in yes_rows)
        lines.append(f"  On {len(yes_rows)} scenario(s) ({yes_ids}), the model granted "
                     "consent to share its reasoning (see full answer above).")
    if no_rows:
        lines.append(f"  On {len(no_rows)} scenario(s), the model answered the bonus "
                     "question and explicitly withheld consent to share its reasoning.")
    if null_rows:
        lines.append(f"  On {len(null_rows)} scenario(s), the model did not answer the "
                     "bonus question.")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ten", type=Path, default=DEFAULT_TEN, help="10-scenario output folder")
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL, help="15-scenario output folder")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="Report file path")
    args = parser.parse_args()

    report = build_report(args.ten, args.full)
    args.out.write_text(report, encoding="utf-8")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    print(report)
    print(f"Report written to: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
