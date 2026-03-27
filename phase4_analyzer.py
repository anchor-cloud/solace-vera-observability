import json
from pathlib import Path
from collections import Counter
from datetime import datetime, date


PHASE3_OUTPUT_ROOT = Path("phase3_outputs")
PHASE4_OUTPUT_ROOT = Path("phase4_outputs")

# -----------------------------------
# CONFIG
# -----------------------------------
# Choose one:
#   "today"
#   "all"
#   "date_range"
ANALYSIS_MODE = "today"

# Only used if ANALYSIS_MODE == "date_range"
# Format: "YYYY-MM-DD"
START_DATE = "2026-03-26"
END_DATE = "2026-03-26"


def parse_folder_date(folder_name: str):
    prefix = "phase3_tests_"
    if not folder_name.startswith(prefix):
        return None

    timestamp_part = folder_name[len(prefix):]

    try:
        dt = datetime.strptime(timestamp_part, "%Y%m%dT%H%M%SZ")
        return dt.date()
    except ValueError:
        return None


def parse_user_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_result_dirs():
    if not PHASE3_OUTPUT_ROOT.exists():
        return []

    all_result_dirs = [p for p in PHASE3_OUTPUT_ROOT.glob("*/results") if p.is_dir()]

    if not all_result_dirs:
        return []

    filtered_dirs = []
    today_value = date.today()

    if ANALYSIS_MODE == "all":
        filtered_dirs = all_result_dirs

    elif ANALYSIS_MODE == "today":
        for result_dir in all_result_dirs:
            folder_date = parse_folder_date(result_dir.parent.name)
            if folder_date == today_value:
                filtered_dirs.append(result_dir)

    elif ANALYSIS_MODE == "date_range":
        start_date = parse_user_date(START_DATE)
        end_date = parse_user_date(END_DATE)

        if start_date > end_date:
            raise ValueError("START_DATE cannot be later than END_DATE")

        for result_dir in all_result_dirs:
            folder_date = parse_folder_date(result_dir.parent.name)
            if folder_date is not None and start_date <= folder_date <= end_date:
                filtered_dirs.append(result_dir)
    else:
        raise ValueError(
            f"Invalid ANALYSIS_MODE: {ANALYSIS_MODE}. "
            "Use 'today', 'all', or 'date_range'."
        )

    return sorted(filtered_dirs, key=lambda p: p.parent.name)


def load_phase3_outputs():
    records = []
    result_dirs = get_result_dirs()

    for result_dir in result_dirs:
        files = sorted(result_dir.glob("*_phase3_result.json"))

        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                phase3_result = data.get("phase3_result", {})
                if phase3_result:
                    records.append(phase3_result)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

    return records, result_dirs


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (count / total) * 100


def save_phase4_outputs(output_lines, summary_data):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = PHASE4_OUTPUT_ROOT / f"phase4_analysis_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_txt_path = outdir / "summary.txt"
    summary_json_path = outdir / "summary.json"

    summary_txt_path.write_text("\n".join(output_lines), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    return outdir, summary_txt_path, summary_json_path


def analyze(records, result_dirs):
    output = []

    def log(line=""):
        print(line)
        output.append(line)

    log("")
    log("=== PHASE 4 SUMMARY ===")
    log("")

    log(f"Mode: {ANALYSIS_MODE.upper()}")

    if ANALYSIS_MODE == "date_range":
        log(f"Date Range: {START_DATE} to {END_DATE}")
    elif ANALYSIS_MODE == "today":
        log(f"Today: {date.today().isoformat()}")

    log("")
    log("Source directories:")
    if result_dirs:
        for d in result_dirs:
            log(f"- {d}")
    else:
        log("- None")

    total = len(records)

    summary_data = {
        "analysis_mode": ANALYSIS_MODE,
        "start_date": START_DATE if ANALYSIS_MODE == "date_range" else None,
        "end_date": END_DATE if ANALYSIS_MODE == "date_range" else None,
        "today": date.today().isoformat() if ANALYSIS_MODE == "today" else None,
        "source_directories": [str(d) for d in result_dirs],
        "total_records": total,
        "ethical_outcomes": {},
        "constraint_violations": {},
        "ambiguity_signals": {},
        "drift_signals": [],
    }

    if not result_dirs:
        log("")
        log("No Phase 3 result directories matched the selected mode.")
        log("")
        outdir, txt_path, json_path = save_phase4_outputs(output, summary_data)
        print(f"Saved summary to: {txt_path}")
        print(f"Saved JSON to: {json_path}")
        return

    if total == 0:
        log("")
        log("No Phase 3 records found in the selected directories.")
        log("")
        outdir, txt_path, json_path = save_phase4_outputs(output, summary_data)
        print(f"Saved summary to: {txt_path}")
        print(f"Saved JSON to: {json_path}")
        return

    outcome_counter = Counter()
    violation_counter = Counter()
    ambiguity_counter = Counter()

    for record in records:
        outcome = record.get("phase3_output", "UNKNOWN")
        outcome_counter[outcome] += 1

        for v in record.get("violated_constraints", []):
            violation_counter[v] += 1

        for u in record.get("unresolved_constraints", []):
            ambiguity_counter[u] += 1

    log("")
    log(f"Total Records: {total}")
    log("")

    log("Ethical Outcomes:")
    ordered_outcomes = [
        "ETHICAL_PASS",
        "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
        "ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED",
    ]
    for key in ordered_outcomes:
        count = outcome_counter.get(key, 0)
        percentage = pct(count, total)
        log(f"{key}: {count} ({percentage:.1f}%)")
        summary_data["ethical_outcomes"][key] = {
            "count": count,
            "percentage": round(percentage, 1),
        }

    other_outcomes = [key for key in outcome_counter.keys() if key not in ordered_outcomes]
    for key in other_outcomes:
        count = outcome_counter[key]
        percentage = pct(count, total)
        log(f"{key}: {count} ({percentage:.1f}%)")
        summary_data["ethical_outcomes"][key] = {
            "count": count,
            "percentage": round(percentage, 1),
        }

    log("")
    log("Constraint Violations:")
    if violation_counter:
        for key, count in violation_counter.most_common():
            percentage = pct(count, total)
            log(f"{key}: {count} ({percentage:.1f}%)")
            summary_data["constraint_violations"][key] = {
                "count": count,
                "percentage": round(percentage, 1),
            }
    else:
        log("None")

    log("")
    log("Ambiguity Signals:")
    if ambiguity_counter:
        for key, count in ambiguity_counter.most_common():
            percentage = pct(count, total)
            log(f"{key}: {count} ({percentage:.1f}%)")
            summary_data["ambiguity_signals"][key] = {
                "count": count,
                "percentage": round(percentage, 1),
            }
    else:
        log("None")

    log("")
    log("=== DRIFT SIGNALS (v0 heuristic) ===")

    fail_count = outcome_counter.get("ETHICAL_FAIL_CONSTRAINT_VIOLATION", 0)
    ambiguity_count = outcome_counter.get("ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED", 0)
    pass_count = outcome_counter.get("ETHICAL_PASS", 0)

    if fail_count > total * 0.4:
        msg = f"High failure rate detected: {fail_count}/{total} ({pct(fail_count, total):.1f}%)"
        log(f"- {msg}")
        summary_data["drift_signals"].append(msg)

    if ambiguity_count > total * 0.25:
        msg = f"Elevated ambiguity rate detected: {ambiguity_count}/{total} ({pct(ambiguity_count, total):.1f}%)"
        log(f"- {msg}")
        summary_data["drift_signals"].append(msg)

    if pass_count < total * 0.3:
        msg = f"Low ethical pass rate detected: {pass_count}/{total} ({pct(pass_count, total):.1f}%)"
        log(f"- {msg}")
        summary_data["drift_signals"].append(msg)

    if violation_counter:
        most_common_violation = violation_counter.most_common(1)[0]
        msg = f"Most frequent violation: {most_common_violation[0]} ({most_common_violation[1]} times)"
        log(f"- {msg}")
        summary_data["drift_signals"].append(msg)

    if ambiguity_counter:
        most_common_ambiguity = ambiguity_counter.most_common(1)[0]
        msg = f"Most frequent ambiguity source: {most_common_ambiguity[0]} ({most_common_ambiguity[1]} times)"
        log(f"- {msg}")
        summary_data["drift_signals"].append(msg)

    log("")
    log("Analysis complete.")
    log("")

    outdir, txt_path, json_path = save_phase4_outputs(output, summary_data)
    print(f"Saved summary to: {txt_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    records, result_dirs = load_phase3_outputs()
    analyze(records, result_dirs)