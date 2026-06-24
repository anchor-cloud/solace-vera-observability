# archive/

Old, superseded, or retired material kept for reference only. Nothing here is
part of the active pipeline.

_Last updated: 2026-06-23_

## Contents

- `pipeline_outputs/` — historical pipeline runs, grouped by month (`YYYY-MM`).
  These were moved out of the top-level `pipeline_outputs/`, which now keeps only
  the most recent run.
  - `2026-04/` — 78 runs (full_pipeline / claude_moc / gemini_moc / grok_moc /
    moc_evidence / calibration_eval / governed_prompt_runs)
  - `2026-05/` — 28 runs
- `temp_scripts/` — one-off / throwaway scripts no longer in active use
  (`tmp_ec_meta_validation.py`, `test_calibration_toggle.py`, `test_gemini.py`,
  `test_grok.py`, `test_structured_prompt.py`). Moved here instead of deleted so
  they remain recoverable.

## Notes

- Safe to delete if you are confident you no longer need historical runs.
- Month buckets are derived from the timestamp in each folder name (falling back
  to last-modified date for folders without a timestamp).
