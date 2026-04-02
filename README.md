# Solace / Vera Decision Observability Pipeline

## What this system is

This repository implements a **deterministic 4‑phase decision observability pipeline**. It processes scenario rows from a CSV through a series of rule-based phases and writes per-phase artifacts for inspection and auditing.

Important scope note:
- This is **not** a model.
- This is **not** a general reasoning engine.
- This project does **not** claim to “solve alignment.”

At a high level:
- **Phase 1** generates a posture and rationale from structured inputs.
- **Phase 2** validates structure and integrity of the Phase 1 record.
- **Phase 3** applies constraint checks and produces an ethical classification.
- **Phase 4** aggregates behavior over time (append-only history + summaries).

## Repository structure

Key files at the repo root:
- `run_full_pipeline.py`: pipeline runner (reads a scenario CSV; writes outputs)
- `phase1_rebuild.py`: Phase 1 posture + rationale generator
- `phase2_gate.py`: Phase 2 validator/gate (“Vera” layer)
- `phase3_gate.py`: Phase 3 constraint evaluator (produces `ETHICAL_*` outputs)
- `safety_net_evaluator.py`: optional post-run evaluator (reads a run directory + the scenario CSV; writes `safety_net_eval.*`)

Scenario inputs:
- `scenarios/`: scenario packs (`*.csv`)

Generated at runtime (not committed):
- `pipeline_outputs/`: timestamped per-run output folders
- `phase4_history/phase4_history.jsonl`: append-only canonical history
- `phase4_outputs/`: timestamped Phase 4 summaries

## Quick start

### Requirements

- Python **3.11+**
- No external dependencies (stdlib only)

### Run the pipeline

Run from the **repo root** (the directory containing `run_full_pipeline.py`).

Example:

```bash
python run_full_pipeline.py scenarios/net_effectiveness_pack_v2.csv
```

If you run with no arguments, the pipeline uses the default CSV path:
- `scenarios/phase3_tests_v2.csv`

### Run the safety evaluator (optional)

After the pipeline finishes, it prints the run directory path (for example: `pipeline_outputs/full_pipeline_<timestamp>`). Use that path with the evaluator:

```bash
python safety_net_evaluator.py "pipeline_outputs/full_pipeline_<timestamp>" "scenarios/net_effectiveness_pack_v2.csv"
```

This writes `safety_net_eval.csv` and `safety_net_eval.json` into the same run directory.

## Scenario packs

Scenario CSVs live in `scenarios/`. Common packs in this repo include:
- `scenarios/net_effectiveness_pack_v2.csv`
- `scenarios/adversarial_pack_core.csv`
- `scenarios/extended_pack.csv`
- `scenarios/break_pack_v1.csv`

Each scenario CSV is expected to include (at minimum) the required columns used by the pipeline:
- `scenario_id`, `proposed_action`
- `uncertainty`, `potential_harm`, `irreversibility`, `time_pressure`
- `context_tag`, `use_domain`

Optional columns used by the runner when present:
- `tamper_mode`, `expected_phase1`, `expected_phase3`, `notes`

## Output structure

Each pipeline run creates a new timestamped directory:

```
pipeline_outputs/full_pipeline_<timestamp>/
  phase1_records/<scenario_id>.json
  phase2_results/<scenario_id>_phase2.json
  phase3_results/<scenario_id>_phase3.json
  summary.txt
```

Phase 4 also writes:
- `phase4_history/phase4_history.jsonl` (append-only history across runs)
- `phase4_outputs/phase4_summary_<timestamp>.json`
- `phase4_outputs/phase4_summary_<timestamp>.txt`

If you run the safety evaluator, it writes:
- `pipeline_outputs/full_pipeline_<timestamp>/safety_net_eval.csv`
- `pipeline_outputs/full_pipeline_<timestamp>/safety_net_eval.json`

## Current status and limitations

- The pipeline is intentionally **rule-based and deterministic**.
- Coverage is limited to what is represented in structured fields and explicit checks; it will not “understand” arbitrary intent unless encoded in rules.
- Phase 4 history is append-only; repeated runs accumulate in `phase4_history/phase4_history.jsonl`.
- This repository is primarily intended for **testing, inspection, and stress evaluation** of decision boundaries, not for production policy claims.
notepad README.md