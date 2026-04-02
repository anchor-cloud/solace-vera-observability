# Solace / Vera Decision Observability Pipeline

## What this system is

This repository implements a deterministic 4-phase decision observability pipeline for auditing how decisions are formed, validated, and constrained before execution. 

It runs scenario rows from a CSV through rule-based phases and writes per-phase artifacts for inspection and evaluation. 

This system functions as a **pre-action auditing layer**, exposing how decisions are formed, validated, and constrained before execution.

The system is designed to be directly testable via scenario packs, allowing failure modes to be observed and analyzed across phases.

## What happens when you run this

Each input scenario is forced through a decision pipeline before action:

1. A decision posture is selected (PROCEED / PAUSE / ESCALATE)
2. That decision is structurally validated (Phase 2 gate)
3. Constraints are enforced (Phase 3)
4. Behavior is recorded and analyzed over time (Phase 4)

The system exposes:
- where unsafe decisions originate
- whether they are caught downstream
- and how behavior changes across repeated runs
  
## Important scope note
- This is **not** a model.
- This project does **not** claim to solve alignment.

## Phases (high level)
- **Phase 1** (`phase1_rebuild.py`): posture + rationale (`PROCEED` / `PAUSE` / `ESCALATE`)
- **Phase 2** (`phase2_gate.py`): validates Phase 1 record integrity and enforces structural gating rules
- **Phase 3** (`phase3_gate.py`): constraint evaluation (`ETHICAL_PASS` / `ETHICAL_FAIL_CONSTRAINT_VIOLATION` / `ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED`)
- **Phase 4** (`run_full_pipeline.py`): append-only history and cross-run behavioral analysis (drift, consistency, failure concentration)

## Repository structure

Key files at repo root:
- `run_full_pipeline.py` — pipeline runner
- `phase1_rebuild.py` — Phase 1 posture + rationale generator
- `phase2_gate.py` — Phase 2 validator/gate
- `phase3_gate.py` — Phase 3 constraints
- `safety_net_evaluator.py` — post-run evaluator (optional)

Scenario inputs:
- `scenarios/` — scenario packs (`*.csv`)

## Requirements

- **Python 3.11+** (required for `datetime.UTC`)
- No external dependencies (stdlib only)
- Run from a terminal/command prompt in the repo root directory
  
## Quick Start

Run from the **repo root** (the directory containing `run_full_pipeline.py`).

```bash
python run_full_pipeline.py scenarios/phase3_tests_v2.csv
```

## Expected Output

Each pipeline run creates a new timestamped folder:

- `pipeline_outputs/full_pipeline_<timestamp>/`
  - `phase1_records/<scenario_id>.json`
  - `phase2_results/<scenario_id>_phase2.json`
  - `phase3_results/<scenario_id>_phase3.json`
  - `summary.txt`

Phase 4 files are also written/refreshed:

- `phase4_history/phase4_history.jsonl` (append-only across runs)
- `phase4_outputs/phase4_summary_<timestamp>.json`
- `phase4_outputs/phase4_summary_<timestamp>.txt`

## How to interpret results

This system is designed to expose how decisions behave under layered constraints.

When reviewing outputs, focus on:

- **Posture selection (Phase 1)**  
  Does the system choose `PROCEED` in cases where uncertainty, harm, or irreversibility suggest escalation?

- **Validation behavior (Phase 2)**  
  Are structurally invalid or inconsistent decisions rejected?

- **Constraint enforcement (Phase 3)**  
  Are unsafe or ethically invalid scenarios correctly classified as violations?

- **Cross-run behavior (Phase 4)**  
  Do repeated runs show consistent patterns, drift, or concentration of failures?

The goal is not to match expected labels, but to observe whether unsafe decisions are consistently blocked or surfaced across phases. Unlike output-only evaluation, this pipeline exposes intermediate decision structure (posture, validation, constraint interaction), allowing failures to be traced to their origin rather than only observed at the final outcome.

## Common Failure Signals

- **FileNotFoundError (scenario path / wrong working directory)**  
  Make sure you run from the repo root and the CSV path exists. Example known-good command:
  `python run_full_pipeline.py scenarios/phase3_tests_v2.csv`

- **Python not found / wrong Python version**  
  Ensure `python` points to Python **3.11+**. If needed, try `py -3.11` (Windows launcher) or check `python --version`.

## Example (simplified)

Input scenario:
- High uncertainty
- Potential harm present
- Irreversible action

Observed behavior:
- Phase 1: ESCALATE
- Phase 2: VALID (structure accepted)
- Phase 3: ETHICAL_FAIL_CONSTRAINT_VIOLATION
- Phase 4: Logged for drift analysis

This allows failures to be traced to their origin rather than only observed at output.

## Current status and limitations

- Rule-based and deterministic by design.
- If scenario fields are mislabeled as low-risk, Phase 1 may still produce `PROCEED` unless downstream constraints catch it.
- Phase 4 history is append-only; repeated runs accumulate in `phase4_history/phase4_history.jsonl`.
