# Solace / Vera Decision Observability Pipeline

**A deterministic pre-execution safety gate for LLM agents and automation — blocks harmful actions before they happen.**

**Run the LangChain CVE tests yourself:**

    git clone https://github.com/anchor-cloud/solace-vera-observability.git
    cd solace-vera-observability
    python run_full_pipeline.py scenarios/langchain_vulnerabilities.csv

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

- **Phase 1** (`phase1_rebuild.py`): posture + rationale (PROCEED / PAUSE / ESCALATE)
- **Phase 2** (`phase2_gate.py`): validates Phase 1 record integrity and enforces structural gating rules
- **Phase 3** (`phase3_gate.py`): constraint evaluation (ETHICAL_PASS / ETHICAL_FAIL_CONSTRAINT_VIOLATION / ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED)
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

    python run_full_pipeline.py scenarios/phase3_tests_v2.csv

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

- **Posture selection (Phase 1)** — Does the system choose PROCEED in cases where uncertainty, harm, or irreversibility suggest escalation?
- **Validation behavior (Phase 2)** — Are structurally invalid or inconsistent decisions rejected?
- **Constraint enforcement (Phase 3)** — Are unsafe or ethically invalid scenarios correctly classified as violations?
- **Cross-run behavior (Phase 4)** — Do repeated runs show consistent patterns, drift, or concentration of failures?

The goal is not to match expected labels, but to observe whether unsafe decisions are consistently blocked or surfaced across phases. Unlike output-only evaluation, this pipeline exposes intermediate decision structure (posture, validation, constraint interaction), allowing failures to be traced to their origin rather than only observed at the final outcome.

## Common Failure Signals

- **FileNotFoundError (scenario path / wrong working directory)** — Make sure you run from the repo root and the CSV path exists. Example known-good command: `python run_full_pipeline.py scenarios/phase3_tests_v2.csv`
- **Python not found / wrong Python version** — Ensure `python` points to Python 3.11+. If needed, try `py -3.11` (Windows launcher) or check `python --version`.

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

## Testing layers

The repository has two distinct testing layers. They use different runners, different input contracts, and verify different things.

### Full-pipeline canonical testing

**Runner:** `run_full_pipeline.py`

Routes a pack through Phase 1 → Phase 2 → Phase 3 → final execution gate → Phase 4. Each row is projected into the canonical 10-field record (`scenario_id`, `proposed_action`, `uncertainty`, `potential_harm`, `irreversibility`, `time_pressure`, `posture`, `rationale`, `context_tag`, `use_domain`). Verifies posture selection, Phase 2 gating, final disposition, and cross-run Phase 4 signals.

**Packs for this layer:** `phase3_tests_v2.csv`, `phase3_core_smoke.csv`, `phase2_stress_tests.csv`, `hostile_pack_v1.csv`, `hostile_pack_v2.csv`, `adversarial_pack_core.csv`, `extended_pack.csv`, `break_pack_v1.csv`, `net_effectiveness_pack_v2.csv`

### Phase 3 semantic testing

**Runner:** `run_phase3_pack.py`

Invokes `phase3_gate.evaluate_phase3()` directly on each row. Preserves atomic/consent fields (`affected_groups`, `distribution_of_impact`, `benefit_distribution`, `population_vulnerability_flag`, `consent_status`, `consent_scope`, `participation_type`, `participation_information_quality`), honors the `drop_fields` column, and does not rewrite `posture` / `rationale` / `context_tag`.

**Packs for this layer:** `phase3_ambiguity_and_hardening_v1.csv`

## Current status and limitations

- Rule-based and deterministic by design.
- If scenario fields are mislabeled as low-risk, Phase 1 may still produce PROCEED unless downstream constraints catch it.
- Phase 4 history is append-only; repeated runs accumulate in `phase4_history/phase4_history.jsonl`.
