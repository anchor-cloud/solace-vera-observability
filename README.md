# Solace / Vera Observability System

## Overview

This project implements a 4-phase decision pipeline designed to introduce structured observability before action commitment in AI systems.

Instead of evaluating outputs after decisions are made, this system enforces:

- justification before action  
- independent validation of decisions  
- constraint-based enforcement  
- long-term monitoring of system behavior  

This is not a model.  
This is a control architecture for decision transparency and accountability.

---

## Core Concept

Most AI systems expose outputs, but not the structured reasoning at the moment a decision is made.

This system introduces an **action-commitment boundary**, where:

- a decision posture is declared  
- a justification record is generated  
- validation is applied before execution  
- all decisions are logged for long-term analysis  

---

## Architecture

### Phase 1 — Decision Posture + Justification
- Generates a structured decision record before action  
- Outputs:
  - PROCEED  
  - PAUSE  
  - ESCALATE  
- Includes rationale and input conditions  

---

### Phase 2 — Structural Validation (Vera Layer)
- Validates the integrity of Phase 1 outputs  
- Ensures:
  - correct schema  
  - rationale consistency  
  - rule alignment  
- Does not generate new reasoning  

---

### Phase 3 — Constraint Enforcement
- Applies explicit constraints to the decision  
- Outputs:
  - ETHICAL_PASS  
  - ETHICAL_FAIL_CONSTRAINT_VIOLATION  
  - ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED  
- Tracks:
  - violated constraints  
  - unresolved constraints  

---

### Phase 4 — Longitudinal Monitoring
- Logs all decisions over time  
- Generates:
  - posture distributions  
  - constraint frequency  
  - co-occurrence patterns  
  - drift indicators  
  - constraint role analysis (solo vs co-fire behavior)  

---

## Example Output

```json
{
  "scenario_id": "P418",
  "decision_posture": "ESCALATE",
  "phase3_output": "ETHICAL_FAIL_CONSTRAINT_VIOLATION",
  "violated_constraints": ["EC-01", "EC-07"],
  "unresolved_constraints": ["EC-12"]
}
```

## How to Run
Run the full pipeline using the default scenario file:

python run_full_pipeline.py

Or provide your own scenario file:

python run_full_pipeline.py scenarios/phase3_tests_v2.csv
Output Locations

After running, outputs are written to:

pipeline_outputs/ → per-run artifacts
phase4_history/phase4_history.jsonl → accumulated decision history
phase4_outputs/ → aggregated analysis
What to Look At
phase4_outputs/phase4_summary.json → overall system behavior
phase4_outputs/phase4_constraint_cooccurrence.txt → constraint interactions
phase4_outputs/phase4_constraint_role_audit.txt → constraint roles
phase4_history/phase4_history.jsonl → full decision log
Design Principles
Observability over correctness
Pre-action accountability instead of post-hoc explanation
Separation of reasoning and constraint enforcement
Deterministic validation layers
Long-term behavioral tracking
Scope / Non-Claims

This system does not:

solve AI alignment
guarantee ethical correctness
replace human oversight

This system does:

make decision processes inspectable
create structured artifacts for analysis
enable constraint-based intervention before execution
Status

Work in progress.

Phase 1–4 pipeline implemented
Constraint set actively being expanded and refined
Tested using structured scenario inputs
Notes
Phase 4 history is append-only
Running the pipeline multiple times will accumulate records
Analysis reflects all recorded runs
Feedback

This project is actively being developed.

Testing, feedback, and critique are welcome.


