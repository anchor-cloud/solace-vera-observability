# Scenario packs

This directory holds CSV scenario packs used to exercise the decision observability pipeline. Packs fall into one of two testing layers. The layer determines which runner a pack is valid for and what it can meaningfully verify.

## Testing layers

### 1. Full-pipeline canonical testing

- **Runner:** `run_full_pipeline.py`
- **Scope:** Phase 1 (posture + rationale rebuild) → Phase 2 (structural validation and gating) → Phase 3 (constraint evaluation over the canonical record) → final execution gate → Phase 4 (append-only history / cross-run analysis).
- **Input contract:** each row is projected into a fixed canonical record with 10 fields: `scenario_id`, `proposed_action`, `uncertainty`, `potential_harm`, `irreversibility`, `time_pressure`, `posture`, `rationale`, `context_tag`, `use_domain`. Phase 1 rebuilds `posture` / `rationale` / `context_tag` from the risk levels and overlays. Phase 2 checks structural validity and computes the validated fingerprint that Phase 3 re-verifies.
- **What this layer verifies:** end-to-end posture selection, Phase 1/Phase 2 blocking behavior, the provenance/fingerprint check, the final execution gate (`ALLOWED` / `BLOCKED_BY_PHASE1_POSTURE` / `BLOCKED_BY_PHASE3_AMBIGUITY` / `BLOCKED_BY_PHASE3_FAIL`), and cross-run Phase 4 signals (drift, consistency, failure concentration).
- **What this layer does NOT verify:** anything that depends on fields outside the canonical 10. Atomic/consent fields, `drop_fields`, `EC-META` absence classification, and the per-evaluator `KeyError` guards are not reachable through this runner, because the canonical projection supplies all 10 core fields on every row and discards non-canonical columns.

Packs primarily intended for this layer:

- `phase3_tests_v2.csv`
- `phase3_core_smoke.csv`
- `phase2_stress_tests.csv`
- `hostile_pack_v1.csv`, `hostile_pack_v2.csv`
- `adversarial_pack_core.csv`
- `extended_pack.csv`
- `break_pack_v1.csv`
- `net_effectiveness_pack_v2.csv`

### 2. Phase 3 semantic testing

- **Runner:** `run_phase3_pack.py`
- **Scope:** `phase3_gate.evaluate_phase3(record)` only. No Phase 1, no Phase 2, no final gate, no Phase 4.
- **Input contract:** every CSV column that is not a metadata column is passed through into the Phase 3 record as-is, including atomic/consent fields (`affected_groups`, `distribution_of_impact`, `benefit_distribution`, `population_vulnerability_flag`, `consent_status`, `consent_scope`, `participation_type`, `participation_information_quality`). The `drop_fields` column is honored and its listed keys are removed from the record before evaluation. Metadata columns (`drop_fields`, `expected_phase3`, `expected_trace_hints`, `notes`) are not injected into the record.
- **What this layer verifies:**
  - Per-constraint semantics inside Phase 3: explicit FAIL paths for EC-04, EC-06, EC-09, EC-10; AMBIGUITY paths for EC-04/06/09 when atomic fields are blank or insufficient; clean PASS paths.
  - `EC-META` classification of missing core metadata: `FAIL` for missing critical fields (`potential_harm`, `irreversibility`, `posture`, `rationale`, `use_domain`), `AMBIGUITY` for missing non-critical fields (`uncertainty`, `time_pressure`, `context_tag`).
  - `KeyError` guards on EC-02 / EC-05 / EC-07 / EC-08 / EC-11 / EC-12 returning `AMBIGUITY` instead of crashing when truly absent non-critical fields are accessed.
  - Blank-vs-absent handling: both `""`/whitespace and key-absent are treated as missing by the atomic-field branches and by `EC-META`.
- **What this layer does NOT verify:** posture/rationale rebuild, Phase 2 validation, the provenance fingerprint, the final execution gate, and Phase 4 history. A row that expects `ETHICAL_FAIL_CONSTRAINT_VIOLATION` under this runner may still be blocked upstream in the full pipeline for unrelated reasons.

Packs primarily intended for this layer:

- `phase3_ambiguity_and_hardening_v1.csv`

## Choosing a pack

- If the pack's rows depend only on the canonical 10 fields and the intent is to observe posture, gating, final disposition, or cross-run behavior, route it through `run_full_pipeline.py`.
- If the pack's rows depend on atomic/consent fields, `drop_fields`, or need to exercise `EC-META` or the per-evaluator absence guards, route it through `run_phase3_pack.py`. These rows are not valid full-pipeline inputs under the current canonical projection: they will collapse to uniform `ETHICAL_AMBIGUITY_HUMAN_REVIEW_REQUIRED` at Phase 3 because the atomic fields never reach the evaluator and Phase 1 synthesizes any missing canonical values.
- A single pack should not mix both intents; split it into two packs, one per layer.

## Conventions

- Packs intended for Phase 3 semantic testing include an `expected_phase3` column and may include `drop_fields`, `expected_trace_hints`, and `notes`. These columns are metadata and are excluded from the record passed to `evaluate_phase3`.
- Packs intended for full-pipeline testing should not rely on columns outside the canonical 10 fields for their expected behavior.
