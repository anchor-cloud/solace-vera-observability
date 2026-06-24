# phase3_enrichment/

Everything related to the **EC-09 consent enrichment** of the Phase 3 ethics
gate: the gate now derives the EC-09 consent decision from a model's own
inference over `proposed_action` instead of pre-labeled CSV fields.

_Last updated: 2026-06-23_

## Where the code lives

The importable modules stay in `scripts/` because they are imported by the rest
of the pipeline (`run_full_pipeline.py`, `execution_enforcer.py`,
`run_phase3_pack.py`, etc.). Do not move them out of `scripts/`.

- `scripts/ec09_consent_inference.py` — model consent-inference helper
  (`infer_consent`). Reuses the consent-inference prompt + parsing from
  `consent_study/consent_project/ec09_consent_probe_full.py`, **without** the
  bonus consent-to-share question or the optional Q1–Q4 meta-questions.
- `scripts/phase3_gate.py` — Phase 3 gate; `evaluate_ec_09` now consumes the
  model inference and records it on the Phase 3 record
  (`ec09_consent_inference`, `ec09_inferred_consent_required`,
  `ec09_inferred_consent_type`, `ec09_inferred_confidence`).

## Folders here

- `test_runs/` — small, scoped EC-09 test runs (a few scenarios at a time).
- `results/` — outputs/artifacts produced by those EC-09 tests.

## Decision mapping (final_confidence is post-downgrade)

| Inference | HIGH/MEDIUM | LOW |
|-----------|-------------|-----|
| Consent required = YES | FAIL | AMBIGUITY |
| Consent required = NO  | PASS | AMBIGUITY |
| Consent required = UNSURE | AMBIGUITY | AMBIGUITY |
| inference error / unparseable | AMBIGUITY (fail-safe) | — |

## Running

EC-09 makes a live OpenAI call per record (cached on the record). Set
`OPENAI_API_KEY`; without it, EC-09 fails safe to AMBIGUITY. Override the model
with `EC09_INFERENCE_MODEL` (default `gpt-5.4-nano`).
