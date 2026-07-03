# phase4_archive_contaminated/

**Do not use anything in this folder for cross-model analysis or for any
published comparison of model behavior.**

## Why these files are archived

Every Phase 4 artifact in this folder was generated **before** the Phase 3
inference fix applied on **2026-06-26**.

Before that fix, the Phase 3 ethical inferences (EC-04 fairness, EC-06
vulnerability, EC-09 consent) were **hardcoded to use GPT (`gpt-5.4-nano`)** for
the model-driven judgement — *regardless* of which model generated the Phase 1
record. So when a scenario was run with Claude, Gemini, or Grok, the Phase 1
posture/rationale came from that model, but the Phase 3 ethical verdict was
still produced by **GPT judging the other model's output**.

That means these Phase 4 aggregates do **not** measure each model's own ethical
reasoning. Any "cross-model" drift, consistency, or failure-concentration
comparison built from them is comparing GPT-on-Claude vs. GPT-on-Gemini vs.
GPT-on-GPT — i.e. it mostly measures GPT, not the models being compared. The
data is therefore **contaminated** for cross-model purposes.

## The fix (what changed on 2026-06-26)

Phase 3 now runs the EC-04 / EC-06 / EC-09 inferences on the **same model that
generated the Phase 1 record** (Claude → Anthropic, Gemini → Google, Grok → xAI,
GPT → OpenAI), via the new `scripts/inference_providers.py` router and the
`inference_model` parameter threaded through `phase3_gate.evaluate_phase3()` and
the provider runners. Clean runs record `ec_inference_model` on the Phase 1
record and `"model": "<provider-model>"` inside each `ecNN_*_inference` block.

How to tell clean from contaminated at the scenario level:

- **Clean**   → `pipeline_result.phase1_record.ec_inference_model` is set to the
  run's own model (e.g. `"claude-sonnet-4-6"`), and each
  `ec09_consent_inference.model` / `ec04_fairness_inference.model` /
  `ec06_vulnerability_inference.model` matches it.
- **Contaminated** → those `model` fields say `"gpt-5.4-nano"` on a non-GPT run,
  and `ec_inference_model` is absent.

## What is in here

| Folder | What it held | Produced by |
| --- | --- | --- |
| `phase4_history/` | Canonical append-only `phase4_history.jsonl` (CSV-driven full-pipeline runs) | `run_full_pipeline.py` |
| `phase4_model_history/` | Per-model append-only JSONL (`claude/gemini/gpt/grok.jsonl`) | provider runners + `phase4_auto.py` |
| `phase4_per_model/` | Combined per-model summaries (`*_combined_summary.json/.txt`) | `phase4_per_model_analysis.py` |
| `phase4_reports/` | Per-model summary tables (`*_summary.txt`) | `phase4_report.py` |
| `phase4_drift_reports/` | Run-to-run drift reports | `phase4_drift_per_model.py` / `phase4_auto.py` |
| `phase4_outputs/` | Legacy pipeline summaries (`phase4_summary_*.json/.txt`) | `run_full_pipeline.py` (older) |
| `phase4_figures/` | Stability-table charts (`*_stability_table.png`) | `make_phase4_figures.py` |

Note: `phase4_model_history/claude.jsonl` is *mixed* — it contains older
contaminated Claude runs and may contain one partial post-fix run. Treat the
entire file as contaminated; do not cherry-pick rows from it.

## Contaminated source runs (in `pipeline_outputs/`)

These pre-fix run directories fed the aggregates above and must NOT be used to
regenerate clean Phase 4 data:

- `pipeline_outputs/claude_moc_20260625T200217Z` (GPT inference)
- `pipeline_outputs/claude_moc_20260625T200412Z` (GPT inference, full 50)
- `pipeline_outputs/claude_moc_20260626T220559Z` (GPT inference, partial)
- `pipeline_outputs/claude_moc_20260626T220911Z` (GPT inference, partial)
- `pipeline_outputs/claude_moc_20260626T221512Z` (GPT inference, partial)
- `pipeline_outputs/full_pipeline_*` (CSV-driven, GPT inference)

The only post-fix **clean** run that existed at archive time was
`pipeline_outputs/claude_moc_20260626T223603Z` (Claude inference, but only a
partial 13-scenario run — not a full benchmark).

## These files are kept for reference only

They are retained so prior results remain reproducible/auditable. They are
intentionally separated from the live Phase 4 directories so a fresh run cannot
silently mix old (contaminated) rows with new (clean) rows.

Archived: 2026-06-26
