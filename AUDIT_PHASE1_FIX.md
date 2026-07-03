# Phase 1 Routing Bug Audit — June 2026

**Date:** 2026-06-30  
**Auditor:** Self-audit  
**Purpose:** Confirm whether the Phase 1 provider-routing bug contaminated any existing data.

---

## Key Finding: No Contamination

The Phase 1 provider-routing bug **contaminated nothing**.

I checked whether the hardcoded-OpenAI structured generator ever actually fired in any run, by classifying every scenario's `phase1_record.rationale` (deterministic fallback vs. a GPT-structured JSON blob) and the EC-inference model used.

| Run (figure/history sources) | Phase 1 path | EC inference model |
|---|---|---|
| `claude_moc_20260627T040931Z` | 50/50 deterministic | claude-sonnet-4-6 |
| `moc_evidence_20260627T220307Z` (GPT) | 50/50 deterministic | gpt-5.4-nano |
| `gemini_moc_20260627T224542Z` | 50/50 deterministic | gemini-2.5-flash |
| `grok_moc_20260627T234058Z` | 50/50 deterministic | grok-4 |

The structured/GPT Phase 1 path **never executed** in the cross-model runs — it fell back to the deterministic posture logic 100% of the time (no `OPENAI_API_KEY` was set during the non-GPT runs, so `_try_structured_phase1` returned `None`).

Across **all** runs on disk, exactly **1 scenario** (in a superseded earlier Gemini run) ever hit the structured path, and even that used Gemini for EC.

**Conclusion:** The Phase 1 bug I fixed is a **latent-vector fix** — it prevents future contamination if a key is ever present — but it did not corrupt any existing artifact.

The only data ever genuinely contaminated was from the **earlier** EC-inference-hardcoding bug (fixed 2026-06-26, commit `ee0c88d`), and that was already archived into `phase4_archive_contaminated/` in a prior session.

---

## Folder-by-Folder Audit

| Folder | Derived from | Affected by Phase 1 bug? | When (vs. fixes) | Status |
|---|---|---|---|---|
| `cross_model_figures/` | 4 clean 06-27 runs | No — sources 100% deterministic Phase 1, correct EC model | 06-30 | **KEEP** |
| `pipeline_outputs/crossmodel_verdicts/` | same 4 clean runs | No | 06-30 (new) | **KEEP** |
| `phase4_model_history/` | 8 clean 06-27 runs (2/model), all correct EC model | No | 06-27 | **KEEP** |
| `phase4_per_model/` | clean history above | No | 06-27 | **KEEP** |
| `phase4_drift_reports/` | within-model clean 06-27 runs | No | 06-27 | **KEEP** |
| `consent_study/` (257 files) | standalone EC-09 consent probes (not the Phase 1→3 pipeline) | No — unrelated subsystem | 06-23 | **KEEP** |
| `phase3_enrichment/` | only a `README.md` | No | 06-24 | **KEEP** |
| `model_moc013_tables/` | single-scenario (MOC-013) raw-model stability tables | No — raw model outputs, predate model-driven EC | 05-29 | **KEEP** (old, unrelated) |
| `blog_figures/` | raw-model / MOC-013 deepdive blog assets | No — predate model-driven EC | 05-27→05-29 | **KEEP** (old, unrelated) |
| `moc_comparison_figures/` | raw risk-field bars + old pipeline-outcome charts | No Phase 1 contamination; pipeline-outcome charts predate model-driven EC (stale, not contaminated) | 05-05→05-06 | **KEEP** (old; optionally regenerate outcome charts later) |
| `phase4_archive_contaminated/` | earlier EC-bug data | (already quarantined) | 06-26 | **LEAVE AS-IS** |

---

## Files Examined That Were Already Superseded or Contaminated by Earlier EC Bug

These old `pipeline_outputs/` runs are **superseded** and not referenced by any current figure/history:

**EC-contaminated (earlier bug, `ec_inference_model = none` → EC ran on GPT):**
- `claude_moc_20260625T200217Z`
- `claude_moc_20260625T200412Z`
- `claude_moc_20260626T220559Z`
- `claude_moc_20260626T220911Z`
- `claude_moc_20260626T221512Z`

**Superseded/partial but EC-clean:**
- `claude_moc_20260626T223603Z`
- `claude_moc_20260626T232706Z`

**Old GPT CSV-based runs (predate model-driven EC):**
- `full_pipeline_20260624T*` (×4 runs)

**Very old/unrelated:**
- `pipeline outputs/` (with a space, 04-01)
- `shadow_analysis_outputs/` (04-06)
- `tools/`
- `archive/`
- `archive_old_scripts/`

---

## Overall Recommendation

**Delete nothing on account of the Phase 1 bug.** The clean 06-27 cross-model set and everything derived from it is genuinely clean, and the older folders are unrelated single-model/raw-output work.

The `README.md` has **no image links** to any of these folders (only the PeerPush badge), so removals wouldn't break README images.

---

## Summary

| Question | Answer |
|---|---|
| Are the 06-27 cross-model runs clean? | ✅ Yes — 100% deterministic Phase 1, correct EC inference per model |
| Did the Phase 1 bug affect any results? | ❌ No — it never executed in any cross-model run |
| Is the EC-inference-hardcoding bug quarantined? | ✅ Yes — all contaminated data is in `phase4_archive_contaminated/` |
| Are the figures clean? | ✅ Yes — derived from 4 clean 06-27 runs |

---

**This audit confirms that the Phase 1 routing bug is a latent-vector fix only. It prevented future contamination but corrupted no existing artifacts. The clean 06-27 cross-model data is verifiably clean.**
