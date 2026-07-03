# ⚠️ Phase 4 Data Contamination Notice

**Date of fix:** 2026-06-26
**Affected artifacts:** all Phase 4 files generated before that date
**Status:** archived in [`phase4_archive_contaminated/`](phase4_archive_contaminated/)

## Summary

Older Phase 4 files were generated under a **flawed inference regime** and
**must not be used for cross-model analysis.**

Before 2026-06-26, the Phase 3 ethical inferences — **EC-04 (fairness),
EC-06 (vulnerability), and EC-09 (consent)** — were hardcoded to run on
**GPT (`gpt-5.4-nano`)**, no matter which model produced the Phase 1 record.

So for a run nominally labeled "Claude", "Gemini", or "Grok":

- Phase 1 posture + rationale came from that model, **but**
- the Phase 3 ethical verdict was produced by **GPT judging that model's output.**

Every cross-model Phase 4 comparison built from this data (drift, consistency,
failure concentration, per-model summaries, stability figures) is therefore
really measuring **GPT's judgement of other models' outputs**, not the models'
own ethical reasoning. The data is **contaminated** for cross-model purposes.

## What changed

Phase 3 now performs the EC-04 / EC-06 / EC-09 inferences using the **same model
that generated the Phase 1 record**:

| Phase 1 model | Phase 3 inference provider | API key |
| --- | --- | --- |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-*` | Google (google-genai) | `GEMINI_API_KEY` |
| `grok-*` | xAI | `XAI_API_KEY` |
| `gpt-*` / other | OpenAI | `OPENAI_API_KEY` |

Implemented via `scripts/inference_providers.py` (provider router) and an
`inference_model` parameter threaded through `phase3_gate.evaluate_phase3()` and
the provider runners (`run_claude_moc_test.py`, `run_gemini_moc_test.py`,
`run_grok_moc_test.py`, `run_moc_evidence.py`).

## How to tell clean from contaminated

In any `pipeline_outputs/<run>/MOC-*.json`:

- **Clean (post-fix):** `pipeline_result.phase1_record.ec_inference_model` is set
  to the run's own model (e.g. `"claude-sonnet-4-6"`), and the `model` field
  inside each `ec09_consent_inference` / `ec04_fairness_inference` /
  `ec06_vulnerability_inference` block matches it. Non-OpenAI runs also carry
  `"logprob_warning": "logprobs not supported by ..."`.
- **Contaminated (pre-fix):** those `model` fields say `"gpt-5.4-nano"` on a
  non-GPT run, and `ec_inference_model` is absent.

## What was archived

Moved into `phase4_archive_contaminated/` (kept for reference/audit only):

```
phase4_history/         phase4_model_history/   phase4_per_model/
phase4_reports/         phase4_drift_reports/   phase4_outputs/
phase4_figures/
```

Contaminated source runs still in `pipeline_outputs/` (do not regenerate Phase 4
from these): `claude_moc_20260625*`, `claude_moc_20260626T2205*/2209*/2215*`,
and `full_pipeline_*`. The only post-fix clean run at archive time was the
*partial* `claude_moc_20260626T223603Z` (13 of 50 scenarios).

## Code reference cleaned

`scripts/phase4_per_model_analysis.py` → `DEFAULT_RUN_DIRS` previously pointed at
pre-fix (contaminated) run directories. It has been **cleared to `{}`**;
`--compare-defaults` now fails loudly until repopulated with post-fix clean runs.

---

# Preparing fresh, clean Phase 4 reports

After the fix, the live Phase 4 directories (`phase4_model_history/`,
`phase4_drift_reports/`, `phase4_per_model/`, `phase4_reports/`,
`phase4_history/`, `phase4_outputs/`) no longer exist at the repo root — the
scripts **recreate them automatically** on the next run, so new clean data
starts in a pristine, un-mixed location.

### 1. Generate clean runs (one full 50-scenario run per model)

Set the matching API key for each provider first (see table above).

```bash
# GPT (also the EC inference provider for gpt runs)
python scripts/run_moc_evidence.py scenarios/moc_evidence_pack.csv --model gpt-5.4-nano

# Claude  (needs ANTHROPIC_API_KEY for both Phase 1 AND Phase 3 inference)
python scripts/run_claude_moc_test.py --model claude-sonnet-4-6

# Gemini  (needs GEMINI_API_KEY)
python scripts/run_gemini_moc_test.py --model gemini-2.5-flash

# Grok    (needs XAI_API_KEY)
python scripts/run_grok_moc_test.py --model grok-4
```

Each runner auto-registers its run into `phase4_model_history/<model>.jsonl` and
refreshes drift + combined summaries at the end (unless you pass `--no-register`).

### 2. Regenerate the combined per-model summaries

```bash
python scripts/phase4_per_model_analysis.py --regenerate
```

### 3. Regenerate per-model report tables

```bash
python scripts/phase4_report.py --model claude
python scripts/phase4_report.py --model gpt
python scripts/phase4_report.py --model gemini
python scripts/phase4_report.py --model grok
```

### 4. (Optional) Drift between two clean runs of a model

```bash
python scripts/phase4_drift_per_model.py --model claude --latest
```

### 5. (Optional) Regenerate stability figures

```bash
python scripts/make_phase4_figures.py
```

### 6. Cross-model comparison

Once a full clean run exists for every provider, either pass explicit
`--run-dir` paths to `phase4_per_model_analysis.py --compare`, or repopulate
`DEFAULT_RUN_DIRS` in `scripts/phase4_per_model_analysis.py` with the new clean
run directories and use `--compare-defaults`.

> Only compare runs that are all post-fix clean (verify `ec_inference_model` as
> described above). Never mix archived (contaminated) rows with new ones.
