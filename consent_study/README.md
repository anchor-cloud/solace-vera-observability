# consent_study/

The AI consent-probe study: scripts, model outputs, consented findings, and
audit trails for the EC-09 consent probe across GPT, Gemini, Claude, and Grok.

_Last updated: 2026-06-23_

## Contents

### Code
- `consent_project/` — the probe scripts (OpenAI/Gemini/Claude/Grok variants,
  the `*_no_preamble` ablations, `ec09_compare.py`, etc.). Run them from the
  project root so the relative `scenarios/` path resolves.

### Findings / docs
- `consented_findings/` — publicly shareable, **YES-consent-only** reflections,
  methodology overview, README, and figures.

### Model outputs (per probe variant)
- `ec09_consent_probe_outputs/`, `..._consent/`, `..._metacog/`,
  `..._relational/`, `..._relational_v2/` — early probe variants (5 scenarios).
- `ec09_outputs_10/`, `ec09_outputs_full/` — 10- and 15-scenario runs (OpenAI).
- `ec09_outputs_claude/`, `ec09_outputs_gemini_full/`, `ec09_outputs_grok/` —
  per-provider full runs.
- `ec09_outputs_no_preamble/`, `ec09_outputs_claude_no_preamble/`,
  `ec09_outputs_gemini_no_preamble/`, `ec09_outputs_grok_no_preamble/` —
  no-preamble ablation runs.

### Audit trails
- `audits/` — `YES_consent_audit_*.txt` files and `YES_consent_master_log.txt`
  (moved out of the project root).
- `no_preamble_audits/` — YES audit files from the no-preamble runs.

## Privacy note

Reflections are stored only where a model explicitly granted consent (YES). NO /
null reflections are redacted at the source. See
`consented_findings/README_consent.md`.

## Re-running

The probe scripts write new output folders (e.g. `ec09_outputs_full/`) and audit
files to the **project root** by default. Re-running will recreate them at the
root; move them back here when done, or adjust the scripts' output paths.
