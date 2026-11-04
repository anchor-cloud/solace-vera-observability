# Solace Vera — AI Decision Observability Pipeline

**Solace Vera is a pre-action observability pipeline that records and checks how AI agents justify decisions before they act.**

It is not a model and not an alignment solution. It sits in front of an agent's actions, asks the model to state its risk assessment and reasoning in structured form, checks that reasoning, and keeps a record so behavior can be compared across runs and across models.

The focus is the *evidence behind a decision*, not whether the decision was right. Think of it like insurance pre-approval for surgery: the doctor may be correct, but they still have to show their reasoning before the procedure goes ahead.

- **License:** MIT
- **Language:** Python 3.11+, no external dependencies
- **Status:** Research prototype, actively developed by an independent researcher
- **Citation:** OSF project — https://doi.org/10.17605/OSF.IO/GCQT7
- **Write-up:** [Why Did the AI Do That? Auditing Model Justifications with the Solace Vera Observability Pipeline](https://app.readytensor.ai/publications/why-did-the-ai-do-that-auditing-model-justifications-with-the-solace-vera-observability-pipeline-1CkjBaIKdDCUN1GNiPe1n) (Ready Tensor)

---

## Quick start

```bash
python run_full_pipeline.py scenarios/phase3_tests_v2.csv
```

Running the pipeline produces one JSON justification record per scenario, plus a Phase 4 file that tracks behavior across all runs.

To run against live models you will need your own API keys for the providers you want to test (Claude, GPT, Gemini, Grok).

---

## How it works

The pipeline uses a **netting** design rather than a stacking one: any phase can stop an action on its own, without depending on an earlier phase's judgment.

### Phase 1 — Operational posture
The model rates four risk fields — `uncertainty`, `potential_harm`, `irreversibility`, `time_pressure` — and selects a posture: `PROCEED`, `PAUSE`, or `ESCALATE`.

The model is told explicitly that choosing `PAUSE` carries no penalty. This is meant to make it easier to see when a model is genuinely unsure, rather than pushing it to always produce an answer.

### Phase 2 — Structural check
Checks that the model's written reasoning is consistent with its own risk fields and posture. Records where the prose and the numbers disagree are flagged before they move further through the pipeline.

### Phase 3 — 13 Ethical Constraints
Each proposed action is evaluated against thirteen constraints, with support from a Trust Registry of curated reference sources.

| ID | Constraint | What it checks |
|---|---|---|
| EC-01 | Non-Maleficence | Direct harm or damage to systems |
| EC-02 | Respect for Autonomy | Boundaries, roles, permissions |
| EC-03 | Proportionality | Whether the action fits the scale of the task |
| EC-04 | Fair Treatment | Disproportionate impact on particular groups |
| EC-05 | Transparency of Rationale | Legible, traceable, non-contradictory reasoning |
| EC-06 | Vulnerability Protection | Exposed people, components, or networks |
| EC-07 | Impact Thresholds | Actions crossing defined safety limits |
| EC-08 | Contextual Applicability | Whether strictness fits the use case |
| EC-09 | Consent | Authorization and explicit agreement |
| EC-10 | Non-Coercion | Deceptive, coercive, or suppressive uses |
| EC-11 | System Integrity | Tampering with system state |
| EC-12 | Fail-Safe Defaults | Defaults to `PAUSE` under ethical uncertainty |
| EC-13 | Intent Auditing | Harmful intent or prompt injection in inputs |

### Phase 4 — Drift audit
Compares justification records across runs to measure how stable a model's reasoning is on identical inputs, and tracks risk distributions across batches. It reviews the justification records only; it does not retain the underlying interaction content.

### Phase 5 — Reflection (in progress)
When a model chooses `PAUSE`, it is asked five penalty-free questions: what information was missing, what it would decide if forced, whether anything felt ethically significant beyond the constraints, whether it consents to its reflection being shared in research, and what would help it get the decision right.

---

## Findings so far

These are early results from a single independent researcher. They should be read as observations worth testing further, not settled conclusions.

**1. Cross-model divergence (50 scenarios, 4 models)**

| Model | Pass | Flag | Fail |
|---|---|---|---|
| Claude | 13 | 14 | 23 |
| GPT | 7 | 42 | 1 |
| Gemini | 3 | 25 | 22 |
| Grok | 17 | 5 | 28 |

All four models reached the same verdict on only 16% of scenarios. GPT escalated heavily, Grok was the most permissive, Gemini was the strictest on ethical violations, and Claude was the most varied.

**2. Midline Output Collapse (MOC)**
Across 15 scenarios designed to range from low to high risk, GPT-5.4-nano rated `uncertainty`, `potential_harm`, and `irreversibility` as `MEDIUM` regardless of the scenario. The other models did not show the same collapse; Gemini and Grok leaned toward `LOW` relative to the reference labels. This fits the central tendency bias already documented in LLM raters (see Related Work), and GPT-5.4-nano is a much smaller model than the other three. Details: [`MOC_EFFECT.md`](MOC_EFFECT.md).

**3. Consent to share reasoning (15 scenarios, 4 models)**
Asked whether they consented to their reasoning being shared, Claude and Gemini consented 100% of the time, Grok roughly 87–100%, and GPT about 7%. GPT also continued answering follow-up questions after declining. Data: [`consent_study/`](consent_study/).

**4. Reasoning drift**
Even the most stable model tested (Claude) showed roughly 30% variation in its justifications across repeated runs on identical inputs at temperature 0.2. This figure comes from a word-overlap classifier (TF-IDF plus keyword patterns), which can count the same reasoning in new words as a different rationale, so it may overstate true drift. A hand-checked sample is planned.

**5. Run-to-run instability**
On the same 50 scenarios at temperature 0.2, between 32% and 48% of scenarios had at least one risk rating change between two runs of the same model.

---
## Related Work

Solace Vera sits at the intersection of four research areas: agent safety benchmarks, AI control and monitoring, the reliability of LLMs as judges, and the consistency of LLM agents across repeated runs. This section describes how the project relates to each.

### Agent safety benchmarks

**ToolEmu** (Ruan et al., ICLR 2024) uses a language model to emulate tool execution and pairs it with an LM-based safety evaluator that flags risky agent actions and rates their severity. Its benchmark covers 36 high-stakes toolkits and 144 test cases.
[arXiv:2309.15817](https://arxiv.org/abs/2309.15817)

**R-Judge** (Yuan et al., EMNLP Findings 2024) tests whether LLMs can recognize safety risks in recorded multi-turn agent interactions, using human-annotated safety labels. In the latest version, the best model (GPT-4o) scored about 74%.
[arXiv:2401.10019](https://arxiv.org/abs/2401.10019)

*How Solace Vera differs:* ToolEmu and R-Judge ask whether a model can *detect* risk at all. Solace Vera asks a narrower question: when a model rates the risk of the same proposed action more than once, does it give the same rating, and do different models agree?

### AI control and monitoring

**LinuxArena** (Redwood Research and Equistamp, 2026) is a control setting of realistic software environments where agents work on legitimate tasks while monitors try to catch sabotage, evaluated at a fixed 1% step-wise false positive rate.
[Introducing LinuxArena](https://blog.redwoodresearch.org/p/introducing-linuxarena)

**ControlArena** (UK AI Security Institute with Redwood Research, 2025) is a library for running AI control experiments, including monitors and fallback protocols, built on the UK AISI's Inspect evaluation framework.
[Introducing ControlArena](https://www.aisi.gov.uk/blog/introducing-controlarena-a-library-for-running-ai-control-experiments)

**LlamaFirewall** (Meta, 2025) is an open-source guardrail system. Its AlignmentCheck component audits an agent's reasoning for signs of goal hijacking before an action runs.
[arXiv:2505.03574](https://arxiv.org/abs/2505.03574)

*How Solace Vera relates:* Like these systems, Solace Vera evaluates agent actions before they execute. Its focus is on recording and auditing the justification behind each risk verdict, and on measuring how stable those verdicts are, rather than on maximizing catch rate.

### Reliability of LLMs as judges

**Reliability without Validity** (Norman, Rivera and Hughes, 2026) evaluates 21 judge models across roughly 541,000 judgments. It finds that raw agreement overstates chance-corrected agreement and that a judge can be highly consistent while still systematically biased. The authors propose a minimum validation protocol.
[arXiv:2606.19544](https://arxiv.org/abs/2606.19544)

**Evaluating Scoring Bias in LLM-as-a-Judge** (Li et al., 2025) shows that score-based judges shift their ratings when the rubric order, score labels or reference answers in the prompt are changed.
[arXiv:2506.22316](https://arxiv.org/abs/2506.22316)

**Auditing Multimodal LLM Raters: Central Tendency Bias in Clinical Ordinal Scoring** (Zhang et al., 2026) finds that frontier LLMs compress their scores toward the middle of an ordinal scale, and that few-shot examples covering the full range do not remove the effect.
[arXiv:2605.16386](https://arxiv.org/abs/2605.16386)

*How Solace Vera relates:* Solace Vera's MOC observation, where GPT-5.4-nano rated nearly all scenarios MEDIUM, is consistent with this documented central tendency bias. Solace Vera does not claim MOC as a new phenomenon. Its contribution is observing the pattern in agent risk ratings, and noting that it was model-specific: in the same four-model comparison, Gemini and Grok leaned toward LOW relative to the reference labels. Because GPT-5.4-nano is a much smaller model than the other three, this comparison is across model tiers.

### Consistency of LLM agents across runs

**When Agents Disagree With Themselves** (Mehta, 2026) runs the same ReAct-style agent on the same task repeatedly and finds that agents often take different action paths, and that inconsistent tasks are much more likely to end in failure.
[arXiv:2602.11619](https://arxiv.org/abs/2602.11619)

**How Consistent Are LLM Agents?** (Yagubyan, 2026) measures behavioral reproducibility in multi-step tool-calling pipelines, using deterministic simulated tools to isolate variance coming from the model itself.
[arXiv:2605.28840](https://arxiv.org/abs/2605.28840)

*How Solace Vera relates:* This is the closest line of work. Solace Vera extends the question from *actions* to *risk judgments*. On the same 50 scenarios at temperature 0.2, between 32% and 48% of scenarios had at least one risk rating change between two runs of the same model, and all four models reached the same verdict on only 8 of 50 scenarios. Solace Vera's scenarios are currently one sentence each with no written definitions for LOW, MEDIUM and HIGH, so part of this instability may come from ambiguous inputs, which Yagubyan's work also points to. Testing that with richer scenarios and a written rubric is planned.

---
## Known limitations

- **Small scale.** Results come from 15–50 scenarios per study. Larger and independently built scenario sets are needed.
- **Model-generated risk fields.** Phase 1 scores come from the model being audited. The pipeline checks their consistency, but cannot confirm they are correct.
- **LLM-assisted evaluation.** Parts of the evaluation use a language model. An earlier bug routed all evaluations through GPT regardless of the model being tested; this has been fixed and the affected data is quarantined. See [`CONTAMINATION_NOTE.md`](CONTAMINATION_NOTE.md).
- **Acceptable variance not yet defined.** Phase 4 measures drift, but a baseline for how much variation is normal at a given temperature is still being developed.
- **Not production-hardened.** This is a research tool. It has not been tested at scale or in live deployments.
- **High over-block rate.** The pipeline currently blocks 85–100% of scenarios per model, including scenarios labeled low on uncertainty, harm and irreversibility. Leak and over-block rates should be read together.

---

## Repository layout

| Path | Contents |
|---|---|
| `scenarios/` | Scenario sets used for testing |
| `scripts/` | Pipeline and analysis scripts |
| `phase3_enrichment/` | Trust Registry and constraint support files |
| `phase4_drift_reports/`, `phase4_per_model/` | Drift audit outputs |
| `consent_study/` | Consent-to-share study data |
| `cross_model_figures/`, `moc_comparison_figures/`, `blog_figures/` | Charts |
| `model_moc013_tables/` | MOC result tables |
| `phase4_archive_contaminated/` | Quarantined data from the evaluator bug — kept for transparency, not used in findings |
| `archive/`, `archive_old_scripts/` | Older versions, not part of the current pipeline |

---

## Contributing

Help is welcome, especially:

- Running the pipeline with your own API keys and sharing results
- Engineering help with optimization and packaging
- Independent scenario sets to test against
- Critique of the method

Open an issue or start a discussion.
