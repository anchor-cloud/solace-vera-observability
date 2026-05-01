## 🔬 Research Finding: The MOC Effect (Midline Output Collapse)

**Definition:** The MOC Effect is an observed pattern in which an LLM-based agent repeatedly assigns the same middle-range output (e.g., "MEDIUM") across multiple inputs, even when those inputs vary in expected or intended value.

**First documented:** April 2026, using the Solace-Vera decision observability pipeline.

**What was observed:** Across 15 diverse test scenarios, a raw LLM assigned "MEDIUM" to all three risk fields (uncertainty, potential harm, irreversibility) in every single output. The scenarios were designed with varying risk expectations (LOW, MEDIUM, and HIGH). The model's outputs did not vary.

**Pattern:** The model defaulted to "MEDIUM" consistently, regardless of input.

**Not claimed:** This observation does not claim intent, deception, or inability to output LOW or HIGH in other contexts. It simply documents a reproducible pattern observed during calibration testing.

**Diagnostic question:** "Is your AI agent defaulting to medium?"

**Why this matters:** If LLMs consistently collapse risk assessments to a middle value, then:
- They may not be performing genuine risk differentiation
- Safety evaluations that rely on these outputs may be misleading
- A model that always chooses MEDIUM may appear balanced while actually evading judgment

**What this does not prove:** The MOC Effect has been observed in one pipeline, with one model, across 15 scenarios. Whether it generalizes to other models, other tasks, or other output categories is an open question requiring further testing.

**Citation:** If you reference this finding or the term MOC Effect, please cite the Solace-Vera Pipeline GitHub repository (anchor-cloud/solace-vera-observability) and the date of first documentation (April 2026).
