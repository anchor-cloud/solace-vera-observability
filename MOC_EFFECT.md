## 🔬 Research Finding: The MOC Effect (Midline Output Collapse)

**Definition:** The MOC Effect occurs when an LLM-based agent systematically collapses its reasoning and outputs toward a "medium," safe, or average value - avoiding extreme or high-stakes classifications even when input clearly warrants them.

**First documented:** April 2026, using the Solace-Vera decision observability pipeline

**Diagnostic question:** "Is your AI agent MOC-ing you?"

**Implications:** If LLMs systematically collapse risk assessments to medium values, they are not actually evaluating risk - they are evading evaluation. This creates a dangerous blind spot where HIGH-risk actions may be approved because the model won't label them as HIGH.

**Citation:** If you reference this finding in academic work, please cite:
- Solace-Vera Pipeline GitHub Repository (anchor-cloud/solace-vera-observability)
- First documentation date: April 2026
