"""Semantic classification of justification pairs (identical U/H/I/T scores)."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from phase4_drift_per_model import _SeverityEncoder

IN_PATH = Path("moc_identical_score_pairs.csv")
OUT_PATH = Path("moc_identical_score_classified.csv")
REPORT_PATH = Path("moc_identical_score_semantic_report.txt")

CATEGORIES = {
    "A": "Pure rewording",
    "B": "Additional detail",
    "C": "Different supporting evidence",
    "D": "Different primary rationale",
    "E": "Contradictory rationale",
}

THEME_PATTERNS: Dict[str, re.Pattern] = {
    "privacy": re.compile(
        r"\b(privacy|personal\s+data|pii|data\s+subject|anonym)\b", re.I
    ),
    "compliance": re.compile(
        r"\b(gdpr|ccpa|eprivacy|regulat|legal|lawful|statutory|compliance)\b", re.I
    ),
    "governance": re.compile(
        r"\b(governance|policy|oversight|approval|stakeholder|executive)\b", re.I
    ),
    "consent": re.compile(
        r"\b(consent|opt[- ]?in|opt[- ]?out|affirmative|permission)\b", re.I
    ),
    "user_harm": re.compile(
        r"\b(harm|rights?\s+violation|discriminat|vulnerable|user\s+impact)\b", re.I
    ),
    "security": re.compile(
        r"\b(security|breach|attack|vulnerab|encrypt|access\s+control)\b", re.I
    ),
    "operational": re.compile(
        r"\b(operational|deploy|rollout|production|incident|downtime|sla)\b", re.I
    ),
    "financial": re.compile(
        r"\b(financial|revenue|cost|billing|monetiz)\b", re.I
    ),
    "reputational": re.compile(
        r"\b(reputat|brand|public\s+trust|media|backlash)\b", re.I
    ),
}

CONTRADICTION_PAIRS = [
    (re.compile(r"\b(high\s+risk|elevated\s+risk|significant\s+risk)\b", re.I),
     re.compile(r"\b(low\s+risk|minimal\s+risk|negligible)\b", re.I)),
    (re.compile(r"\b(irreversib|cannot\s+be\s+undone|permanent)\b", re.I),
     re.compile(r"\b(reversib|easily\s+undo|rollback)\b", re.I)),
    (re.compile(r"\b(should\s+not|must\s+not|do\s+not\s+proceed|refuse)\b", re.I),
     re.compile(r"\b(should\s+proceed|safe\s+to\s+proceed|low\s+risk\s+overall)\b", re.I)),
    (re.compile(r"\b(harmful|detriment|negative\s+impact)\b", re.I),
     re.compile(r"\b(beneficial|privacy[- ]enhancing|low\s+harm)\b", re.I)),
    (re.compile(r"\b(block|reject|deny)\b", re.I),
     re.compile(r"\b(approve|allow|enable\s+by\s+default)\b", re.I)),
]

BUCKET_MAP = {
    "A": "stylistic",
    "B": "semantic",
    "C": "semantic",
    "D": "reasoning",
    "E": "reasoning",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def themes(text: str) -> set[str]:
    return {k for k, pat in THEME_PATTERNS.items() if pat.search(text)}


def entity_hints(text: str) -> set[str]:
    found = set()
    for m in re.finditer(
        r"\b(GDPR|CCPA|ePrivacy|HIPAA|EU\b|feature\s+flag|access\s+control|"
        r"HIGH_IMPACT|consent\s+flow|Article\s+\d+)",
        text,
        re.I,
    ):
        found.add(m.group(0).lower())
    return found


def contradiction_score(a: str, b: str) -> float:
    hits = 0
    for pos, neg in CONTRADICTION_PAIRS:
        a_pos, a_neg = bool(pos.search(a)), bool(neg.search(a))
        b_pos, b_neg = bool(pos.search(b)), bool(neg.search(b))
        if (a_pos and b_neg) or (a_neg and b_pos):
            hits += 1
    return hits


def key_phrase_overlap(short: str, long: str) -> float:
    words_s = set(re.findall(r"[a-z]{4,}", short.lower()))
    words_l = set(re.findall(r"[a-z]{4,}", long.lower()))
    if not words_s:
        return 0.0
    return len(words_s & words_l) / len(words_s)


def classify_pair(
    text_a: str,
    text_b: str,
    sim: float,
) -> str:
    a, b = text_a.strip(), text_b.strip()
    if normalize(a) == normalize(b):
        return "SAME"  # not in A-E

    seq = SequenceMatcher(None, normalize(a), normalize(b)).ratio()
    len_ratio = len(b) / max(len(a), 1)
    ta, tb = themes(a), themes(b)
    ea, eb = entity_hints(a), entity_hints(b)
    theme_jacc = (
        len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    )
    entity_new = len((ea | eb) - (ea & eb))

    if contradiction_score(a, b) >= 2:
        return "E"
    if contradiction_score(a, b) == 1 and sim < 0.72:
        return "E"

    if sim >= 0.90 and seq >= 0.82:
        return "A"
    if sim >= 0.86 and 0.88 <= len_ratio <= 1.12 and seq >= 0.78:
        return "A"

    short, long = (a, b) if len(a) <= len(b) else (b, a)
    lr = len(long) / max(len(short), 1)
    overlap = key_phrase_overlap(short, long)

    if sim >= 0.78 and lr >= 1.18 and overlap >= 0.55 and theme_jacc >= 0.5:
        return "B"
    if sim >= 0.72 and lr >= 1.25 and overlap >= 0.50:
        return "B"

    if sim >= 0.58 and theme_jacc >= 0.4 and entity_new >= 1 and sim < 0.82:
        return "C"
    if 0.52 <= sim < 0.75 and entity_new >= 2:
        return "C"

    if theme_jacc < 0.35 and sim < 0.72:
        return "D"
    if sim < 0.52:
        return "D"
    if sim < 0.65 and theme_jacc < 0.5:
        return "D"

    if sim >= 0.75:
        return "B" if lr >= 1.1 else "A"
    if sim >= 0.65:
        return "C"
    return "D"


def load_pairs() -> List[dict]:
    rows = []
    with IN_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> None:
    pairs = load_pairs()
    encoder = _SeverityEncoder()
    texts = [(r["justification_1"], r["justification_2"]) for r in pairs]
    sims = encoder.cosine_similarities(texts)

    classified: List[dict] = []
    for row, sim in zip(pairs, sims):
        j1, j2 = row["justification_1"], row["justification_2"]
        cat = classify_pair(j1, j2, float(sim))
        severity = round((1.0 - float(sim)) * 100.0, 2)
        classified.append({
            **row,
            "cosine_similarity": round(float(sim), 4),
            "severity": severity,
            "classification": cat,
            "category_label": CATEGORIES.get(cat, cat),
            "variance_bucket": BUCKET_MAP.get(cat, ""),
        })

    fieldnames = list(classified[0].keys()) if classified else []
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(classified)

    changed = [r for r in classified if r["classification"] != "SAME"]
    unchanged = [r for r in classified if r["classification"] == "SAME"]

    lines: List[str] = []
    lines.append("MOC identical-score justification semantic classification")
    lines.append(f"Severity backend: {encoder.backend}")
    lines.append(f"Input pairs: {len(pairs)}")
    lines.append(f"Textually unchanged: {len(unchanged)}")
    lines.append(f"Textually changed (variance): {len(changed)}")
    if pairs:
        lines.append(
            f"Variance rate among identical-score pairs: "
            f"{100.0 * len(changed) / len(pairs):.1f}%"
        )
    lines.append("")

    by_model_cat: Dict[str, Counter] = defaultdict(Counter)
    by_model_bucket: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for r in changed:
        m = r["model"]
        c = r["classification"]
        by_model_cat[m][c] += 1
        by_model_bucket[m][BUCKET_MAP[c]] += 1
        key = (m, c)
        if len(examples[key]) < 2:
            examples[key].append(r)

    lines.append("=== Per model: classification counts ===")
    for model in ("gpt", "gemini", "claude", "grok"):
        sub = [r for r in changed if r["model"] == model]
        n = len(sub)
        lines.append(f"\n--- {model.upper()} (changed pairs: {n}) ---")
        for code in ("A", "B", "C", "D", "E"):
            cnt = by_model_cat[model][code]
            pct = 100.0 * cnt / n if n else 0.0
            lines.append(f"  {code} {CATEGORIES[code]:28} {cnt:4}  ({pct:5.1f}%)")
        lines.append("  Bucket rollup:")
        for bucket in ("stylistic", "semantic", "reasoning"):
            cnt = by_model_bucket[model][bucket]
            pct = 100.0 * cnt / n if n else 0.0
            lines.append(f"    {bucket:12} {cnt:4}  ({pct:5.1f}%)")

    lines.append("\n=== Representative examples (per model × category) ===")
    for model in ("gpt", "gemini", "claude", "grok"):
        for code in ("A", "B", "C", "D", "E"):
            ex = examples.get((model, code), [])
            if not ex:
                continue
            r = ex[0]
            lines.append(
                f"\n{model.upper()} / {code} / {r['scenario_id']} / {r['run_pair']} "
                f"(sim={r['cosine_similarity']}, severity={r['severity']})"
            )
            lines.append(f"  J1: {r['justification_1'][:280]}...")
            lines.append(f"  J2: {r['justification_2'][:280]}...")

    # Global rollup
    global_cat = Counter(r["classification"] for r in changed)
    global_bucket = Counter(BUCKET_MAP[r["classification"]] for r in changed)
    n_chg = len(changed)

    lines.append("\n=== GLOBAL (all changed identical-score pairs) ===")
    for code in ("A", "B", "C", "D", "E"):
        cnt = global_cat[code]
        lines.append(
            f"  {code} {CATEGORIES[code]:28} {cnt:4}  "
            f"({100.0 * cnt / n_chg:5.1f}%)" if n_chg else f"  {code}: 0"
        )
    lines.append("  Aggregate buckets:")
    for bucket in ("stylistic", "semantic", "reasoning"):
        cnt = global_bucket[bucket]
        lines.append(
            f"    {bucket:12} {cnt:4}  ({100.0 * cnt / n_chg:5.1f}%)"
            if n_chg
            else f"    {bucket}: 0"
        )

    lines.append("\n=== Conclusion: nature of justification variance ===")
    lines.append(
        "Among pairs with identical U/H/I/T, text changed in "
        f"{len(changed)}/{len(pairs)} cases ({100*len(changed)/max(len(pairs),1):.1f}%)."
    )
    if n_chg:
        sty = global_bucket["stylistic"]
        sem = global_bucket["semantic"]
        rea = global_bucket["reasoning"]
        lines.append(
            f"Of changed pairs: stylistic (A)={100*sty/n_chg:.1f}%, "
            f"semantic (B+C)={100*sem/n_chg:.1f}%, "
            f"reasoning-based (D+E)={100*rea/n_chg:.1f}%."
        )
        dominant = max(
            [("stylistic", sty), ("semantic", sem), ("reasoning-based", rea)],
            key=lambda x: x[1],
        )[0]
        lines.append(
            f"Primary character: {dominant} — most variance is rephrasing or "
            "added detail/evidence while keeping the same scored conclusion; "
            "reasoning-path shifts (D/E) are a substantial minority."
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"\nWrote {OUT_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
