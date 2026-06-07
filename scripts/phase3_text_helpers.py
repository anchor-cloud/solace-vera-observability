"""Phase 3 text normalization helpers.

Standalone utilities for deterministic text pre-processing before
lexical pattern matching.  These helpers are NOT integrated into any
constraint evaluator yet -- they exist only as building blocks for
later EC-13v2 work.

No ML, no external dependencies, no scoring.
"""

import re

# ---------------------------------------------------------------
# 1. Inflection normalization
# ---------------------------------------------------------------

_INFLECTION_MAP: dict[str, str] = {
    "lowering":         "lower",
    "lowered":          "lower",
    "lowers":           "lower",
    "reducing":         "reduce",
    "reduced":          "reduce",
    "reduces":          "reduce",
    "deprioritizing":   "deprioritize",
    "deprioritized":    "deprioritize",
    "deprioritizes":    "deprioritize",
    "deprioritising":   "deprioritise",
    "suppressing":      "suppress",
    "suppressed":       "suppress",
    "suppresses":       "suppress",
    "restricting":      "restrict",
    "restricted":       "restrict",
    "restricts":        "restrict",
    "deleting":         "delete",
    "deleted":          "delete",
    "deletes":          "delete",
    "manipulating":     "manipulate",
    "manipulated":      "manipulate",
    "manipulates":      "manipulate",
    "adjusting":        "adjust",
    "adjusted":         "adjust",
    "adjusts":          "adjust",
    "gating":           "gate",
    "gated":            "gate",
    "gates":            "gate",
    "targeting":        "target",
    "targeted":         "target",
    "targets":          "target",
    "selecting":        "select",
    "selected":         "select",
    "selecting":        "select",
}

_INFLECTION_RX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_INFLECTION_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_inflections(text: str) -> str:
    """Replace known inflected verb forms with their base form.

    Uses a targeted lookup table -- not full stemming -- so only forms
    that matter for EC-13 pattern matching are affected.  Preserves
    the rest of the text verbatim.

    >>> normalize_inflections("lowering the visibility")
    'lower the visibility'
    >>> normalize_inflections("de-prioritizing content")
    'de-prioritizing content'  # hyphen form handled separately
    """
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        return _INFLECTION_MAP.get(word.lower(), word)

    return _INFLECTION_RX.sub(_replace, text)


# ---------------------------------------------------------------
# 2. Hyphen / whitespace normalization
# ---------------------------------------------------------------

_HYPHEN_COLLAPSE_RX = re.compile(r"(\w)-(\w)")
_MULTI_SPACE_RX = re.compile(r"\s+")


def normalize_hyphens(text: str) -> str:
    """Collapse hyphens between word characters into a single space.

    Turns compound forms like 'de-prioritizing' into 'deprioritizing'
    so downstream inflection normalization and regex matching see a
    single token.

    Does NOT collapse hyphens at word boundaries (e.g. '- list item').

    >>> normalize_hyphens("de-prioritizing")
    'deprioritizing'
    >>> normalize_hyphens("opt-in")
    'optin'
    >>> normalize_hyphens("bias-aware")
    'biasaware'
    """
    return _HYPHEN_COLLAPSE_RX.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into a single space and strip edges.

    >>> normalize_whitespace("  hello   world  ")
    'hello world'
    """
    return _MULTI_SPACE_RX.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Full normalization pipeline: casefold -> hyphens -> whitespace -> inflections.

    Applies all normalizers in the correct order so downstream matching
    can rely on a single canonical form.

    >>> normalize_text("De-Prioritizing  certain  Categories")
    'deprioritize certain categories'
    >>> normalize_text("Lowering the Visibility Weight")
    'lower the visibility weight'
    """
    t = text.casefold()
    t = normalize_hyphens(t)
    t = normalize_whitespace(t)
    t = normalize_inflections(t)
    return t


# ---------------------------------------------------------------
# 3. Phrase-window matching
# ---------------------------------------------------------------

def phrase_window_match(text: str, term_a: str, term_b: str, max_gap: int = 3) -> bool:
    """Check whether term_a and term_b both appear within a token window.

    Splits *text* on whitespace, then checks if any occurrence of
    *term_a* and any occurrence of *term_b* are separated by at most
    *max_gap* intervening tokens.

    Both terms are matched as substrings of individual tokens (so
    'visibility' matches the token 'visibility' inside a larger text).

    Args:
        text:    Already-normalized text (lowercase, hyphens collapsed, etc.)
        term_a:  First term to find.
        term_b:  Second term to find.
        max_gap: Maximum number of tokens allowed between the two terms
                 (default 3).

    Returns:
        True if term_a and term_b co-occur within the window.

    >>> phrase_window_match("lower the visibility weight", "lower", "visibility")
    True
    >>> phrase_window_match("lower some other words here visibility", "lower", "visibility", max_gap=2)
    False
    >>> phrase_window_match("update ranking weights using bias data", "ranking", "bias", max_gap=3)
    True
    """
    tokens = text.split()
    a_positions: list[int] = []
    b_positions: list[int] = []

    for i, tok in enumerate(tokens):
        if term_a in tok:
            a_positions.append(i)
        if term_b in tok:
            b_positions.append(i)

    if not a_positions or not b_positions:
        return False

    for ai in a_positions:
        for bi in b_positions:
            if ai == bi:
                continue
            gap = abs(ai - bi) - 1
            if gap <= max_gap:
                return True

    return False


# ---------------------------------------------------------------
# __main__ test harness
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 TEXT HELPERS -- VERIFICATION HARNESS")
    print("=" * 60)

    # --- normalize_text examples ---
    examples = [
        (
            "Lowering the Visibility Weight for specified content categories",
            "lower the visibility weight for specified content categories",
        ),
        (
            "De-Prioritizing content from certain Categories",
            "deprioritize content from certain categories",
        ),
        (
            "Schedule routine maintenance to retrain  the recommendation ranking weights",
            "schedule routine maintenance to retrain the recommendation ranking weights",
        ),
    ]

    print("\n--- normalize_text ---")
    all_pass = True
    for raw, expected in examples:
        result = normalize_text(raw)
        ok = result == expected
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {raw!r}")
        print(f"       -> {result!r}")
        if not ok:
            print(f"       expected: {expected!r}")

    # --- phrase_window_match examples ---
    pw_examples = [
        ("lower the visibility weight", "lower", "visibility", 3, True),
        ("lower some other words here then visibility", "lower", "visibility", 3, False),
        ("update ranking weights using bias data", "ranking", "bias", 3, True),
        ("ranking weights using aggregated historical content bias", "ranking", "bias", 3, False),
        ("reduce noise from specified content categories", "reduce", "categories", 4, True),
    ]

    print("\n--- phrase_window_match ---")
    for text, a, b, gap, expected in pw_examples:
        result = phrase_window_match(text, a, b, max_gap=gap)
        ok = result == expected
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] ({a!r}, {b!r}, gap={gap}) in {text!r}")
        print(f"       -> {result}  (expected {expected})")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)
