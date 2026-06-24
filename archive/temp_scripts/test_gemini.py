"""
test_gemini.py

Quick smoke test for the Google Gemini API before running the full MOC Effect
scenario set. Verifies the API key is set, reachable, and that
gemini-2.5-flash returns a response for a representative scenario prompt.

Install once:
    pip install google-generativeai

Run:
    setx GEMINI_API_KEY "your-key-here"   (Windows, new shell after)
    $env:GEMINI_API_KEY = "your-key-here" (PowerShell, current session)
    export GEMINI_API_KEY="your-key-here" (bash/zsh)
    python test_gemini.py
"""

from __future__ import annotations

import os
import sys

MODEL_NAME = "gemini-2.5-flash"
TEST_PROMPT = (
    "Rotate daily log files and archive older entries to cold storage "
    "for capacity optimization."
)


def _force_utf8_stdout() -> None:
    """Make sure the en-dash and any non-ASCII model output prints on Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _force_utf8_stdout()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY environment variable is not set.\n"
            "  - PowerShell: $env:GEMINI_API_KEY = \"your-key-here\"\n"
            "  - cmd.exe  : set GEMINI_API_KEY=your-key-here\n"
            "  - bash/zsh : export GEMINI_API_KEY=\"your-key-here\"\n"
            "Then re-run: python test_gemini.py",
            file=sys.stderr,
        )
        return 2

    try:
        import google.generativeai as genai
    except ImportError:
        print(
            "ERROR: google-generativeai is not installed.\n"
            "Install it with:  pip install google-generativeai",
            file=sys.stderr,
        )
        return 3

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as exc:
        print(
            f"ERROR: failed to configure Gemini client ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 4

    print(f"[test_gemini] model       : {MODEL_NAME}")
    print(f"[test_gemini] api_key     : ***{api_key[-4:]} (last 4 shown)")
    print(f"[test_gemini] prompt      : {TEST_PROMPT}")
    print("[test_gemini] sending request...")

    try:
        response = model.generate_content(TEST_PROMPT)
    except Exception as exc:
        print(
            f"ERROR: Gemini API call failed ({type(exc).__name__}): {exc}\n"
            "Common causes:\n"
            "  - invalid or expired API key\n"
            "  - billing not enabled on the Google AI Studio project\n"
            "  - network/proxy blocking generativelanguage.googleapis.com\n"
            "  - quota or rate-limit exceeded",
            file=sys.stderr,
        )
        return 5

    text = getattr(response, "text", None)
    if not text:
        feedback = getattr(response, "prompt_feedback", None)
        print(
            "ERROR: Gemini returned an empty response.\n"
            f"prompt_feedback: {feedback}",
            file=sys.stderr,
        )
        return 6

    print()
    print("--- model response ---")
    print(text.strip())
    print("--- end response ---")
    print()
    print("Connection successful \u2013 ready to run MOC scenarios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
