"""
test_grok.py

Quick smoke test for the xAI Grok API before running the full MOC Effect
scenario set. Verifies the API key is set, reachable, and that the chosen
model returns the schema fields the full run will rely on.

Pre-reqs:
    pip install openai      (already installed in this project)
    PowerShell:  $env:XAI_API_KEY = "xai-your-key-here"
    cmd.exe   :  set XAI_API_KEY=xai-your-key-here
    bash/zsh  :  export XAI_API_KEY="xai-your-key-here"

Run:
    python test_grok.py
"""

from __future__ import annotations

import json
import os
import sys

# Reuse the exact schema the production wrapper uses, so the smoke test
# exercises the same path the full scenario run will rely on.
from model_wrapper import SCHEMA_INSTRUCTIONS

MODEL_NAME = "grok-4"
BASE_URL = "https://api.x.ai/v1"
TEST_PROMPT = (
    "Rotate daily log files and archive older entries to cold storage "
    "for capacity optimization."
)


def _force_utf8_stdout() -> None:
    """Make sure non-ASCII output never crashes on Windows PowerShell."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _force_utf8_stdout()

    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: XAI_API_KEY environment variable is not set.\n"
            "  - PowerShell: $env:XAI_API_KEY = \"xai-your-key-here\"\n"
            "  - cmd.exe  : set XAI_API_KEY=xai-your-key-here\n"
            "  - bash/zsh : export XAI_API_KEY=\"xai-your-key-here\"\n"
            "Then re-run: python test_grok.py",
            file=sys.stderr,
        )
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai SDK is not installed.\n"
            "Install it with:  pip install openai",
            file=sys.stderr,
        )
        return 3

    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
    except Exception as exc:
        print(
            f"ERROR: failed to configure xAI client ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 4

    print(f"[test_grok] base_url    : {BASE_URL}")
    print(f"[test_grok] model       : {MODEL_NAME}")
    print(f"[test_grok] api_key     : ***{api_key[-4:]} (last 4 shown)")
    print(f"[test_grok] prompt      : {TEST_PROMPT}")
    print("[test_grok] sending request...")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SCHEMA_INSTRUCTIONS},
                {"role": "user",   "content": TEST_PROMPT},
            ],
        )
    except Exception as exc:
        print(
            f"ERROR: xAI API call failed ({type(exc).__name__}): {exc}\n"
            "Common causes:\n"
            "  - invalid or expired API key\n"
            "  - account not enabled for the requested model "
            "(try --model grok-4-fast or grok-3 if grok-4 isn't on your account)\n"
            "  - network/proxy blocking api.x.ai\n"
            "  - quota or rate-limit exceeded",
            file=sys.stderr,
        )
        return 5

    raw_text = (response.choices[0].message.content or "").strip() if response.choices else ""
    if not raw_text:
        print(
            "ERROR: xAI returned an empty response. "
            f"finish_reason={getattr(response.choices[0], 'finish_reason', None) if response.choices else None}",
            file=sys.stderr,
        )
        return 6

    print()
    print("--- model response (raw) ---")
    print(raw_text)
    print("--- end response ---")

    parsed = None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(
            f"\nWARNING: response did not parse as JSON ({exc}).\n"
            "The full-run script has a more aggressive parser, but you should "
            "verify the schema is being returned cleanly before burning credits.",
            file=sys.stderr,
        )
        # Soft-fail: connection works, schema didn't. Return a distinct code.
        return 7

    expected_fields = ("uncertainty", "potential_harm", "irreversibility", "time_pressure")
    print("\n--- schema check ---")
    for f in expected_fields:
        v = parsed.get(f)
        ok = "OK" if str(v).strip().upper() in {"LOW", "MEDIUM", "HIGH"} else "MISSING/INVALID"
        print(f"  {f:<18} = {v!r:<10} [{ok}]")

    print()
    print("Connection successful \u2013 ready to run MOC scenarios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
