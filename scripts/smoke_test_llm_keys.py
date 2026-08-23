#!/usr/bin/env python3
"""Make one minimal text-generation request to OpenAI and Anthropic."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OPENAI_MODEL = os.getenv("OPENAI_SMOKE_MODEL", "gpt-5-nano")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_SMOKE_MODEL", "claude-haiku-4-5")
PROMPT = "Reply with exactly OK."


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing existing variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def safe_error(error: Exception, secrets: tuple[str, ...]) -> str:
    if isinstance(error, urllib.error.HTTPError):
        detail = error.read().decode("utf-8", errors="replace")
        message = f"HTTP {error.code}: {detail}"
    else:
        message = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def test_openai(api_key: str) -> None:
    result = post_json(
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {api_key}"},
        {
            "model": OPENAI_MODEL,
            "input": PROMPT,
            "max_output_tokens": 16,
            "store": False,
        },
    )
    if not result.get("id"):
        raise RuntimeError("OpenAI returned no response ID")
    print(f"PASS OpenAI ({result.get('model', OPENAI_MODEL)})")


def test_anthropic(api_key: str) -> None:
    result = post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": PROMPT}],
        },
    )
    if not result.get("id"):
        raise RuntimeError("Anthropic returned no message ID")
    print(f"PASS Anthropic ({result.get('model', ANTHROPIC_MODEL)})")


def main() -> int:
    load_dotenv(ROOT / ".env")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    secrets = (openai_key, anthropic_key)
    tests = (
        ("OpenAI", openai_key, test_openai),
        ("Anthropic", anthropic_key, test_anthropic),
    )
    failures = 0

    for provider, key, test in tests:
        if not key:
            print(f"FAIL {provider}: API key is missing")
            failures += 1
            continue
        try:
            test(key)
        except Exception as error:
            print(f"FAIL {provider}: {safe_error(error, secrets)}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
