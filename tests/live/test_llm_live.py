"""Live Claude generation smoke test.

Skipped unless ``RUN_LIVE=1`` and an Anthropic API key is configured. Run with:

    RUN_LIVE=1 pytest -m live

Spends one real Claude call. Asks a trivially-answerable question and asserts a
non-empty text response.
"""

from __future__ import annotations

import os

import pytest

from src.config import Settings

pytestmark = pytest.mark.live

_RUN_LIVE = os.getenv("RUN_LIVE") == "1"


@pytest.mark.skipif(not _RUN_LIVE, reason="RUN_LIVE != 1; live tests disabled")
def test_live_generate_returns_nonempty_text():
    settings = Settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from src.llm.client import ClaudeClient

    client = ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        timeout=settings.anthropic_timeout,
    )

    answer = client.generate(
        system="Answer in one short sentence.",
        user="What is 2 + 2?",
    )

    assert isinstance(answer, str)
    assert answer.strip() != ""
