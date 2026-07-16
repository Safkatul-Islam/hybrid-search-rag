"""ClaudeClient behavior, verified against a fake client (no network)."""

from __future__ import annotations

import pytest

from src.llm.client import ClaudeClient, LLMError
from tests.conftest import (
    FakeAnthropicClient,
    make_nontext_block,
    make_text_block,
)


def test_request_passes_through_model_tokens_system_and_user(fake_anthropic_client):
    client = ClaudeClient(client=fake_anthropic_client, model="claude-sonnet-5", max_tokens=512)
    client.generate(system="You are helpful.", user="Hello")

    call = fake_anthropic_client.create_calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 512
    assert call["system"] == "You are helpful."
    assert call["messages"] == [{"role": "user", "content": "Hello"}]


def test_extracts_text_from_a_single_block(fake_anthropic_client):
    client = ClaudeClient(client=fake_anthropic_client)
    assert client.generate(system="s", user="u") == "Generated answer."


def test_concatenates_multiple_text_blocks():
    fake = FakeAnthropicClient(blocks=[make_text_block("Hello "), make_text_block("world")])
    client = ClaudeClient(client=fake)
    assert client.generate(system="s", user="u") == "Hello world"


def test_ignores_non_text_blocks():
    fake = FakeAnthropicClient(
        blocks=[make_nontext_block("thinking"), make_text_block("Answer")]
    )
    client = ClaudeClient(client=fake)
    assert client.generate(system="s", user="u") == "Answer"


def test_empty_content_returns_empty_string():
    fake = FakeAnthropicClient(blocks=[])
    client = ClaudeClient(client=fake)
    assert client.generate(system="s", user="u") == ""


def test_provider_failure_raises_safe_llm_error_preserving_cause():
    boom = ValueError("anthropic internal detail request-id abc-123")
    fake = FakeAnthropicClient(error=boom)
    client = ClaudeClient(client=fake)

    with pytest.raises(LLMError) as excinfo:
        client.generate(system="s", user="u")

    assert "internal detail" not in str(excinfo.value)
    assert "abc-123" not in str(excinfo.value)
    assert excinfo.value.__cause__ is boom
