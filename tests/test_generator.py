"""RagGenerator orchestration, verified against a fake LLM client (no network)."""

from __future__ import annotations

import pytest

from src.generation.generator import RagGenerator
from src.llm.client import ClaudeClient, LLMError
from src.models.chunk import Chunk
from src.prompts.templates import FALLBACK_ANSWER, SYSTEM_PROMPT
from tests.conftest import FakeAnthropicClient


def _chunk(chunk_id: str, position: int) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source_title="src.pdf",
        position=position,
        page=position + 1,
    )


def _generator(fake: FakeAnthropicClient) -> RagGenerator:
    return RagGenerator(llm=ClaudeClient(client=fake))


def test_empty_context_falls_back_without_calling_the_model():
    fake = FakeAnthropicClient(text="should not be used")
    result = _generator(fake).generate("q", [])

    assert result.is_fallback is True
    assert result.text == FALLBACK_ANSWER
    assert fake.create_calls == []


def test_sends_system_prompt_and_numbered_context():
    fake = FakeAnthropicClient(text="Answer [1].")
    _generator(fake).generate("What is text c0?", [_chunk("c0", 0)])

    call = fake.create_calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert "[1] (src.pdf, p.1) text c0" in call["messages"][0]["content"]


def test_valid_citations_resolve_to_context_chunks():
    fake = FakeAnthropicClient(text="Combined answer [1][2].")
    result = _generator(fake).generate("q", [_chunk("c0", 0), _chunk("c1", 1)])

    assert result.is_fallback is False
    assert [(c.number, c.chunk_id) for c in result.citations] == [(1, "c0"), (2, "c1")]
    assert result.invalid_citation_numbers == []


def test_hallucinated_citation_is_flagged_and_answer_kept():
    fake = FakeAnthropicClient(text="Grounded [1] and made up [99].")
    result = _generator(fake).generate("q", [_chunk("c0", 0)])

    assert result.text == "Grounded [1] and made up [99]."  # answer kept as-is
    assert [c.chunk_id for c in result.citations] == ["c0"]
    assert result.invalid_citation_numbers == [99]


def test_empty_model_response_falls_back():
    fake = FakeAnthropicClient(text="")
    result = _generator(fake).generate("q", [_chunk("c0", 0)])

    assert result.is_fallback is True
    assert result.text == FALLBACK_ANSWER


def test_llm_error_propagates():
    fake = FakeAnthropicClient(error=RuntimeError("boom"))
    with pytest.raises(LLMError):
        _generator(fake).generate("q", [_chunk("c0", 0)])
