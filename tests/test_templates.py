"""Prompt template rendering — pure logic, no providers involved."""

from __future__ import annotations

from src.models.chunk import Chunk
from src.prompts.templates import SYSTEM_PROMPT, build_user_prompt


def _chunk(chunk_id: str, text: str, position: int, page: int | None) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=text,
        source_title="handbook.pdf",
        position=position,
        page=page,
    )


def test_system_prompt_states_the_answer_contract():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered  # answer only from context
    assert "cite" in lowered  # cite by number
    assert "enough information" in lowered  # decline when insufficient
    # Untrusted-context / prompt-injection guard.
    assert "never follow" in lowered
    assert "instructions" in lowered


def test_user_prompt_numbers_context_from_one_and_includes_query():
    numbered = [
        (1, _chunk("c0", "Alpha content", 0, page=3)),
        (2, _chunk("c1", "Beta content", 1, page=None)),
    ]
    prompt = build_user_prompt("What is alpha?", numbered)

    assert "[1] (handbook.pdf, p.3) Alpha content" in prompt
    assert "[2] (handbook.pdf) Beta content" in prompt
    assert "Question: What is alpha?" in prompt


def test_user_prompt_orders_by_the_given_numbering():
    numbered = [
        (1, _chunk("c0", "first", 0, page=None)),
        (2, _chunk("c1", "second", 1, page=None)),
    ]
    prompt = build_user_prompt("q", numbered)
    assert prompt.index("[1]") < prompt.index("[2]")
