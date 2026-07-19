"""RAG answer generation: prompt, generate, validate citations, fall back safely.

Ties the prompt templates, the Claude client, and citation validation together.
Two safe-fallback paths are deterministic and never involve an unsupported
answer: no context chunks, and an empty model response. When context is present
but weak, the system prompt instructs the model to decline — that path is the
model's to take, not something we fake-detect here.

The generator operates on plain ``Chunk`` data (the caller resolves reranked
``chunk_id``s to chunks); it never sees provider SDK objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.generation.citations import Citation, parse_citations, validate_and_resolve
from src.llm.client import ClaudeClient
from src.models.chunk import Chunk
from src.prompts.templates import FALLBACK_ANSWER, SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class GeneratedAnswer:
    """The result of answering a query over retrieved context."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    invalid_citation_numbers: list[int] = field(default_factory=list)
    is_fallback: bool = False


class RagGenerator:
    """Generates a grounded, cited answer from a query and its context chunks."""

    def __init__(self, *, llm: ClaudeClient) -> None:
        self._llm = llm

    def generate(self, query: str, chunks: Sequence[Chunk]) -> GeneratedAnswer:
        """Answer ``query`` using ``chunks`` as the only source of truth.

        Raises:
            LLMError: the generation call failed (propagated from the client).
        """
        if not chunks:
            return GeneratedAnswer(text=FALLBACK_ANSWER, is_fallback=True)

        numbered = list(enumerate(chunks, start=1))
        user_prompt = build_user_prompt(query, numbered)
        answer = self._llm.generate(system=SYSTEM_PROMPT, user=user_prompt)

        if not answer.strip():
            return GeneratedAnswer(text=FALLBACK_ANSWER, is_fallback=True)

        cited = parse_citations(answer)
        citations, invalid = validate_and_resolve(cited, numbered)
        return GeneratedAnswer(
            text=answer,
            citations=citations,
            invalid_citation_numbers=invalid,
        )
