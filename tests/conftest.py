"""Shared test fixtures.

The PDF fixture is generated in-process with correct cross-reference offsets so
that no binary file needs to be committed and pypdf can extract its text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


def _make_minimal_pdf(lines: list[str]) -> bytes:
    """Build a one-page, text-extractable PDF from the given lines."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
    ]

    parts = [b"BT", b"/F1 24 Tf"]
    y = 700
    for line in lines:
        escaped = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1")
        )
        parts.append(b"1 0 0 1 72 %d Tm (%s) Tj" % (y, escaped))
        y -= 30
    parts.append(b"ET")
    stream = b"\n".join(parts)

    objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = b"%PDF-1.4\n"
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, obj)

    xref_pos = len(pdf)
    size = len(objects) + 1
    pdf += b"xref\n0 %d\n" % size
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += b"%010d 00000 n \n" % offset
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        size,
        xref_pos,
    )
    return pdf


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.txt"
    path.write_text(
        (
            "Retrieval augmented generation combines search with language models. "
            "This document is used to test ingestion and chunking behavior. "
        )
        * 20,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(_make_minimal_pdf(["Hello RAG PDF", "Second line of text"]))
    return path


@dataclass
class _FakeEmbeddings:
    float: list[list[float]]


@dataclass
class _FakeEmbedResponse:
    embeddings: _FakeEmbeddings


@dataclass
class _FakeRerankResult:
    index: int
    relevance_score: float


@dataclass
class _FakeRerankResponse:
    results: list[_FakeRerankResult]


class FakeCohereClient:
    """Stand-in for ``cohere.ClientV2``.

    Records each ``embed`` call and returns deterministic vectors whose length
    matches the requested ``output_dimension``, so tests can assert on
    input_type, batching, and dimension without any network access.
    """

    def __init__(self, *, rerank_error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self._rerank_error = rerank_error

    def embed(self, *, texts, model, input_type, embedding_types, output_dimension):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "embedding_types": list(embedding_types),
                "output_dimension": output_dimension,
            }
        )
        vectors = [[float(len(text))] * output_dimension for text in texts]
        return _FakeEmbedResponse(embeddings=_FakeEmbeddings(float=vectors))

    def rerank(self, *, model, query, documents, top_n, max_tokens_per_doc):
        self.rerank_calls.append(
            {
                "model": model,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
                "max_tokens_per_doc": max_tokens_per_doc,
            }
        )
        if self._rerank_error is not None:
            raise self._rerank_error
        # Deterministic reordering: the LAST document is most relevant, with
        # scores decreasing down the ranking, so tests can assert on reorder.
        count = len(documents)
        order = list(reversed(range(count)))
        results = [
            _FakeRerankResult(index=idx, relevance_score=(count - rank) / count)
            for rank, idx in enumerate(order)
        ]
        return _FakeRerankResponse(results=results[:top_n])


@pytest.fixture
def fake_cohere_client() -> FakeCohereClient:
    return FakeCohereClient()


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class _FakeIndex:
    """In-memory stand-in for a Pinecone index handle."""

    def __init__(self) -> None:
        self.vectors: dict[str, dict] = {}
        self.upsert_batches: list[list[dict]] = []

    def upsert(self, *, vectors) -> None:
        batch = list(vectors)
        self.upsert_batches.append(batch)
        for record in batch:
            self.vectors[record["id"]] = {
                "values": list(record["values"]),
                "metadata": dict(record.get("metadata", {})),
            }

    def query(self, *, vector, top_k, include_metadata=False):
        scored = [
            {"id": vid, "score": _dot(vector, rec["values"])}
            for vid, rec in self.vectors.items()
        ]
        scored.sort(key=lambda match: match["score"], reverse=True)
        return {"matches": scored[:top_k]}


class FakePineconeClient:
    """Stand-in for ``pinecone.Pinecone``.

    Records index creation and hands back a single in-memory index, so tests can
    assert on upsert shape, batching, and query ordering without any network.
    """

    def __init__(self, *, existing: bool = False) -> None:
        self.created: list[dict] = []
        self.index = _FakeIndex()
        self._has_index = existing

    def has_index(self, name: str) -> bool:
        return self._has_index

    def create_index(self, *, name, dimension, metric, spec) -> None:
        self.created.append(
            {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        )
        self._has_index = True

    def Index(self, name: str) -> _FakeIndex:  # noqa: N802 - mirrors SDK method name
        return self.index


@pytest.fixture
def fake_pinecone_client() -> FakePineconeClient:
    return FakePineconeClient()


@dataclass
class _FakeContentBlock:
    type: str
    text: str


@dataclass
class _FakeMessage:
    content: list


class _FakeMessages:
    def __init__(self, parent: FakeAnthropicClient) -> None:
        self._parent = parent

    def create(self, *, model, max_tokens, system, messages):
        self._parent.create_calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": list(messages),
            }
        )
        if self._parent._error is not None:
            raise self._parent._error
        return _FakeMessage(content=list(self._parent._blocks))


class FakeAnthropicClient:
    """Stand-in for ``anthropic.Anthropic``.

    Records each ``messages.create`` call and returns configurable content
    blocks, so tests can assert on the request and on text extraction without
    any network. Pass ``error`` to make the call raise.
    """

    def __init__(
        self,
        *,
        text: str = "Generated answer.",
        blocks: list | None = None,
        error: Exception | None = None,
    ) -> None:
        self.create_calls: list[dict] = []
        self._error = error
        self._blocks = (
            blocks if blocks is not None else [_FakeContentBlock("text", text)]
        )
        self.messages = _FakeMessages(self)


def make_text_block(text: str) -> _FakeContentBlock:
    return _FakeContentBlock("text", text)


def make_nontext_block(block_type: str = "thinking") -> _FakeContentBlock:
    return _FakeContentBlock(block_type, "")


@pytest.fixture
def fake_anthropic_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()
