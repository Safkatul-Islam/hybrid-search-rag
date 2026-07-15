"""Cohere embedding wrapper — the single place Cohere's SDK is used.

The rest of the system deals only in plain float vectors. Documents and queries
are embedded with different ``input_type`` values, as Cohere's models require
(mismatching them quietly degrades retrieval quality). Calls are batched to
respect the API's 96-text-per-request limit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import cohere

# Cohere's embed endpoint accepts at most 96 texts per request.
_MAX_BATCH = 96


class CohereEmbedder:
    """Embeds documents and queries via Cohere.

    The ``client`` argument allows injecting a fake in tests; in normal use a
    real ``cohere.ClientV2`` is created from the API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        dimension: int = 1024,
        batch_size: int = _MAX_BATCH,
        timeout: float = 30.0,
        client: object | None = None,
    ) -> None:
        if not 1 <= batch_size <= _MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {_MAX_BATCH}")
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = client or cohere.ClientV2(api_key=api_key, timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts for storage/search (``input_type="search_document"``)."""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (``input_type="search_query"``)."""
        return self._embed([text], input_type="search_query")[0]

    def _embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            response = self._client.embed(
                texts=list(batch),
                model=self._model,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self._dimension,
            )
            vectors.extend(response.embeddings.float)
        return vectors


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
