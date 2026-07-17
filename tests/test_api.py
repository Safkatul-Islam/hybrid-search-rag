"""HTTP boundary behavior, exercised in-process with fake-backed services."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import AppServices, create_app
from src.config import Settings
from src.embeddings.embedder import CohereEmbedder
from src.generation.citations import Citation
from src.generation.generator import RagGenerator
from src.llm.client import ClaudeClient, LLMError
from src.reranking.reranker import CohereReranker
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.services.indexing_service import IndexingService
from src.services.query_service import QueryResult, QueryService
from src.storage.chunk_store import ChunkStore
from src.storage.vector_store import PineconeVectorStore


class StubQueryService:
    """Duck-typed stand-in for QueryService, to isolate the HTTP layer."""

    def __init__(self, *, result: QueryResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[str] = []

    def answer(self, query: str) -> QueryResult:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return self._result


class StubIndexingService:
    """Stand-in whose index_bytes raises, to test unexpected-failure mapping."""

    def __init__(self, error: Exception):
        self._error = error

    def index_bytes(self, content: bytes, *, filename: str):
        raise self._error

    def rebuild_bm25(self):  # pragma: no cover - not reached when index_bytes raises
        return BM25Index([])


def _query_app(stub: StubQueryService) -> TestClient:
    services = AppServices(query_service=stub, indexing_service=None, hybrid=None)
    return TestClient(create_app(services=services, settings=Settings(_env_file=None)))


# --- health + query -------------------------------------------------------


def test_health_returns_ok():
    client = _query_app(StubQueryService(result=QueryResult(answer="x")))
    assert client.get("/health").json() == {"status": "ok"}


def test_query_happy_path_returns_answer_and_citations():
    result = QueryResult(
        answer="Paris is the capital [1].",
        citations=[Citation(number=1, chunk_id="doc-0001", source_title="geo.pdf", page=2)],
        used_chunk_ids=["doc-0001"],
    )
    stub = StubQueryService(result=result)
    response = _query_app(stub).post("/query", json={"question": "capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is the capital [1]."
    assert body["citations"][0]["chunk_id"] == "doc-0001"
    assert stub.calls == ["capital of France?"]


def test_blank_question_is_rejected_422():
    stub = StubQueryService(result=QueryResult(answer="x"))
    response = _query_app(stub).post("/query", json={"question": "   "})
    assert response.status_code == 422
    assert stub.calls == []


def test_missing_question_is_rejected_422():
    response = _query_app(StubQueryService(result=QueryResult(answer="x"))).post(
        "/query", json={}
    )
    assert response.status_code == 422


def test_oversized_question_is_rejected_422():
    stub = StubQueryService(result=QueryResult(answer="x"))
    response = _query_app(stub).post("/query", json={"question": "a" * 3000})
    assert response.status_code == 422
    assert stub.calls == []


def test_generation_failure_maps_to_502_without_leaking():
    stub = StubQueryService(error=LLMError("secret provider detail req-xyz"))
    response = _query_app(stub).post("/query", json={"question": "hi"})

    assert response.status_code == 502
    assert "secret provider detail" not in response.text
    assert "req-xyz" not in response.text


# --- ingest ---------------------------------------------------------------


def _real_stack(
    tmp_path, fake_cohere, fake_pinecone, fake_anthropic, *, max_upload_bytes=10_000_000
) -> TestClient:
    store = ChunkStore(tmp_path / "chunks.sqlite")
    embedder = CohereEmbedder(client=fake_cohere, dimension=8)
    vector_store = PineconeVectorStore(client=fake_pinecone, dimension=8)
    indexing = IndexingService(
        store=store,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=200,
        chunk_overlap=40,
    )
    hybrid = HybridRetriever(
        dense=DenseRetriever(embedder=embedder, vector_store=vector_store),
        bm25=BM25Index(store.all_chunks()),
        dense_top_n=10,
        bm25_top_n=10,
        rrf_k=60,
        fusion_top_n=10,
    )
    query = QueryService(
        hybrid=hybrid,
        store=store,
        reranker=CohereReranker(client=fake_cohere),
        generator=RagGenerator(llm=ClaudeClient(client=fake_anthropic)),
        rerank_top_n=5,
    )
    services = AppServices(query_service=query, indexing_service=indexing, hybrid=hybrid)
    settings = Settings(_env_file=None, max_upload_bytes=max_upload_bytes)
    return TestClient(create_app(services=services, settings=settings))


def test_ingest_indexes_and_makes_doc_queryable(
    tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
):
    client = _real_stack(
        tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
    )
    content = b"Onboarding: new hires must complete security training in week one. " * 5

    ingest = client.post("/ingest", files={"file": ("handbook.txt", content, "text/plain")})
    assert ingest.status_code == 200
    assert ingest.json()["chunk_count"] > 0

    # BM25 was refreshed, so the freshly-ingested doc is now retrievable.
    query = client.post("/query", json={"question": "security training"})
    assert query.status_code == 200
    assert query.json()["used_chunk_ids"]


def test_ingest_sanitizes_path_traversal_filename(
    tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
):
    client = _real_stack(
        tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
    )
    response = client.post(
        "/ingest",
        files={"file": ("../../etc/evil.txt", b"harmless text content here", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["source_title"] == "evil.txt"  # no path components


def test_ingest_rejects_unsupported_type_415(
    tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
):
    client = _real_stack(
        tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
    )
    response = client.post(
        "/ingest", files={"file": ("malware.exe", b"MZ...", "application/octet-stream")}
    )
    assert response.status_code == 415


def test_ingest_rejects_oversized_413(
    tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
):
    client = _real_stack(
        tmp_path,
        fake_cohere_client,
        fake_pinecone_client,
        fake_anthropic_client,
        max_upload_bytes=50,
    )
    response = client.post(
        "/ingest", files={"file": ("big.txt", b"a" * 100, "text/plain")}
    )
    assert response.status_code == 413


def test_ingest_rejects_empty_file_400(
    tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
):
    client = _real_stack(
        tmp_path, fake_cohere_client, fake_pinecone_client, fake_anthropic_client
    )
    response = client.post(
        "/ingest", files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 400


def test_ingest_unexpected_failure_does_not_leak():
    services = AppServices(
        query_service=StubQueryService(result=QueryResult(answer="x")),
        indexing_service=StubIndexingService(RuntimeError("cohere secret req-abc")),
        hybrid=None,
    )
    # raise_server_exceptions=False so the 500 response is returned, not re-raised.
    client = TestClient(
        create_app(services=services, settings=Settings(_env_file=None)),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/ingest", files={"file": ("notes.txt", b"some text", "text/plain")}
    )
    assert response.status_code == 500
    assert "cohere secret" not in response.text
    assert "req-abc" not in response.text


def test_create_app_wires_real_services_offline(tmp_path):
    settings = Settings(
        _env_file=None,
        sqlite_path=str(tmp_path / "chunks.sqlite"),
        cohere_api_key="x",
        pinecone_api_key="x",
        anthropic_api_key="x",
    )
    app = create_app(settings=settings)
    assert isinstance(app.state.query_service, QueryService)
    assert TestClient(app).get("/health").status_code == 200
