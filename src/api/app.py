"""Application composition: build the services once and mount the routes.

``create_app`` is a factory so tests can inject a fake-backed ``QueryService``
and exercise the HTTP layer entirely offline. In normal use it builds the real
services from settings. Provider clients are constructed lazily (no network at
startup); the only startup I/O is reading the SQLite chunk store to build the
BM25 index.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from src.api.routes import router
from src.config import Settings, get_settings
from src.embeddings.embedder import CohereEmbedder
from src.generation.generator import RagGenerator
from src.llm.client import ClaudeClient
from src.reranking.reranker import CohereReranker
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.services.indexing_service import IndexingService
from src.services.query_service import QueryService
from src.storage.chunk_store import ChunkStore
from src.storage.vector_store import PineconeVectorStore


@dataclass(frozen=True)
class AppServices:
    """The long-lived services shared by the routes via ``app.state``.

    ``hybrid`` is exposed so the ingest route can refresh its BM25 index after a
    document is added.
    """

    query_service: QueryService
    indexing_service: IndexingService
    hybrid: HybridRetriever


def create_app(
    *,
    services: AppServices | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        services: inject prebuilt services (tests); otherwise built from settings.
        settings: configuration; defaults to the cached global settings.
    """
    resolved_settings = settings or get_settings()
    app = FastAPI(title="Hybrid Search RAG", version="0.1.0")
    app.include_router(router)
    resolved = services or build_services(resolved_settings)
    app.state.query_service = resolved.query_service
    app.state.indexing_service = resolved.indexing_service
    app.state.hybrid = resolved.hybrid
    app.state.max_upload_bytes = resolved_settings.max_upload_bytes
    return app


def build_services(settings: Settings) -> AppServices:
    """Wire the full pipeline from settings using real providers."""
    store = ChunkStore(settings.sqlite_path)

    embedder = CohereEmbedder(
        api_key=settings.cohere_api_key,
        model=settings.cohere_embed_model,
        dimension=settings.embed_dimension,
        batch_size=settings.embed_batch_size,
        timeout=settings.cohere_timeout,
    )
    vector_store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=settings.embed_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        batch_size=settings.vector_upsert_batch_size,
    )
    hybrid = HybridRetriever(
        dense=DenseRetriever(embedder=embedder, vector_store=vector_store),
        bm25=BM25Index(store.all_chunks()),
        dense_top_n=settings.dense_top_n,
        bm25_top_n=settings.bm25_top_n,
        rrf_k=settings.rrf_k,
        fusion_top_n=settings.fusion_top_n,
    )
    reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        max_tokens_per_doc=settings.rerank_max_tokens_per_doc,
        timeout=settings.cohere_timeout,
    )
    generator = RagGenerator(
        llm=ClaudeClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            timeout=settings.anthropic_timeout,
        )
    )
    query_service = QueryService(
        hybrid=hybrid,
        store=store,
        reranker=reranker,
        generator=generator,
        rerank_top_n=settings.rerank_top_n,
    )
    indexing_service = IndexingService(
        store=store,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return AppServices(
        query_service=query_service,
        indexing_service=indexing_service,
        hybrid=hybrid,
    )
