"""Application composition: build the services once and mount the routes.

``create_app`` is a factory so tests can inject a fake-backed ``QueryService``
and exercise the HTTP layer entirely offline. In normal use it builds the real
services from settings. Provider clients are constructed lazily (no network at
startup); the only startup I/O is reading the SQLite chunk store to build the
BM25 index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import FastAPI

from src.api.observability import (
    RequestContextMiddleware,
    configure_logging,
    logger,
)
from src.api.rate_limit import RateLimiter, parse_rate_limit
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
from src.services.ingest_jobs import IngestJobStore
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
    ingest_jobs: IngestJobStore


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
    if not resolved_settings.api_key:
        raise RuntimeError(
            "API_KEY is not set. The API refuses to start unauthenticated; "
            "set API_KEY in .env (see .env.example)."
        )

    configure_logging(resolved_settings.log_level)

    app = FastAPI(title="Hybrid Search RAG", version="0.1.0")
    app.include_router(router)
    app.add_middleware(RequestContextMiddleware)
    _install_rate_limiting(app, resolved_settings)

    resolved = services or build_services(resolved_settings)
    app.state.api_key = resolved_settings.api_key
    app.state.query_service = resolved.query_service
    app.state.indexing_service = resolved.indexing_service
    app.state.hybrid = resolved.hybrid
    app.state.max_upload_bytes = resolved_settings.max_upload_bytes
    app.state.ingest_jobs = resolved.ingest_jobs
    app.state.providers_configured = all(
        (
            resolved_settings.cohere_api_key,
            resolved_settings.pinecone_api_key,
            resolved_settings.anthropic_api_key,
        )
    )

    # Background tasks die with the process, so any job still in flight from a
    # previous run is unrecoverable. Retire them now rather than let the status
    # endpoint report work that will never finish.
    stranded = resolved.ingest_jobs.fail_unfinished()
    if stranded:
        logger.warning("retired %d ingest job(s) interrupted by restart", stranded)

    _warn_if_multi_worker()
    return app


def _warn_if_multi_worker() -> None:
    """Warn when the deployment looks multi-process.

    Best-effort only: a worker spawned by ``uvicorn --workers`` cannot see the
    worker count, so this checks the environment variables the common runners
    set. It catches the usual misconfiguration, not every one.

    It matters because three pieces of state are per-process — the BM25 index,
    the rate-limit counters, and background ingest jobs. Across workers each
    holds its own copy, so limits multiply and a job submitted to one worker is
    invisible to the others.
    """
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
        raw = os.environ.get(name)
        if raw and raw.strip().isdigit() and int(raw) > 1:
            logger.warning(
                "%s=%s but BM25, rate limiting, and ingest jobs are per-process; "
                "run a single worker",
                name,
                raw,
            )
            return


def _install_rate_limiting(app: FastAPI, settings: Settings) -> None:
    """Give this app its own limiter.

    Limits are enforced by a route dependency, so ``/health`` — which does not
    declare it — is never limited, and no route can be silently skipped. State
    belongs to this app alone: building a second app does not disturb it.
    """
    app.state.rate_limiter = RateLimiter(
        parse_rate_limit(settings.rate_limit),
        enabled=settings.rate_limit_enabled,
    )


def build_services(settings: Settings) -> AppServices:
    """Wire the full pipeline from settings using real providers."""
    store = ChunkStore(settings.sqlite_path)

    def _embedder(max_retries: int) -> CohereEmbedder:
        return CohereEmbedder(
            api_key=settings.cohere_api_key,
            model=settings.cohere_embed_model,
            dimension=settings.embed_dimension,
            batch_size=settings.embed_batch_size,
            timeout=settings.cohere_timeout,
            max_retries=max_retries,
        )

    # Two embedders on purpose. The query path keeps a short retry budget and
    # degrades visibly rather than making a caller wait; ingestion runs in the
    # background where nobody is blocked, so it can afford to be patient.
    embedder = _embedder(settings.provider_max_retries)
    ingest_embedder = _embedder(settings.ingest_max_retries)
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
        max_retries=settings.provider_max_retries,
    )
    generator = RagGenerator(
        llm=ClaudeClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            timeout=settings.anthropic_timeout,
            max_retries=settings.provider_max_retries,
        )
    )
    query_service = QueryService(
        hybrid=hybrid,
        store=store,
        reranker=reranker,
        generator=generator,
        rerank_top_n=settings.rerank_top_n,
        rerank_score_threshold=settings.rerank_score_threshold,
    )
    indexing_service = IndexingService(
        store=store,
        embedder=ingest_embedder,  # patient retries; nothing is waiting on it
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return AppServices(
        query_service=query_service,
        indexing_service=indexing_service,
        hybrid=hybrid,
        ingest_jobs=IngestJobStore(settings.sqlite_path),
    )
