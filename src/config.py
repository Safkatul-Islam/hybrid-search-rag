"""Centralized configuration.

Settings load from environment variables (and an optional .env file). Only the
values needed by the current build are defined here; provider settings are added
in the batches that use them. Secrets never live in source code.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider identity settings whose blank ("") env value should fall back to the
# code default rather than override it. A blank line like ``COHERE_RERANK_MODEL=``
# in .env otherwise sends an empty string to the provider and fails only at the
# network boundary; falling back keeps a sane default fail-safe.
_BLANK_TO_DEFAULT = (
    "cohere_embed_model",
    "cohere_rerank_model",
    "pinecone_index_name",
    "pinecone_cloud",
    "pinecone_region",
    "anthropic_model",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ingestion / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Storage
    sqlite_path: str = "./data/chunks.sqlite"

    # Embeddings (Cohere). API key is optional so the app imports without it;
    # it is only required for live embedding calls.
    cohere_api_key: str | None = None
    cohere_embed_model: str = "embed-v4.0"
    embed_dimension: int = 1024
    embed_batch_size: int = 96
    cohere_timeout: float = 30.0

    # Reranking (Cohere, same key/timeout as embeddings). Model is config-driven
    # so swapping models is an env change, not a code change.
    cohere_rerank_model: str = "rerank-v4.0-pro"
    rerank_top_n: int = 8
    rerank_max_tokens_per_doc: int = 4096

    # Vector store (Pinecone). API key is optional so the app imports without it;
    # it is only required to reach the live index. The index dimension must match
    # ``embed_dimension`` above.
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "hybrid-rag-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    vector_upsert_batch_size: int = 100

    # Answer generation (Anthropic Claude). API key is optional so the app
    # imports without it; it is only required for live generation calls.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = 1024
    anthropic_timeout: float = 60.0

    # Retrieval
    bm25_top_n: int = 20
    dense_top_n: int = 20
    # Reciprocal rank fusion: k damps top-rank dominance (60 is the common
    # default); fusion_top_n is the size of the fused ranking handed downstream.
    rrf_k: int = 60
    fusion_top_n: int = 20

    # API boundary
    max_query_length: int = 2000
    max_upload_bytes: int = 10_000_000  # ~10 MB per uploaded document

    @field_validator(*_BLANK_TO_DEFAULT, mode="before")
    @classmethod
    def _blank_falls_back_to_default(cls, value: object, info) -> object:
        """A blank/whitespace env value uses the field default instead of ""."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return cls.model_fields[info.field_name].default
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
