"""Settings behavior — the blank-env-falls-back-to-default guard.

``_env_file=None`` isolates these from the developer's real .env.
"""

from __future__ import annotations

from src.config import Settings


def test_blank_rerank_model_falls_back_to_default():
    settings = Settings(_env_file=None, cohere_rerank_model="")
    assert settings.cohere_rerank_model == "rerank-v4.0-pro"


def test_whitespace_model_falls_back_to_default():
    settings = Settings(_env_file=None, pinecone_index_name="   ")
    assert settings.pinecone_index_name == "hybrid-rag-docs"


def test_explicit_model_value_is_kept():
    settings = Settings(_env_file=None, cohere_rerank_model="rerank-v3.5")
    assert settings.cohere_rerank_model == "rerank-v3.5"


def test_guard_applies_across_provider_identity_fields():
    settings = Settings(
        _env_file=None,
        cohere_embed_model="",
        pinecone_cloud="",
        pinecone_region="  ",
    )
    assert settings.cohere_embed_model == "embed-v4.0"
    assert settings.pinecone_cloud == "aws"
    assert settings.pinecone_region == "us-east-1"
