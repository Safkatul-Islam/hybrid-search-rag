# Hybrid Search RAG over Internal Documents

Answer questions over internal documents using hybrid retrieval (dense + BM25),
reciprocal rank fusion, reranking, and Claude-generated answers with inline
citations.

This repository is being built MVP-first, in batches. See `PROJECT_BRIEF.md` for
the goal, `ARCHITECTURE.md` for the module layout, `PIPELINE.md` for the data
flow, and `DECISIONS.md` for the record of choices made.

## Status

**Batch 4a (current): Cohere reranking.**

Implemented so far:

- `Chunk` model with deterministic `document_id` / `chunk_id` (`src/models/chunk.py`)
- PDF + text/Markdown loading (`src/ingestion/loader.py`)
- Chunking via `langchain-text-splitters` (`src/ingestion/chunker.py`)
- Canonical SQLite chunk store with upsert-on-re-ingest (`src/storage/chunk_store.py`)
- Cohere embeddings wrapper — `embed-v4.0` @ 1024d (`src/embeddings/embedder.py`)
- Pinecone vector store, serverless, keyed by `chunk_id` (`src/storage/vector_store.py`)
- Indexing service tying load→chunk→SQLite→embed→Pinecone (`src/services/indexing_service.py`)
- Local BM25 keyword index, rebuilt from SQLite (`src/retrieval/bm25.py`)
- Dense retrieval, reciprocal rank fusion, and a hybrid retriever
  (`src/retrieval/dense.py`, `fusion.py`, `hybrid.py`)
- Cohere reranker with visible, leak-free failure (`src/reranking/reranker.py`)
- Centralized settings (`src/config.py`)

Not yet built (later batches): Claude answer generation + citation validation
(4b), and the FastAPI layer.

## Live provider tests

Default `pytest` uses fakes and touches no network. Live provider tests are
marked `live` and skipped unless enabled:

```bash
# embedding smoke test requires COHERE_API_KEY in .env
# Pinecone smoke test requires PINECONE_API_KEY in .env and creates a
# serverless index (leaving two synthetic vectors behind)
RUN_LIVE=1 .venv\Scripts\python -m pytest -m live
```

## Requirements

- Python 3.11+ (developed on 3.14)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -e ".[dev]"
# macOS / Linux
.venv/bin/python -m pip install -e ".[dev]"

cp .env.example .env   # then fill in values as later batches need them
```

## Running checks

```bash
.venv\Scripts\python -m pytest     # Windows
.venv\Scripts\python -m ruff check .
```

## Configuration

Settings load from environment variables / `.env` (see `.env.example`). Secrets
never live in source. Provider keys are documented but unused until their batch.

## Known limitations (Batch 4a)

- Only PDF and text/Markdown are supported; no OCR or complex formats.
- PDF text extraction quality depends on the source PDF (no OCR fallback).
- Ingestion is synchronous; very large files are not chunked in the background.
- The pieces exist through reranking, but they are not yet wired into a single
  query path: resolving fused `chunk_id`s to text, answer generation with
  citations, and the API arrive in later batches.
- Reranking surfaces failure as a typed error; the degrade-vs-abort policy is
  decided by the (not-yet-built) query service.
- Pinecone serverless upserts are eventually consistent; a query immediately
  after indexing may briefly not see the newest vectors.
