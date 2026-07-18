# Hybrid Search RAG over Internal Documents

Answer questions over internal documents using hybrid retrieval (dense + BM25),
reciprocal rank fusion, reranking, and Claude-generated answers with inline
citations.

This repository is being built MVP-first, in batches. See `PROJECT_BRIEF.md` for
the goal, `ARCHITECTURE.md` for the module layout, `PIPELINE.md` for the data
flow, and `DECISIONS.md` for the record of choices made.

## Status

**v1 complete: full RAG pipeline behind an HTTP API.**

Implemented:

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
- Claude LLM transport — `claude-sonnet-5`, isolated behind `messages.create`
  (`src/llm/client.py`)
- Grounded answer generation: numbered-context prompt, citation parse/validate,
  and safe fallback (`src/prompts/templates.py`, `src/generation/`)
- End-to-end query service: retrieve → resolve → rerank → generate, with
  surfaced rerank-degrade and citation flags (`src/services/query_service.py`)
- FastAPI app: `GET /health`, `POST /query`, `POST /ingest` (multipart upload
  with filename sanitization, extension allowlist, size cap, and immediate BM25
  refresh), with boundary validation and safe error mapping (`src/api/`, `main.py`)
- Centralized settings (`src/config.py`)

Possible next steps (out of scope for v1): auth / rate limiting, async ingest for
very large files, OCR, and horizontal scaling of the in-memory BM25 index.

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

`pyproject.toml` holds the direct dependencies (with `>=` floors);
`requirements.lock` pins the full resolved tree that the project was built and
live-tested against. For a reproducible / CI install, use the lock instead:

```bash
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install -e . --no-deps   # add the local package
```

## Running checks

```bash
.venv\Scripts\python -m pytest     # Windows
.venv\Scripts\python -m ruff check .
```

## Running the API

The server builds the real providers on startup, so the provider keys must be
set in `.env` (`COHERE_API_KEY`, `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`) first.

```bash
.venv\Scripts\python -m uvicorn main:app --reload
```

- `GET /health` → `{"status": "ok"}`
- `POST /ingest` (multipart `file`, PDF/txt/md) → `{document_id, source_title, chunk_count}`
- `POST /query` with `{"question": "..."}` → answer + citations
- Interactive docs at `http://127.0.0.1:8000/docs`

```bash
# ingest a document, then ask a question
curl -F "file=@handbook.pdf" http://127.0.0.1:8000/ingest
curl -H "Content-Type: application/json" \
  -d '{"question": "what is the leave policy?"}' http://127.0.0.1:8000/query
```

## Configuration

Settings load from environment variables / `.env` (see `.env.example`). Secrets
never live in source.

## Known limitations (v1)

- Only PDF and text/Markdown are supported; no OCR or complex formats.
- PDF text extraction quality depends on the source PDF (no OCR fallback).
- Ingestion is synchronous with a ~10 MB per-file cap; very large files are not
  processed in the background.
- Single-process: the BM25 index is in memory and refreshed via an atomic swap
  after ingest — correct for one process, not shared across workers.
- No auth or rate limiting on the API yet; not hardened for hostile public load.
- On rerank failure the service degrades to fusion order and flags
  `rerank_failed`; hallucinated citations are flagged in
  `invalid_citation_numbers` and the answer is kept (no hard reject).
- Pinecone serverless upserts are eventually consistent; a query immediately
  after indexing may briefly not see the newest vectors.
