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
# end-to-end query and rerank-threshold tests require all three keys
RUN_LIVE=1 .venv\Scripts\python -m pytest -m live
```

`tests/live/test_query_live.py` and `tests/live/test_threshold_live.py` run the
whole pipeline against real providers. Each ingests its own throwaway document
rather than depending on any existing corpus, using a temporary SQLite store —
but Pinecone is shared, so **these tests leave a few synthetic vectors in the
configured index**.

They are paced apart (default 20s, see `tests/live/conftest.py`) because a
Cohere Trial key allows only 10 API calls/minute and a single test can spend
three. On a Production key, shorten it:

```bash
LIVE_TEST_DELAY_SECONDS=0 RUN_LIVE=1 .venv\Scripts\python -m pytest -m live
```

## Rerank score measurement

`scripts/measure_rerank_scores.py` runs the real retrieve → rerank prefix of the
pipeline over the indexed corpus and prints the relevance scores the API does not
expose, so `RERANK_SCORE_THRESHOLD` can be chosen from data rather than guessed.
It makes live Cohere and Pinecone calls (no LLM call, no writes to either store)
and paces itself under the Cohere Trial rate limit, so a full run takes a few
minutes. Its query set encodes ground truth for the `samples/` documents — re-run
it, and revise those queries, after any material change to the indexed corpus.
See `DECISIONS.md` #22 for the result and why the threshold ships disabled.

```bash
.venv\Scripts\python scripts\measure_rerank_scores.py
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

# coverage (src only; currently 98% of 638 statements)
.venv\Scripts\python -m pytest --cov=src --cov-report=term-missing

# dependency vulnerability scan
.venv\Scripts\python -m pip_audit
```

## Running the API

The server builds the real providers on startup, so the provider keys must be
set in `.env` (`COHERE_API_KEY`, `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`) first.

```bash
.venv\Scripts\python -m uvicorn main:app --reload
```

`API_KEY` must also be set — **the server refuses to start without it.** There is
no unauthenticated mode. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

- `GET /health` → `{"status": "ok"}` — liveness. Open, no key, never rate limited
- `GET /ready` → dependency check; `503` when anything required is unusable
- `POST /ingest` (multipart `file`, PDF/txt/md) → **`202`** `{job_id, status}`
- `GET /ingest/{job_id}` → job status and, once finished, `document_id` / `chunk_count`
- `POST /query` with `{"question": "..."}` → answer + citations
- Interactive docs at `http://127.0.0.1:8000/docs`

`/query` and `/ingest` require an `X-API-Key` header. A missing or wrong key
returns `401` with a generic message; exceeding `RATE_LIMIT` (default
`20/minute` per client IP) returns `429` with a `Retry-After` header.

**Ingestion is asynchronous.** `POST /ingest` validates the upload (filename,
type, size, non-empty) and returns `202` immediately; indexing runs in the
background. Poll `GET /ingest/{job_id}` for the outcome — `pending`, `running`,
`succeeded`, or `failed`. A document is not searchable until its job succeeds.

```bash
# submit a document
curl -H "X-API-Key: $API_KEY" \
  -F "file=@handbook.pdf" http://127.0.0.1:8000/ingest
# -> {"job_id": "3f2a...", "status": "pending", "source_title": "handbook.pdf"}

# poll until it reaches a terminal state
curl -H "X-API-Key: $API_KEY" http://127.0.0.1:8000/ingest/3f2a...

# then ask a question
curl -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "what is the leave policy?"}' http://127.0.0.1:8000/query
```

Every response carries an `X-Request-ID` header, echoed from the request when
supplied. It appears in each log line for that request, so a failure can be
traced without correlating timestamps.

In `/docs`, click **Authorize** — or add the header manually — or requests will
come back `401`.

## Configuration

Settings load from environment variables / `.env` (see `.env.example`). Secrets
never live in source.

## Known limitations (v1)

- Only PDF and text/Markdown are supported; no OCR or complex formats.
- PDF text extraction quality depends on the source PDF (no OCR fallback).
- Ingestion is asynchronous but in-process: background tasks do not survive a
  restart, so any job still running when the server stops is marked `failed`
  with an "interrupted" error at next startup. Jobs are not transactional — a
  job interrupted after vectors reach Pinecone leaves them indexed while the
  job reads `failed`. Re-ingesting is safe (`chunk_id` is content-derived, so
  writes upsert). There is a ~10 MB per-file cap and no cap on concurrent jobs
  beyond the request rate limit.
- **Run a single worker.** The BM25 index, the rate-limit counters, and ingest
  jobs are all per-process. With `--workers > 1` limits multiply, a job
  submitted to one worker is invisible to the others, and a document indexed by
  one worker is not keyword-searchable from the rest. Startup warns if
  `WEB_CONCURRENCY`/`UVICORN_WORKERS`/`GUNICORN_WORKERS` is above 1, but a
  worker cannot detect the count directly, so this is best-effort.
- Provider retries cannot bridge a per-minute account rate limit. Every SDK
  retries internally (Cohere and Anthropic default to 2 attempts) with a backoff
  measured in seconds, while a Trial-key window takes ~60s to reset. Retries
  reduce transient failures; they do not prevent rate-limit failures.
- `/ready` checks only what this process owns — SQLite, the BM25 index, and
  whether provider keys are configured. It makes no provider calls, so it will
  report ready during a provider outage.
- Authentication is a single shared API key, not per-user identity: there is no
  user model, no key rotation, and no revocation short of changing `API_KEY`.
- Rate limiting is keyed on the client IP and counted in-process. Behind a
  reverse proxy every request appears to come from the proxy unless
  `X-Forwarded-For` is handled, which this does not do; across multiple workers
  each process would keep its own counters.
- On rerank failure the service degrades to fusion order and flags
  `rerank_failed`; hallucinated citations are flagged in
  `invalid_citation_numbers` and the answer is kept (no hard reject).
- Pinecone serverless upserts are eventually consistent; a query immediately
  after indexing may briefly not see the newest vectors.
- Provider rate limits are key-tier dependent and are not handled with retries.
  A Cohere Trial key allows 10 API calls/minute and each query spends two (one
  embedding, one rerank), so sustained throughput is roughly five queries per
  minute. Beyond that the rerank call is rate-limited and the service degrades to
  fusion order with `rerank_failed` set, rather than failing the request.
- A prompt-level decline is not flagged. When the LLM judges the retrieved
  context insufficient it answers with a refusal but returns `is_fallback: false`
  and an empty `citations` list; no field distinguishes "declined" from
  "answered", so decline rate can only be inferred from empty citations.
