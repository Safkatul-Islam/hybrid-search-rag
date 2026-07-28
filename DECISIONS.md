# Decisions

A running record of significant choices. Each entry: the decision, why, and the
alternatives considered. Kept lightweight — updated at meaningful milestones, not
after every change.

## 1. SQLite as the canonical chunk store

**Decision:** Use SQLite as the single source of truth for chunk text/metadata.

**Reason:** Simple, zero-configuration, easy to inspect, and BM25 can rebuild
from it. Also gives citations a stable place to resolve `chunk_id → text`.

**Alternatives:** PostgreSQL (too much for MVP); storing everything in Pinecone
(poor fit for BM25 and for a source-of-truth).

## 2. SDK-first; LangChain only as a scalpel

**Decision:** Build with official provider SDKs. Use LangChain only where it
clearly saves time without hiding a seam we need to control. First and only such
use so far: the standalone `langchain-text-splitters` package for chunking.

**Reason:** The pipeline is tightly specified (RRF, chunk-id preservation,
visible rerank failures). LangChain's orchestration abstractions would hide
exactly those seams. Direct SDKs keep the core transparent and testable.

**Alternatives:** Full LangChain framework (hides retrieval/fusion internals);
hand-rolling the text splitter (reinvents proven overlap logic).

## 3. Cohere for embeddings

**Decision:** Generate embeddings with Cohere.

**Reason:** The brief lists no embedding provider (Claude does not embed, Pinecone
only stores vectors). Cohere is already in the stack for rerank, so this adds no
new provider or key. Model name/dimension will be taken from official docs.

**Alternatives:** OpenAI / Voyage embeddings (new provider + key); Pinecone
hosted inference (more coupling for the MVP).

## 4. Numbered citations

**Decision:** Claude receives a numbered list of context chunks and cites by
number (`[1]`, `[2]`); we map number → `chunk_id` and validate.

**Reason:** Deterministic to parse and validate, which the brief's citation
checks require.

**Alternatives:** Free-form or inline `chunk_id` citations (harder to parse
reliably).

## 5. Deterministic, hash-based IDs with upsert on re-ingest

**Decision:** `document_id` = hash of file bytes; `chunk_id` = `document_id` +
zero-padded position. Re-ingesting the same file upserts the same ids.

**Reason:** Stable identity across all stages; re-ingestion never duplicates.
Editing a document changes its bytes, so it is correctly treated as new.

**Alternatives:** Random/UUID ids (unstable across re-ingest); content-hash per
chunk (churns ids when neighbors shift).

## 6. `src/` layout with one provider per file

**Decision:** Per-responsibility modules under `src/`, each external provider
isolated behind a single file (embedder, vector_store, reranker, llm client).

**Reason:** Swapping a provider becomes a one-file change; easy to navigate.

**Alternatives:** Flat `app/` layout (mixes concerns); grouping providers
together (defeats the swap-point goal).

## 7. `pydantic-settings` as the single config source

**Decision:** One typed settings source: secrets from `.env`, tunables as typed
defaults. No separate `config.yaml`.

**Reason:** Fewer moving parts for the MVP; add YAML later only if tunables
sprawl.

**Alternatives:** `config.yaml` + `.env` + a loader (three moving parts now).

## 8. v1 formats = PDF + text/Markdown; stack is provisional

**Decision:** Support PDF and text/Markdown for v1. Treat the stack as
provisional, changing it only to meet a current MVP need.

**Reason:** Covers the intended inputs without OCR/complex-format scope.

**Alternatives:** Broad format support up front (out of scope for v1).

## 9. Embedding model: `embed-v4.0` at 1024 dimensions

**Decision:** Use Cohere `embed-v4.0` with `output_dimension=1024`, cosine
similarity. Documents use `input_type="search_document"`, queries use
`input_type="search_query"`; requests are batched at 96 texts (the API limit).

**Reason:** Latest Cohere embedding model, handles multilingual content, and
1024 keeps vectors compact while high quality. This fixes the Pinecone index
dimension to 1024. Model, dimension, input types, and batch limit were all
verified against official Cohere docs.

**Alternatives:** `embed-english-v3.0` (fixed 1024, English-only); default 1536
dimensions (larger vectors for marginal gain at this scale).

## 10. BM25 via `rank-bm25` with a shared tokenizer, rebuilt from SQLite

**Decision:** Use `rank-bm25` (BM25Okapi). One lowercased word-token tokenizer is
used for both indexing and querying. The index holds no persistent state and is
rebuilt from the SQLite chunk store.

**Reason:** Matches the brief; identical preprocessing keeps scores meaningful;
SQLite as source of truth means BM25 never drifts from the canonical chunks.

**Alternatives:** Persisting the BM25 index to disk (risks drift from SQLite);
a heavier search engine (out of scope for the MVP).

## 11. Pinecone stores vectors + lean metadata; text resolves from SQLite

**Decision:** Vectors are stored in a serverless Pinecone index keyed by
`chunk_id`, dimension 1024 / cosine, on AWS `us-east-1`. Pinecone metadata is
kept minimal (`document_id`, `page`, `source_title`); null values are omitted.
The authoritative chunk text is never stored in Pinecone — it is always resolved
from SQLite by `chunk_id`.

**Reason:** One source of truth (SQLite) means the two stores cannot drift.
Lean metadata keeps vectors cheap and avoids duplicating text that could go
stale. `chunk_id` as the vector id keeps identity consistent across dense, BM25,
rerank, and citations. Dimension/metric/region were taken from Pinecone's
official Python SDK docs. The index is created lazily on first upsert.

**Alternatives:** Storing full chunk text in Pinecone metadata (duplication +
drift risk, and Pinecone's per-record metadata size limits); a pod-based index
(more configuration and cost than the MVP needs).

## 12. An indexing service orchestrates ingestion via injected collaborators

**Decision:** `services/indexing_service.py` owns the ingestion flow
(load → chunk → SQLite → embed → Pinecone, plus BM25 rebuild-from-SQLite). It
receives the store, embedder, and vector store by constructor injection and
calls no provider SDK directly. SQLite is written before the vector upsert.

**Reason:** Keeps the orchestration in one readable place, keeps providers
isolated and swappable, and makes the whole flow testable with fakes and no
network. Writing SQLite first means the canonical record survives an embed or
upsert failure.

**Alternatives:** Orchestrating inside a route handler (mixes transport with
pipeline logic); each provider calling the next (couples providers together).

## 13. Hybrid retrieval via reciprocal rank fusion (rank-based, k=60)

**Decision:** Combine the dense (Pinecone) and BM25 rankings with reciprocal
rank fusion: each list contributes `1 / (k + rank)` per item (rank from 1), fused
scores are summed across lists, and results are deduplicated by `chunk_id`. `k`
defaults to 60. Fusion consumes only rank *position*, never the raw provider
scores. Ties break by `chunk_id` for deterministic output.

**Reason:** Dense cosine scores and BM25 scores live on different, incomparable
scales, so averaging them is meaningless — RRF sidesteps this by using rank
only. `k=60` is the widely-cited default that damps top-rank dominance. Passing
ranked id lists (not `(id, score)` pairs) into fusion makes the "no score mixing"
rule structural rather than a convention. `HybridRetriever` composes the two
retrievers plus fusion behind one call, returning ranked `chunk_id`s for the
reranker to consume.

**Alternatives:** Weighted score averaging / normalization (scale-sensitive and
fragile); a single retriever only (loses either semantic or exact-term recall);
learned fusion (out of scope for the MVP).

## 14. Cohere rerank (`rerank-v4.0-pro`); failure is visible and leak-free

**Decision:** Rerank the fused candidate chunks with Cohere Rerank, default model
`rerank-v4.0-pro` (config-driven via `COHERE_RERANK_MODEL`). Documents are sent
as plain chunk text in candidate order; each result's `index` maps back to the
source `Chunk`, preserving the `chunk_id` spine. On a provider failure the
reranker raises a typed `RerankError` with a generic, client-safe message and
chains the original exception (`from exc`) for server-side logs only. The caller
decides whether to degrade or abort — the reranker never silently returns the
un-reranked order.

**Reason:** Reranking is what turns broad hybrid recall into precise top-K.
We initially chose `rerank-v4.0-fast` for cost/latency, but the live smoke test
showed it **consistently times out** on our account, while `rerank-v4.0-pro`
works and returns a strong relevance signal (0.93 vs 0.58 for `rerank-v3.5` on
the same sample). At MVP scale (reranking ~20 chunks per query) the cost
difference is negligible, so `-pro` is the default; the model stays config-driven
so `rerank-v3.5` or a future `-fast` fix is a one-line env change. Surfacing
failure as a typed, non-leaking error satisfies two project rules at once —
"reranking failure is visible" and "never expose provider internals to the
client." Reusing the existing Cohere key/timeout adds no new provider.

**Lesson:** verify a model *works live*, not just that its name appears in the
docs. The doc lookup confirmed `-fast` existed but not that it responded.

**Alternatives:** `rerank-v4.0-fast` (times out here — rejected); `rerank-v3.5`
(works, cheaper, lower relevance — kept as a one-line fallback); no reranking
(hybrid recall alone is noisier); silently falling back to fusion order on
failure (violates the visible-failure rule).

## 15. Blank env values fall back to code defaults for provider identity fields

**Decision:** A blank/whitespace env value for a provider identity setting
(`cohere_embed_model`, `cohere_rerank_model`, `pinecone_index_name`,
`pinecone_cloud`, `pinecone_region`) falls back to the code default instead of
overriding it with `""`. Enforced by a `field_validator` in `config.py`.

**Reason:** A stale blank line like `COHERE_RERANK_MODEL=` in `.env` otherwise
sends an empty model name to the provider and fails only at the network boundary
(this actually happened during the 4a live test). Falling back keeps a sane
default fail-safe and fails fast/clearly instead. Secrets (API keys) are
deliberately excluded — a blank key must stay unset, not acquire a default.

**Alternatives:** Trusting operators to never leave blank lines (fragile);
raising on blank (stricter, but the default is the obviously-intended value).

## 16. Claude for answer generation (`claude-sonnet-5`); isolated transport

**Decision:** Generate answers with Anthropic Claude via the Messages API,
default model `claude-sonnet-5` (config-driven via `ANTHROPIC_MODEL`). The SDK is
isolated behind `llm/client.py` — a thin `generate(system, user) → str` that
holds no prompt-building or citation logic. Failure raises a typed `LLMError`
with a client-safe message, chaining the original for server logs only; response
text is extracted defensively by concatenating all text content blocks.

**Reason:** The brief names Claude for answers. `sonnet-5` is the best
speed/intelligence balance for grounded RAG answers and is cost-appropriate for
an MVP versus Opus; the model stays config-driven so `claude-opus-4-8` (higher
quality) or `claude-haiku-4-5` (cheaper/faster) is a one-line env swap. Keeping
the client a pure transport preserves the swap-point pattern and mirrors the
`RerankError` contract exactly — `LLMError` is its consistent sibling, not a new
style. Package/API surface (`anthropic>=0.116.0`, `messages.create`,
`content[*].text`, client `timeout`) were verified against official docs.

**Alternatives:** `claude-opus-4-8` (higher quality, more cost — one-line swap);
`claude-haiku-4-5` (cheaper/faster, lower quality); putting prompt/citation logic
in the client (couples transport to RAG logic — deferred to `generation/` in 4c).

## 17. Grounded generation: one numbering, validated citations, safe fallback

**Decision:** `RagGenerator` numbers the reranked context chunks `1..N` once; the
same numbering feeds both the prompt and citation validation. The model cites by
number; `citations.py` parses `[n]` references (tolerating `[1][2]` and `[1, 2]`)
and resolves them against that numbering. In-range numbers become `Citation`s;
out-of-range numbers are reported in `invalid_citation_numbers` — flagged, with
the answer kept as-is (hard-reject policy is deferred to the API layer). Two
deterministic safe fallbacks return `FALLBACK_ANSWER` without an unsupported
answer: no context chunks, and an empty model response. The context is presented
as untrusted data, and the system prompt forbids following any instructions
inside it.

**Reason:** A single source-of-truth numbering means a citation can only ever
resolve to a chunk that was actually in context — the structural guard against
fabricated sources the brief requires. Flag-don't-drop keeps the pipeline honest
(a hallucinated citation is visible, not silently swallowed) while leaving the
enforcement decision to the layer that owns the response. Treating documents as
untrusted enforces the PIPELINE invariant against prompt injection from ingested
content.

**Alternatives:** Blanking invalid citations inline in the answer text (mutates
the model's output; harder to audit); trusting the model's citations without
validation (violates the citation-resolves-to-real-chunk rule); detecting
"insufficient context" heuristically (fragile — the model, instructed, is the
right judge for the weak-context case).

## 18. Query service wiring, and the two deferred policies

**Decision:** `services/query_service.py` runs the pipeline end to end —
hybrid retrieve → resolve fused `chunk_id`s to `Chunk`s from SQLite (one batched
`get_chunks`) → rerank → generate. Two previously-deferred policies are decided
here:

- **Rerank failure → surfaced degrade.** A `RerankError` is caught; the service
  answers from the fusion-ranked top-N instead and sets `rerank_failed=True`.
- **Hallucinated citations → flag and return.** The answer is returned with
  `invalid_citation_numbers` populated; it is not rejected.

An empty/whitespace query short-circuits to the safe fallback with no provider
calls. `QueryResult` also carries `used_chunk_ids` for observability.

**Reason:** Degrading (not aborting) on rerank failure keeps the product usable
while still honouring "reranking failure is visible, never pretended" — it is
flagged, not silent. Flag-and-return matches the generation-layer decision (#17)
and leaves hard-reject to the eventual API layer, which owns the response
contract. Resolving ids from SQLite (the source of truth) keeps the `chunk_id`
spine intact from retrieval through citation. Collaborators are injected, so the
whole pipeline is testable offline with fakes.

**Alternatives:** Abort on rerank failure (worse MVP UX for a degradable step);
hard-reject answers with any invalid citation (heavier policy, better owned at
the API boundary); resolving text from the vector store (would duplicate the
source of truth and risk drift).

## 19. HTTP boundary: FastAPI, thin routes, app-factory composition

**Decision:** Expose the pipeline over FastAPI. `create_app()` is a factory that
builds the services once (from settings) and stores the `QueryService` in
`app.state`; tests inject a fake-backed service instead. Routes are thin — they
validate input via Pydantic schemas, call the service, and map errors: input
validation → 422 (empty/blank/oversized question), `LLMError` → 502 with a
generic detail, everything else → FastAPI's generic 500. No provider message,
stack trace, or local path is ever put in a response body. Provider clients are
constructed lazily, so startup does no network I/O (only reads SQLite to build
BM25). `main:app` is the uvicorn entrypoint.

**Reason:** The factory keeps the whole HTTP layer testable in-process with
`TestClient` and no network. Thin routes honour the layering rule (transport
translates, services decide). Validating at the boundary (before any provider
call) plus mapping provider failures to safe status codes is the external-facing
half of the "never leak internals" rule. `rerank_failed` and
`invalid_citation_numbers` travel as response fields, not errors, so clients get
the signal without the request failing.

**Alternatives:** Module-level global app with import-time service construction
(hard to test, network at import); returning 200-with-fallback for a blank
question (hides a client error that 422 states clearly); a heavier framework
(unneeded for three endpoints).

## 20. Ingest upload: parse-from-bytes, validated, with immediate BM25 refresh

**Decision:** `POST /ingest` accepts a multipart file and indexes it **from
memory** via `loader.load_bytes` / `IndexingService.index_bytes` — no filesystem
path is ever built from the upload. The client filename is reduced to a bare
basename (both separators normalized) and used only for format detection and as
the display `source_title`. Validation order: basename present → extension in the
allowlist (`415`) → size ≤ `max_upload_bytes` (`413`, read one byte past the cap)
→ content parses (`400` on empty/unreadable/no-text). After a successful ingest,
BM25 is rebuilt from SQLite and swapped into the live `HybridRetriever`
(`update_bm25`) so the new document is searchable immediately. Unexpected
provider failures surface as FastAPI's generic `500` (safe body, no leak).

**Reason:** Parsing from bytes makes path traversal *structurally* impossible
rather than filtered — there is no path to traverse. Declared MIME is not
trusted; we validate by extension and then actually parse. Reading one byte past
the cap detects overflow without loading an unbounded body. Refreshing BM25 via
an atomic reference swap keeps ingest→query consistent in a single process
without locking. Mapping unexpected errors to a generic 500 (not a broad
`except` → 502) avoids masking bugs while still never leaking provider text,
paths, or stack traces.

**Alternatives:** Writing the upload to a temp file named from the client
(reintroduces path handling / traversal surface); trusting `Content-Type` (spoofable);
rebuilding the whole `QueryService` after ingest (wasteful vs. swapping one index);
a broad `except Exception → 502` (hides genuine bugs behind a friendly status).


## 21. Rerank score threshold: post-cap relevance gate, off by default

**Decision:** After reranking caps candidates at `rerank_top_n`, `QueryService`
drops any survivor whose Cohere relevance score is below
`rerank_score_threshold` (config, env `RERANK_SCORE_THRESHOLD`, validated to
`[0.0, 1.0]`). The default is `0.0` — a no-op that keeps every capped chunk, so
v1 behavior is unchanged until the threshold is deliberately raised. The gate
lives in `QueryService`, alongside the existing rerank-failure and citation
policies, rather than in the `CohereReranker` (which stays a thin provider
wrapper returning scored results). If **every** capped chunk scores below the
bar, the query returns the safe `FALLBACK_ANSWER` with `is_fallback=True` and
**no generation call is made**. The threshold is applied only on the rerank
*success* path; the rerank-failure degrade path carries no scores, so it falls
back to fusion order unchanged.

**Reason:** `rerank_top_n` is a fixed count cap, not a relevance filter — it
drops the least-relevant tail but still forwards marginally-relevant chunks up to
the cap. A score gate lets genuinely irrelevant text be withheld from the LLM,
tightening grounding and saving tokens. Defaulting to `0.0` keeps the change
risk-free: absolute rerank scores vary by query and corpus, so a non-zero
default without empirical tuning could reject valid answers. Declining without a
generation call when nothing clears the bar makes the code-level fallback
reachable in production (previously only an empty retrieval triggered it) and
avoids paying for an answer with no grounding.

**Alternatives:** Filtering inside `CohereReranker` (mixes provider I/O with
answer policy); replacing the count cap with a pure score cap (loses the bounded
context guarantee `rerank_top_n` gives the prompt); shipping a non-zero default
(risks silently dropping good answers before the threshold is tuned); still
calling the LLM on an empty context and relying only on the prompt-level decline
(wastes a call when the code already knows nothing is relevant).
