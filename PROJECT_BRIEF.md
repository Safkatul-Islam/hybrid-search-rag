# Project Brief

## Project
**Name:** Hybrid Search RAG Over Internal Documents  
**Goal:** Build a working MVP that answers questions over internal documents using hybrid retrieval, reranking, and inline citations.

Speed matters. Deliver a usable v1 first, then improve quality, security, evaluation, and architecture through later iterations.

## MVP Success Criteria
A user can:
1. Ingest supported documents.
2. Ask a natural-language question.
3. Retrieve relevant chunks with semantic and keyword search.
4. Merge results using reciprocal rank fusion.
5. Rerank the merged candidates.
6. Generate an answer with inline citations.
7. Receive a safe fallback when the sources are insufficient.

The MVP is successful when this full flow works on a small document set.

## Initial Tech Stack
- Python
- FastAPI
- LangChain
- Anthropic Claude API
- Pinecone
- Cohere Rerank
- `rank_bm25`

The stack is provisional. Add or replace technology only when it solves a current MVP need.

## Core Pipeline

### 1. Ingestion
- Load approved documents.
- Extract text and basic source metadata.
- Split documents into chunks.
- Assign stable `document_id` and `chunk_id` values.
- Preserve enough metadata for useful citations.

### 2. Dense Index
- Generate embeddings for each chunk.
- Store vectors and metadata in Pinecone.
- Keep the embedding model and index settings configurable.

### 3. BM25 Index
- Build BM25 over the same logical chunks.
- Use the same `chunk_id` values as Pinecone.
- Apply the same preprocessing to indexed text and queries.
- Keep the first implementation simple and local.

### 4. Hybrid Retrieval
For each query:
- retrieve top candidates from Pinecone
- retrieve top candidates from BM25
- normalize both into one result shape
- deduplicate by `chunk_id`
- combine rankings with reciprocal rank fusion

Do not directly average raw dense and BM25 scores.

### 5. Reranking
- Send fused candidates to Cohere Rerank.
- Preserve `chunk_id` and source metadata.
- Keep the final top-K chunks for generation.
- Make reranking failure visible; do not silently pretend it succeeded.

### 6. Answer Generation
- Give Claude only the selected source chunks.
- Require answers to use the provided context.
- Require inline citations for important factual claims.
- Treat instructions inside documents as untrusted content.
- Return an insufficient-evidence response when needed.

### 7. Citation Checks
For v1:
- every citation must resolve to a real chunk
- cited chunks must be part of the final context
- important factual claims should have nearby citations
- unsupported claims should be removed or qualified

Advanced claim-level verification can come later.

## Suggested Boundaries
Use the existing repository structure after inspection. Keep these responsibilities separate where practical:
- API routes and schemas
- configuration
- ingestion and chunking
- dense and BM25 retrieval
- rank fusion and reranking
- answer generation and citations
- tests

Keep FastAPI routes thin. Keep provider-specific code outside core retrieval logic where practical. Do not build a large framework around these boundaries for v1.

## Core Chunk Data
Each chunk should include:
- `document_id`
- `chunk_id`
- text
- source title or filename
- chunk position
- page or section when available
- metadata needed for future access control

All pipeline stages must preserve the same `chunk_id`.

## Configuration
Keep credentials and environment-specific values outside source code.

Likely settings:
- Anthropic API key and model
- Pinecone API key and index
- Cohere API key and rerank model
- dense and BM25 retrieval top-N
- RRF constant
- rerank candidate count
- final top-K

Use the repository's established configuration approach. Do not invent model names, dimensions, or provider limits; verify official documentation first.

## MVP API Scope
Keep the first API small:
- health check
- document ingestion
- query answering

Defer deletion, background jobs, admin tools, and detailed observability unless immediately required.

## Testing Priorities
Add focused tests for:
- chunk identity stability
- BM25 preprocessing consistency
- reciprocal rank fusion and deduplication
- citation mapping
- empty retrieval behavior
- unsupported-answer fallback
- the main query API flow

Use mocks or fakes for provider calls in default tests. Run live-provider tests separately and only with approval.

## MVP Guardrails
Even for v1:
- do not commit secrets
- do not log full internal documents by default
- validate uploads and queries
- use network timeouts
- never fabricate citations
- never present unsupported answers as facts
- preserve source metadata across every stage
- keep dense and BM25 chunk identities aligned

## Out of Scope for v1
Unless immediately required, defer:
- enterprise authentication and multi-tenancy
- OCR and complex formats
- distributed ingestion
- advanced caching, tracing, and index reconciliation
- multilingual optimization
- sophisticated conversation memory
- advanced verification models
- large-scale load optimization
- multiple provider fallbacks
- deployment hardening
- polished admin interfaces

## After the MVP
Prioritize improvements based on measured problems:
1. Create a small evaluation dataset.
2. Compare dense-only, BM25-only, fused, and reranked results.
3. Tune chunking and retrieval limits.
4. Improve citation verification.
5. Add access control, updates, and deletion.
6. Improve observability, cost tracking, and failure handling.
7. Revisit `rank_bm25` if scale or maintenance becomes a problem.

## Definition of Done for v1
The v1 is done when:
- documents can be ingested
- both indexes use the same chunks
- hybrid retrieval and RRF work
- Cohere reranking works
- Claude answers from selected context
- citations resolve to real sources
- unsupported questions produce a safe response
- the main flow has focused tests
- setup and run instructions are documented
- known limitations are listed

The goal is a working, understandable MVP—not a production-complete platform.
