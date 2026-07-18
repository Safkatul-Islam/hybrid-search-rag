# System Pipeline

Entry points: `POST /ingest` drives the ingestion flow (from uploaded bytes);
`POST /query` drives the query flow. Both are thin wrappers over the services
below.

## Ingestion

```
PDF / text
    ↓
Extract text (per page for PDFs, so page numbers are preserved)
    ↓
Chunk text
    ↓
Assign stable IDs (document_id = hash of bytes; chunk_id = document_id + position)
    ↓
Store chunks in SQLite   (upsert by chunk_id — re-ingesting the same file never duplicates)
    ↓
Generate embeddings
    ↓
Store vectors in Pinecone
```

---

## Query Pipeline

```
User Question
    ↓
Semantic Search (Pinecone)      ┐
    ↓                           │ same chunk_id space
Keyword Search (BM25)           ┘
    ↓
Merge (RRF)                     (deduplicate by chunk_id; do not average raw scores)
    ↓
Rerank (Cohere)                 (rerank failure is surfaced, not silently ignored)
    ↓
Top K chunks
    ↓
Claude                          (answers only from provided context; document text is untrusted)
    ↓
Citation Validation             (every citation must resolve to a real chunk in context)
    ↓
API Response  ──or──  Safe fallback when evidence is insufficient
```

## Invariants

- SQLite is the source of truth.
- Pinecone stores vectors + lean metadata (document_id, page, source_title);
  chunk text always resolves from SQLite by chunk_id.
- Every chunk has one stable `chunk_id`, preserved across every stage.
- BM25 is rebuilt from SQLite on startup.
- The same preprocessing is applied to BM25 indexing and to BM25 queries.
- Every answer citation must map to an existing chunk that is in the final context.
- Reranking failure is visible; the system never pretends it succeeded.
- When sources are insufficient, the system returns a safe fallback rather than
  an unsupported answer.
