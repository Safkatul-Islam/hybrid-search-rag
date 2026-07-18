# Architecture

Single responsibility per module. Each external provider is isolated behind one
file (marked **← swap point**), so replacing a provider means changing one place.

```
p-2/
├── README.md
├── pyproject.toml               # deps + ruff/pytest config
├── requirements.lock            # pinned full dependency tree (reproducible install)
├── main.py                      # uvicorn entrypoint (main:app)
├── .env.example                 # documents settings; real .env is gitignored
├── .gitignore
├── DECISIONS.md / PIPELINE.md / PROJECT_BRIEF.md
├── src/
│   ├── config.py                # pydantic-settings: secrets + tunables
│   ├── models/
│   │   └── chunk.py             # Chunk model + deterministic IDs  ← the spine
│   ├── ingestion/
│   │   ├── loader.py            # PDF (pypdf) + text/Markdown
│   │   └── chunker.py           # langchain-text-splitters → Chunk
│   ├── embeddings/
│   │   └── embedder.py          # Cohere embeddings (embed-v4.0, 1024d)  ← swap point
│   ├── storage/
│   │   ├── chunk_store.py       # SQLite canonical store (source of truth)
│   │   └── vector_store.py      # Pinecone (embed-v4.0 dim, cosine)  ← swap point
│   ├── services/
│   │   ├── indexing_service.py  # load→chunk→SQLite→embed→Pinecone; rebuild BM25
│   │   └── query_service.py     # retrieve→resolve→rerank→generate (end to end)
│   ├── retrieval/
│   │   ├── dense.py             # embed query → vector_store query
│   │   ├── bm25.py              # rank_bm25 (rebuilds from chunk_store)
│   │   ├── fusion.py            # reciprocal rank fusion (rank-based)
│   │   └── hybrid.py            # dense + BM25 → fusion (ranked chunk_ids)
│   ├── reranking/
│   │   └── reranker.py          # Cohere rerank (v4.0-pro)      ← swap point
│   ├── generation/
│   │   ├── generator.py         # RAG answer logic + safe fallback
│   │   └── citations.py         # numbered-citation parse + validate
│   ├── llm/
│   │   └── client.py            # Claude transport (sonnet-5)   ← swap point
│   ├── prompts/
│   │   └── templates.py         # answer prompt + citation instructions
│   ├── api/
│   │   ├── app.py              # create_app factory + AppServices composition
│   │   ├── routes.py           # health, query, ingest (thin)
│   │   └── schemas.py           # request/response models
│   └── utils/                   # only when something is genuinely shared     [planned]
└── tests/
```

`[planned]` modules are created in the batch that first uses them, to avoid
empty placeholder files.

## Layering

- **FastAPI routes stay thin** — they call services and translate errors only.
- **Provider code stays out of core retrieval logic** — `retrieval/fusion.py`
  and citation logic operate on plain `Chunk` data, not provider SDK objects.
- **SQLite is the source of truth**; Pinecone stores vectors + lean metadata
  (`document_id`, `page`, `source_title`) only — text always resolves from SQLite.
- **Services orchestrate; providers stay isolated** — `indexing_service.py`
  wires the ingestion chain by calling injected collaborators (store, embedder,
  vector store), never provider SDKs directly, so each provider stays swappable
  and the flow is testable with fakes.
