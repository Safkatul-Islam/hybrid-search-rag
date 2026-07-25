"""Canonical SQLite chunk store — the single source of truth for chunk data.

Pinecone will hold only vectors; this store holds the authoritative chunk text
and metadata. BM25 rebuilds from here on startup, and citations resolve
``chunk_id -> text`` from here. Re-ingesting a document upserts by ``chunk_id``
so identical content never duplicates.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path

from src.models.chunk import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    text         TEXT NOT NULL,
    source_title TEXT NOT NULL,
    position     INTEGER NOT NULL,
    page         INTEGER,
    section      TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""

_UPSERT = """
INSERT INTO chunks
    (chunk_id, document_id, text, source_title, position, page, section, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chunk_id) DO UPDATE SET
    document_id  = excluded.document_id,
    text         = excluded.text,
    source_title = excluded.source_title,
    position     = excluded.position,
    page         = excluded.page,
    section      = excluded.section,
    metadata     = excluded.metadata
"""


class ChunkStore:
    """A small SQLite-backed store keyed by ``chunk_id``."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Insert or update chunks by ``chunk_id``. Returns the number written."""
        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.text,
                c.source_title,
                c.position,
                c.page,
                c.section,
                json.dumps(c.metadata),
            )
            for c in chunks
        ]
        if not rows:
            return 0
        with closing(self._connect()) as conn:
            conn.executemany(_UPSERT, rows)
            conn.commit()
        return len(rows)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY position",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id in one query, returned in the given id order.

        Ids with no matching row are skipped (SQLite is the source of truth, so
        a missing id is not expected — this is defensive, not silent behavior a
        caller relies on).
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
        by_id = {row["chunk_id"]: _row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def all_chunks(self) -> list[Chunk]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY document_id, position"
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        text=row["text"],
        source_title=row["source_title"],
        position=row["position"],
        page=row["page"],
        section=row["section"],
        metadata=json.loads(row["metadata"]),
    )
