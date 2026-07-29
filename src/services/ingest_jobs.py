"""Job records for asynchronous ingestion.

Ingestion runs in a background task, so the HTTP response cannot carry the
result. This store holds the outcome instead: the client submits a document,
receives a ``job_id``, and polls for the terminal state.

Jobs are **not** transactional with respect to indexing. A job interrupted after
chunks reach Pinecone but before it is marked ``succeeded`` leaves those vectors
indexed while the job reads ``failed``. Re-ingesting the same content is safe —
``chunk_id`` is content-derived, so the write upserts rather than duplicates.

The store shares the SQLite file with ``ChunkStore`` but owns its own table.

A connection is opened per operation, so ``:memory:`` will not work — each call
would get a fresh, empty database. Use a real path, including in tests.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    source_title TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    document_id  TEXT,
    chunk_count  INTEGER,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status ON ingest_jobs(status);
"""

INTERRUPTED_ERROR = "Ingestion was interrupted by a server restart."


class JobStatus(StrEnum):
    """Lifecycle of an ingestion job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: States that cannot survive a process restart, since background tasks run
#: in-process and are lost when it exits.
_UNFINISHED = (JobStatus.PENDING.value, JobStatus.RUNNING.value)


@dataclass(frozen=True)
class IngestJob:
    """A submitted ingestion and its outcome so far."""

    job_id: str
    status: JobStatus
    source_title: str
    created_at: str
    updated_at: str
    document_id: str | None = None
    chunk_count: int | None = None
    error: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IngestJobStore:
    """SQLite-backed record of ingestion jobs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, *, source_title: str) -> IngestJob:
        """Record a newly submitted job in ``pending``."""
        now = _now()
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            status=JobStatus.PENDING,
            source_title=source_title,
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO ingest_jobs "
                "(job_id, status, source_title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job.job_id, job.status.value, source_title, now, now),
            )
            conn.commit()
        return job

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status=JobStatus.RUNNING)

    def mark_succeeded(
        self, job_id: str, *, document_id: str, chunk_count: int
    ) -> None:
        self._update(
            job_id,
            status=JobStatus.SUCCEEDED,
            document_id=document_id,
            chunk_count=chunk_count,
        )

    def mark_failed(self, job_id: str, *, error: str) -> None:
        """Record a terminal failure.

        ``error`` must be a caller-safe summary: provider messages and stack
        traces never reach it, so nothing internal leaks through the status
        endpoint.
        """
        self._update(job_id, status=JobStatus.FAILED, error=error)

    def get(self, job_id: str) -> IngestJob | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _to_job(row) if row else None

    def fail_unfinished(self) -> int:
        """Fail jobs left mid-flight by a restart; returns how many.

        Called at startup. Background tasks do not survive the process, so any
        job still ``pending`` or ``running`` is unrecoverable — leaving it would
        make the status endpoint report work that will never finish.
        """
        placeholders = ", ".join("?" for _ in _UNFINISHED)
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                f"UPDATE ingest_jobs SET status = ?, error = ?, updated_at = ? "  # noqa: S608
                f"WHERE status IN ({placeholders})",
                (JobStatus.FAILED.value, INTERRUPTED_ERROR, _now(), *_UNFINISHED),
            )
            conn.commit()
            return cursor.rowcount

    def _update(self, job_id: str, *, status: JobStatus, **fields) -> None:
        assignments = ["status = ?", "updated_at = ?"]
        values: list[object] = [status.value, _now()]
        for name, value in fields.items():
            assignments.append(f"{name} = ?")
            values.append(value)
        values.append(job_id)
        with closing(self._connect()) as conn:
            conn.execute(
                f"UPDATE ingest_jobs SET {', '.join(assignments)} WHERE job_id = ?",  # noqa: S608
                values,
            )
            conn.commit()


def _to_job(row: sqlite3.Row) -> IngestJob:
    return IngestJob(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        source_title=row["source_title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        document_id=row["document_id"],
        chunk_count=row["chunk_count"],
        error=row["error"],
    )
