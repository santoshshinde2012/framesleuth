"""SQLite-backed job store with idempotency key support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from framesleuth.schemas import JobState


@dataclass
class StoredJob:
    """In-memory view of persisted job."""

    id: str
    state: JobState
    progress_pct: int
    content_hash: str
    source_video: str
    bundle_path: str | None
    error_json: dict[str, str] | None
    created_at: str
    updated_at: str
    cancel_requested: bool = False


class JobStore:
    """Transactional job store with state updates and idempotency lookups."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store backed by the SQLite database at ``db_path``."""
        self.db_path = db_path

    async def initialize(self) -> None:
        """Initialize schema if needed (and migrate older databases forward)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    progress_pct INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_video TEXT NOT NULL,
                    bundle_path TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash)"
            )
            # Forward-migrate databases created before the cancel_requested column.
            async with conn.execute("PRAGMA table_info(jobs)") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            if "cancel_requested" not in columns:
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                )
            await conn.commit()

    async def create_job(self, job_id: str, content_hash: str, source_video: str) -> None:
        """Create a queued job record."""
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO jobs(
                    id, state, progress_pct, content_hash, source_video,
                    bundle_path, error_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    JobState.QUEUED.value,
                    0,
                    content_hash,
                    source_video,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            await conn.commit()

    async def find_by_content_hash(self, content_hash: str) -> StoredJob | None:
        """Find existing job by content hash for idempotency."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT * FROM jobs WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
                (content_hash,),
            ) as cursor:
                row = await cursor.fetchone()
        return _row_to_job(row)

    async def get_job(self, job_id: str) -> StoredJob | None:
        """Fetch job by id."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cursor:
                row = await cursor.fetchone()
        return _row_to_job(row)

    async def update_job(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        progress_pct: int | None = None,
        bundle_path: str | None = None,
        error_json: dict[str, str] | None = None,
    ) -> None:
        """Update selected mutable fields for a job."""
        updates: list[str] = []
        values: list[Any] = []

        if state is not None:
            updates.append("state = ?")
            values.append(state.value)
        if progress_pct is not None:
            updates.append("progress_pct = ?")
            values.append(progress_pct)
        if bundle_path is not None:
            updates.append("bundle_path = ?")
            values.append(bundle_path)
        if error_json is not None:
            updates.append("error_json = ?")
            values.append(json.dumps(error_json))

        updates.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())
        values.append(job_id)

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                tuple(values),
            )
            await conn.commit()

    async def request_cancel(self, job_id: str) -> bool:
        """Flag a job for cooperative cancellation; return whether a row was updated."""
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            await conn.commit()
            return bool(cursor.rowcount and cursor.rowcount > 0)

    async def is_cancel_requested(self, job_id: str) -> bool:
        """Whether a cancellation has been requested for ``job_id``."""
        async with aiosqlite.connect(self.db_path) as conn, conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row[0]) if row else False

    async def count_active(self) -> int:
        """Number of jobs currently queued or running (the live queue depth)."""
        active = ",".join(f"'{s.value}'" for s in _ACTIVE_STATES)
        async with aiosqlite.connect(self.db_path) as conn, conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE state IN ({active})"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def list_jobs(self) -> list[StoredJob]:
        """Return every persisted job (used by retention cleanup)."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM jobs ORDER BY created_at") as cursor:
                rows = await cursor.fetchall()
        return [job for job in (_row_to_job(row) for row in rows) if job is not None]

    async def delete_job(self, job_id: str) -> None:
        """Delete a job row (its bundle directory is removed separately)."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await conn.commit()


# Job states that count toward the live queue depth (not yet terminal).
_ACTIVE_STATES = (
    JobState.QUEUED,
    JobState.PREPROCESSING,
    JobState.UNDERSTANDING,
    JobState.CLASSIFYING,
    JobState.EXTRACTING,
    JobState.GROUNDING,
)


def _row_to_job(row: Any) -> StoredJob | None:
    if row is None:
        return None
    keys = row.keys() if hasattr(row, "keys") else []
    return StoredJob(
        id=row["id"],
        state=JobState(row["state"]),
        progress_pct=int(row["progress_pct"]),
        content_hash=row["content_hash"],
        source_video=row["source_video"],
        bundle_path=row["bundle_path"],
        error_json=json.loads(row["error_json"]) if row["error_json"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        cancel_requested=bool(row["cancel_requested"]) if "cancel_requested" in keys else False,
    )
