"""Tests for SQLite job store and idempotency behavior."""

from pathlib import Path

import pytest

from framesleuth.jobs.store import JobStore
from framesleuth.schemas import JobState


@pytest.mark.asyncio
async def test_job_store_create_get_and_hash_lookup(tmp_path: Path) -> None:
    """Store should persist, fetch, and find jobs by content hash."""
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    await store.initialize()

    await store.create_job("job-1", "hash-abc", "video.mp4")

    by_id = await store.get_job("job-1")
    by_hash = await store.find_by_content_hash("hash-abc")

    assert by_id is not None
    assert by_hash is not None
    assert by_id.id == by_hash.id == "job-1"
    assert by_id.state == JobState.QUEUED


@pytest.mark.asyncio
async def test_job_store_update_transitions(tmp_path: Path) -> None:
    """Store updates should persist state/progress and optional bundle path."""
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    await store.initialize()

    await store.create_job("job-2", "hash-def", "video.mp4")
    await store.update_job(
        "job-2", state=JobState.DONE, progress_pct=100, bundle_path="bundle.json"
    )

    job = await store.get_job("job-2")
    assert job is not None
    assert job.state == JobState.DONE
    assert job.progress_pct == 100
    assert job.bundle_path == "bundle.json"


@pytest.mark.asyncio
async def test_request_and_read_cancellation(tmp_path: Path) -> None:
    """Cancellation flag is persisted and read back."""
    store = JobStore(tmp_path / "jobs.db")
    await store.initialize()
    await store.create_job("job-c", "h", "v.mp4")

    assert await store.is_cancel_requested("job-c") is False
    assert await store.request_cancel("job-c") is True
    assert await store.is_cancel_requested("job-c") is True
    assert await store.request_cancel("missing") is False  # no row updated

    job = await store.get_job("job-c")
    assert job is not None and job.cancel_requested is True


@pytest.mark.asyncio
async def test_count_active_excludes_terminal(tmp_path: Path) -> None:
    """Queue depth counts only non-terminal jobs."""
    store = JobStore(tmp_path / "jobs.db")
    await store.initialize()
    await store.create_job("q1", "h1", "v.mp4")
    await store.create_job("q2", "h2", "v.mp4")
    await store.update_job("q2", state=JobState.DONE)
    await store.create_job("q3", "h3", "v.mp4")
    await store.update_job("q3", state=JobState.CANCELLED)

    assert await store.count_active() == 1  # only q1 is still active


@pytest.mark.asyncio
async def test_list_and_delete_jobs(tmp_path: Path) -> None:
    """Jobs can be enumerated and deleted (used by retention)."""
    store = JobStore(tmp_path / "jobs.db")
    await store.initialize()
    await store.create_job("a", "h1", "v.mp4")
    await store.create_job("b", "h2", "v.mp4")
    assert {j.id for j in await store.list_jobs()} == {"a", "b"}

    await store.delete_job("a")
    assert {j.id for j in await store.list_jobs()} == {"b"}


@pytest.mark.asyncio
async def test_migration_adds_cancel_column(tmp_path: Path) -> None:
    """An older DB without cancel_requested is migrated forward on initialize."""
    import aiosqlite

    db_path = tmp_path / "jobs.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, state TEXT NOT NULL, progress_pct INTEGER NOT NULL,
                content_hash TEXT NOT NULL, source_video TEXT NOT NULL, bundle_path TEXT,
                error_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            "INSERT INTO jobs VALUES('old','queued',0,'h','v.mp4',NULL,NULL,'t','t')"
        )
        await conn.commit()

    store = JobStore(db_path)
    await store.initialize()  # should ALTER TABLE to add the column
    job = await store.get_job("old")
    assert job is not None and job.cancel_requested is False
    assert await store.request_cancel("old") is True
