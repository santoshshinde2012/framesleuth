"""Tests for bundle retention / TTL cleanup."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from framesleuth.jobs.retention import purge_expired_bundles
from framesleuth.jobs.store import JobStore


async def _seed(store: JobStore, bundle_dir: Path, job_id: str, created: datetime) -> None:
    await store.create_job(job_id, f"h-{job_id}", "v.mp4")
    # Backdate created_at directly so age can be controlled deterministically.
    import aiosqlite

    async with aiosqlite.connect(store.db_path) as conn:
        await conn.execute(
            "UPDATE jobs SET created_at = ? WHERE id = ?", (created.isoformat(), job_id)
        )
        await conn.commit()
    (bundle_dir / job_id).mkdir(parents=True, exist_ok=True)
    (bundle_dir / job_id / "bundle.json").write_text("{}", encoding="utf-8")


@pytest.mark.asyncio
async def test_purge_removes_only_expired(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles"
    store = JobStore(tmp_path / "jobs.db")
    await store.initialize()
    now = datetime.now(UTC)
    await _seed(store, bundle_dir, "old", now - timedelta(days=40))
    await _seed(store, bundle_dir, "fresh", now - timedelta(days=1))

    purged = await purge_expired_bundles(store, bundle_dir, ttl_days=30, now=now)

    assert purged == 1
    assert {j.id for j in await store.list_jobs()} == {"fresh"}
    assert not (bundle_dir / "old").exists()
    assert (bundle_dir / "fresh").exists()


@pytest.mark.asyncio
async def test_purge_disabled_when_ttl_zero(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles"
    store = JobStore(tmp_path / "jobs.db")
    await store.initialize()
    await _seed(store, bundle_dir, "old", datetime.now(UTC) - timedelta(days=999))

    assert await purge_expired_bundles(store, bundle_dir, ttl_days=0) == 0
    assert {j.id for j in await store.list_jobs()} == {"old"}
