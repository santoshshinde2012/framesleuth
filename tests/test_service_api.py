"""Tests for FastAPI service endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from framesleuth.config import Settings
from framesleuth.schemas import JobState
from framesleuth.service.api import create_app


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        BUNDLE_DIR=tmp_path / "bundles",
        DATABASE_PATH=tmp_path / "jobs.db",
        MAX_UPLOAD_MB=10,
        CHROME_EXTENSION_ORIGIN="chrome-extension://test",
    )


def test_analyze_and_report_roundtrip(tmp_path: Path) -> None:
    """Analyze endpoint should create idempotent job and report should be retrievable."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps({"id": job_id, "title": "test", "source_video": source_video}),
            encoding="utf-8",
        )
        (bundle_dir / "metrics.json").write_text(
            json.dumps({"stages": {"preprocess": 0.1, "understand": 1.2}, "degraded": []}),
            encoding="utf-8",
        )
        await state.store.update_job(
            job_id,
            state=JobState.DONE,
            progress_pct=100,
            bundle_path=str(bundle_path),
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run

        response = client.post(
            "/v1/analyze",
            files={"video": ("sample.mp4", b"video-content", "video/mp4")},
        )
        # Analysis is accepted (202) and runs in the background; the TestClient
        # drains the background task before returning, so the job is done by now.
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        job_id = payload["job_id"]

        job = client.get(f"/v1/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["state"] == "done"
        # Per-stage metrics are surfaced to the poller for observability.
        assert job.json()["metrics"]["stages"]["understand"] == 1.2

        report = client.get(f"/v1/report/{job_id}")
        assert report.status_code == 200
        assert report.json()["id"] == job_id

        # The temp upload is cleaned up, not left accumulating in the bundle dir.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))

        # Same upload should hit idempotency path.
        second = client.post(
            "/v1/analyze",
            files={"video": ("sample.mp4", b"video-content", "video/mp4")},
        )
        assert second.status_code == 202
        assert second.json()["idempotent"] == "true"


def test_analyze_records_typed_failure_in_background(tmp_path: Path) -> None:
    """A typed pipeline failure marks the job FAILED with a structured error."""
    from framesleuth.errors import UnsupportedMediaError

    app = create_app(_make_settings(tmp_path))

    async def boom(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        raise UnsupportedMediaError("bad codec")

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = boom
        resp = client.post("/v1/analyze", files={"video": ("x.mp4", b"vid", "video/mp4")})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        job = client.get(f"/v1/jobs/{job_id}").json()
        assert job["state"] == "failed"
        assert job["error"]["code"] == "unsupported_media"

        # The temp upload is still cleaned up on the failure path.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))


def test_get_video_serves_correct_media_type(tmp_path: Path) -> None:
    """The stored source is served with a content-type matching its container."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        # An mp4 source must not be advertised as webm.
        (bundle_dir / "source.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        video = client.get(f"/v1/video/{job_id}")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/mp4"


def _write_sample_video(path: Path, *, frames: int = 16, fps: int = 8) -> None:
    """Encode a tiny synthetic mp4 for endpoints that need a real recording."""
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 128
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            arr = np.full((96, 128, 3), (i * 12) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_get_gif_renders_and_caches_preview(tmp_path: Path) -> None:
    """GET /v1/gif encodes a GIF from the stored source and caches it on disk."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        _write_sample_video(bundle_dir / "source.mp4")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        resp = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/gif"
        assert resp.content[:6] in (b"GIF87a", b"GIF89a")

        # The render is cached on disk keyed by its parameters.
        bundle_dir = app.state.app_state.settings.BUNDLE_DIR / job_id
        cached = list(bundle_dir.glob("preview-*.gif"))
        assert len(cached) == 1

        # A second identical request reuses the cache (still 200, same file).
        again = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert again.status_code == 200
        assert len(list(bundle_dir.glob("preview-*.gif"))) == 1


def test_get_gif_returns_404_when_no_source(tmp_path: Path) -> None:
    """A job without a stored recording yields a 404, not a 500."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post(
            "/v1/analyze", files={"video": ("bug.mp4", b"vid", "video/mp4")}
        ).json()["job_id"]

        resp = client.get(f"/v1/gif/{job_id}")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "missing_video"


def test_skills_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/skills returns the default and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/skills")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "summary"
        names = {s["name"] for s in body["skills"]}
        assert {"summary", "bug_report", "tutorial"} <= names


def test_actions_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/actions returns the default, auto flag, and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/actions")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "fix"
        assert body["auto"] is True
        names = {a["name"] for a in body["actions"]}
        assert {"fix", "explain", "triage", "test", "report", "reproduce"} <= names


def test_analyze_forwards_action_fields(tmp_path: Path) -> None:
    """action/action_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files={"video": ("bug.mp4", b"vid", "video/mp4")},
            data={"action": "triage"},
        )
        assert resp.status_code == 202
        assert captured["action"] == "triage"
        assert captured["action_prompt"] is None


def test_analyze_forwards_skill_and_system_prompt(tmp_path: Path) -> None:
    """skill/system_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files={"video": ("bug.mp4", b"vid", "video/mp4")},
            data={"skill": "tutorial", "intent": "explain it"},
        )
        assert resp.status_code == 202
        assert captured["skill"] == "tutorial"
        assert captured["user_intent"] == "explain it"
        assert captured["system_prompt"] is None


def test_cors_allows_any_chrome_extension_origin(tmp_path: Path) -> None:
    """A real (dynamic) extension origin must be allowed without prior config."""
    app = create_app(_make_settings(tmp_path))
    ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    with TestClient(app) as client:
        # Preflight from the extension origin.
        preflight = client.options(
            "/v1/analyze",
            headers={
                "Origin": ext_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.headers.get("access-control-allow-origin") == ext_origin

        # Actual request echoes the allowed origin too.
        health = client.get("/v1/healthz", headers={"Origin": ext_origin})
        assert health.headers.get("access-control-allow-origin") == ext_origin

    # A normal web origin is NOT allowed.
    with TestClient(app) as client:
        denied = client.get("/v1/healthz", headers={"Origin": "https://evil.example.com"})
        assert denied.headers.get("access-control-allow-origin") is None


def test_upload_limit_enforced(tmp_path: Path) -> None:
    """Oversized uploads should be rejected with 413."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        large = b"a" * (11 * 1024 * 1024)
        response = client.post(
            "/v1/analyze",
            files={"video": ("big.mp4", large, "video/mp4")},
        )
        assert response.status_code == 413


def test_render_html_rejects_missing_html(tmp_path: Path) -> None:
    """No/blank HTML is a 400 with a stable error code, before any render work."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        for body in ({}, {"html": ""}, {"html": "   "}):
            resp = client.post("/v1/render-html", json=body)
            assert resp.status_code == 400
            assert resp.json()["detail"]["code"] == "missing_html"


def test_render_html_rejects_bad_format(tmp_path: Path) -> None:
    """An unsupported output format is rejected up front (400 bad_options)."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>hi</h1>", "format": "tiff"})
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "bad_options"


def test_render_html_success_streams_file(tmp_path: Path, monkeypatch) -> None:
    """A successful render returns the encoded file with the right media type.

    The Chromium/ffmpeg work is stubbed so the endpoint wiring (validation,
    media-type mapping, download filename, response body) is tested without the
    optional render dependencies.
    """
    import framesleuth.pipeline.html_render as hr

    async def fake_render(html: str, options, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"render.{options.fmt}"
        out.write_bytes(b"ENCODED-BYTES")
        return out

    monkeypatch.setattr(hr, "render_html", fake_render)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/render-html",
            json={"html": "<h1>hi</h1>", "format": "gif", "duration_s": 2, "fps": 12},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/gif")
        assert "animation.gif" in resp.headers.get("content-disposition", "")
        assert resp.content == b"ENCODED-BYTES"


def test_render_html_unavailable_returns_503(tmp_path: Path, monkeypatch) -> None:
    """When Playwright/ffmpeg are missing the endpoint reports 503, not 500."""
    import framesleuth.pipeline.html_render as hr

    async def boom(html: str, options, out_dir: Path) -> Path:
        raise hr.HtmlRenderError("Playwright is not installed.")

    monkeypatch.setattr(hr, "render_html", boom)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>hi</h1>", "format": "mp4"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "render_unavailable"


def _seed_job_sync(settings: Settings, job_id: str, state: JobState | None = None) -> None:
    """Insert a job row directly (shared SQLite file) before the app starts."""
    import asyncio

    from framesleuth.jobs.store import JobStore

    async def _seed() -> None:
        store = JobStore(settings.DATABASE_PATH)
        await store.initialize()
        await store.create_job(job_id, f"h-{job_id}", "v.mp4")
        if state is not None:
            await store.update_job(job_id, state=state)

    asyncio.run(_seed())


def test_cancel_queued_job_sets_flag(tmp_path: Path) -> None:
    """DELETE on an active job requests cancellation."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_job_sync(settings, "q-cancel")
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.delete("/v1/jobs/q-cancel")
        assert resp.status_code == 200
        assert resp.json()["cancel_requested"] is True

        job = client.get("/v1/jobs/q-cancel").json()
        # Still queued (a real run would transition it to cancelled at a checkpoint).
        assert job["state"] == "queued"


def test_cancel_terminal_job_is_409(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_job_sync(settings, "q-done", state=JobState.DONE)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.delete("/v1/jobs/q-done")
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "not_cancellable"


def test_cancel_missing_job_is_404(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.delete("/v1/jobs/nope").status_code == 404


def test_job_events_stream_emits_terminal_state(tmp_path: Path) -> None:
    """The SSE stream yields a snapshot and closes once the job is terminal."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_job_sync(settings, "q-sse", state=JobState.DONE)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/q-sse/events")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data:" in resp.text
        assert '"state": "done"' in resp.text


def test_webhook_fires_on_completion(tmp_path: Path, monkeypatch) -> None:
    """A completed job POSTs a compact payload to WEBHOOK_URL."""
    posts: list[tuple[str, dict]] = []

    class _FakeSession:
        def __init__(self, *a, **k) -> None:
            pass

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *a) -> bool:
            return False

        async def post(self, url: str, json: dict | None = None) -> None:
            posts.append((url, json or {}))

    import framesleuth.service.api as api_module

    monkeypatch.setattr(api_module.aiohttp, "ClientSession", _FakeSession)

    settings = _make_settings(tmp_path)
    settings.WEBHOOK_URL = "http://example.test/hook"
    app = create_app(settings)

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps({"id": job_id, "title": "demo", "action": "summarize"}), encoding="utf-8"
        )
        await state.store.update_job(job_id, state=JobState.DONE, bundle_path=str(bundle_path))
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post("/v1/analyze", files={"video": ("s.mp4", b"vid", "video/mp4")})
        assert resp.status_code == 202

    assert posts, "expected a webhook POST"
    url, payload = posts[0]
    assert url == "http://example.test/hook"
    assert payload["state"] == "done"
    assert payload["title"] == "demo"


def test_healthz_includes_render_availability(tmp_path: Path) -> None:
    """/v1/healthz surfaces the optional HTML→video readiness block."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/healthz")
        assert resp.status_code == 200
        render = resp.json()["render"]
        for key in ("playwright", "chromium", "ffmpeg", "python", "ready"):
            assert key in render
        assert isinstance(render["ready"], bool)


def test_cors_allows_hosted_site_to_reach_local_agent(tmp_path: Path) -> None:
    """The hosted site origin can preflight the loopback API (CORS + Private Network).

    This is what lets framesleuth.com talk to a locally-running agent: the browser
    sends a Private Network Access preflight, and the API must echo the origin plus
    Access-Control-Allow-Private-Network so Chrome doesn't block the real request.
    """
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={
                "Origin": "https://framesleuth.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://framesleuth.com"
        assert resp.headers.get("access-control-allow-private-network") == "true"


def test_cors_rejects_unknown_web_origin(tmp_path: Path) -> None:
    """An origin not on the allowlist is never granted access."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") is None


def test_web_origins_setting_is_honored(tmp_path: Path) -> None:
    """A custom WEB_ORIGINS allowlist is respected (and parsed from the CSV string)."""
    settings = _make_settings(tmp_path)
    settings.WEB_ORIGINS = "https://example.dev, http://localhost:4000"
    assert settings.web_origins_list == ["https://example.dev", "http://localhost:4000"]
    app = create_app(settings)
    with TestClient(app) as client:
        ok = client.options(
            "/v1/healthz",
            headers={"Origin": "https://example.dev", "Access-Control-Request-Method": "GET"},
        )
        assert ok.headers.get("access-control-allow-origin") == "https://example.dev"
        # framesleuth.com is no longer in the (overridden) allowlist.
        no = client.options(
            "/v1/healthz",
            headers={"Origin": "https://framesleuth.com", "Access-Control-Request-Method": "GET"},
        )
        assert no.headers.get("access-control-allow-origin") is None
