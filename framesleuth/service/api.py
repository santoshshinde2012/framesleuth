"""FastAPI surface for Framesleuth local analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from framesleuth.clients.health import get_health_status
from framesleuth.clients.vlm import VLMClient
from framesleuth.config import Settings, get_settings
from framesleuth.errors import FramesleutheException, JobNotFoundError, UploadTooLargeError
from framesleuth.jobs.store import JobStore
from framesleuth.logging_config import get_logger
from framesleuth.orchestrator.graph import AnalysisOrchestrator
from framesleuth.schemas import JobState

logger = get_logger("service.api")

# Content types for stored source recordings, keyed by file suffix.
_VIDEO_MEDIA_TYPES = {
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def _find_source_video(bundle_dir: Path) -> Path | None:
    """Return the stored ``source.*`` recording in ``bundle_dir``, if present."""
    for source in sorted(bundle_dir.glob("source.*")):
        if source.suffix.lower() in _VIDEO_MEDIA_TYPES:
            return source
    return None


def _safe_json(raw: str | None) -> Any:
    """Parse a JSON form field, tolerating absent or malformed input."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


class AppState:
    """State container for app dependencies."""

    def __init__(self, settings: Settings) -> None:
        """Build the store, VLM client, and orchestrator from ``settings``."""
        self.settings = settings
        self.store = JobStore(settings.DATABASE_PATH)
        self.vlm_client = VLMClient.from_settings(settings)
        self.orchestrator = AnalysisOrchestrator(settings, self.store, self.vlm_client)
        # Bounds how many analyses run the heavy pipeline at once. Excess jobs sit
        # in QUEUED state (the analyze response already returned their id), giving
        # real backpressure instead of fanning out unbounded VLM/ffmpeg work.
        self._job_semaphore = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_JOBS))

    async def run_job(
        self,
        job_id: str,
        temp_path: Path,
        source_video: str,
        *,
        sidecars: Any,
        capture_options: str | None,
        intent: str | None,
        skill: str | None,
        system_prompt: str | None,
        action: str | None,
        action_prompt: str | None,
    ) -> None:
        """Run one analysis to completion in the background, bounded by the semaphore.

        The orchestrator marks the job FAILED on unexpected errors; this wrapper
        additionally records typed ``FramesleutheException``s (e.g. unsupported
        media) so a poller sees a structured ``error`` rather than a stuck job,
        and always removes the temp upload.
        """
        async with self._job_semaphore:
            try:
                await self.orchestrator.run(
                    job_id,
                    temp_path,
                    source_video,
                    sidecars=sidecars,
                    user_intent=intent,
                    skill=skill,
                    system_prompt=system_prompt,
                    action=action,
                    action_prompt=action_prompt,
                )
                if capture_options:
                    persisted = await self.store.get_job(job_id)
                    if persisted and persisted.bundle_path:
                        Path(persisted.bundle_path).parent.joinpath(
                            "capture_options.json"
                        ).write_text(capture_options, encoding="utf-8")
            except FramesleutheException as exc:
                await self.store.update_job(job_id, state=JobState.FAILED, error_json=exc.to_dict())
            except Exception:  # orchestrator already logged + marked FAILED
                logger.exception("Background analysis failed for job %s", job_id)
            finally:
                temp_path.unlink(missing_ok=True)


def create_app(settings: Settings | None = None) -> FastAPI:  # noqa: C901
    """Create configured FastAPI application.

    Complexity is inherent to an app factory that registers every route as a
    nested closure over shared state; the individual handlers stay simple.
    """
    settings = settings or get_settings()
    state = AppState(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings.validate_paths()
        await state.store.initialize()
        try:
            yield
        finally:
            # Release the pooled VLM HTTP session on shutdown.
            await state.vlm_client.aclose()

    app = FastAPI(title="Framesleuth API", version="1.0.0", lifespan=lifespan)
    app.state.app_state = state

    # Accept the configured extension origin plus any chrome-extension:// origin
    # (the extension ID is only known after the unpacked build is loaded). This
    # stays scoped to extension origins and the loopback bind — it never opens
    # the API to arbitrary web origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.CHROME_EXTENSION_ORIGIN],
        allow_origin_regex=r"chrome-extension://[a-p]{32}",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/v1/healthz")
    async def healthz() -> dict[str, object]:
        health = await get_health_status(
            vlm_url=settings.VLM_URL,
            coder_url=settings.CODER_URL,
            bundle_dir=str(settings.BUNDLE_DIR),
            queue_depth=0,
        )
        payload = health.model_dump()
        # Surface optional HTML→video readiness so it's diagnosable in the running
        # process (cheap, no browser launch).
        from framesleuth.pipeline.html_render import render_availability

        payload["render"] = render_availability()
        return payload

    @app.get("/v1/skills")
    async def skills() -> dict[str, object]:
        """List built-in summary skills the caller can pass to ``/v1/analyze``."""
        from framesleuth.skills import DEFAULT_SKILL, list_skills

        return {"default": DEFAULT_SKILL, "skills": list_skills()}

    @app.get("/v1/actions")
    async def actions() -> dict[str, object]:
        """List built-in action modes the caller can pass to ``/v1/analyze``."""
        from framesleuth.actions import DEFAULT_ACTION, list_actions

        return {"default": DEFAULT_ACTION, "auto": True, "actions": list_actions()}

    @app.post("/v1/analyze", status_code=202)
    async def analyze(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        sidecars: str | None = Form(default=None),
        capture_options: str | None = Form(default=None),
        intent: str | None = Form(default=None),
        skill: str | None = Form(default=None),
        system_prompt: str | None = Form(default=None),
        action: str | None = Form(default=None),
        action_prompt: str | None = Form(default=None),
    ) -> dict[str, str]:
        """Accept a recording and queue it for analysis.

        Returns ``202`` with the new ``job_id`` immediately; the analysis runs in
        the background (bounded by ``MAX_CONCURRENT_JOBS``). Poll ``/v1/jobs/{id}``
        for progress and ``/v1/report/{id}`` once the state is ``done``.
        """
        try:
            suffix = Path(video.filename or "upload.mp4").suffix.lower()
            content = await video.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > settings.MAX_UPLOAD_MB:
                raise UploadTooLargeError(size_mb, settings.MAX_UPLOAD_MB)

            content_hash = hashlib.sha256(content).hexdigest()
            existing = await state.store.find_by_content_hash(content_hash)
            if existing is not None:
                return {"job_id": existing.id, "status": existing.state.value, "idempotent": "true"}

            temp_path = settings.BUNDLE_DIR / f"upload-{uuid.uuid4()}{suffix}"
            temp_path.write_bytes(content)
            job_id = str(uuid.uuid4())
            await state.store.create_job(job_id, content_hash, video.filename or "upload.mp4")

            # Hand off to a background task: the request returns the job id now and
            # the heavy pipeline (copied into the bundle dir, temp removed after)
            # runs under the concurrency semaphore. Poll /v1/jobs/{id} for state.
            background_tasks.add_task(
                state.run_job,
                job_id,
                temp_path,
                video.filename or "upload.mp4",
                sidecars=_safe_json(sidecars),
                capture_options=capture_options,
                intent=intent,
                skill=skill,
                system_prompt=system_prompt,
                action=action,
                action_prompt=action_prompt,
            )
            return {"job_id": job_id, "status": JobState.QUEUED.value, "idempotent": "false"}
        except FramesleutheException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, object]:
        job = await state.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=JobNotFoundError(job_id).to_dict())
        return {
            "id": job.id,
            "state": job.state.value,
            "progress_pct": job.progress_pct,
            "bundle_path": job.bundle_path,
            "error": job.error_json,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    @app.get("/v1/report/{job_id}")
    async def get_report(job_id: str) -> dict[str, object]:
        job = await state.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=JobNotFoundError(job_id).to_dict())
        if not job.bundle_path:
            raise HTTPException(
                status_code=409,
                detail={"error": "Report not ready", "code": "not_ready"},
            )

        bundle_path = Path(job.bundle_path)
        if not bundle_path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": "Report missing", "code": "missing_bundle"},
            )

        payload: dict[str, Any] = json.loads(bundle_path.read_text(encoding="utf-8"))
        return payload

    @app.get("/v1/video/{job_id}")
    async def get_video(job_id: str) -> FileResponse:
        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise HTTPException(status_code=404, detail=JobNotFoundError(job_id).to_dict())
        bundle_dir = Path(job.bundle_path).parent
        source = _find_source_video(bundle_dir)
        if source is None:
            raise HTTPException(
                status_code=404, detail={"error": "Video missing", "code": "missing_video"}
            )
        # Serve the correct content-type per container; the source may be any
        # supported format (mp4/mov/mkv/avi), not just webm.
        media_type = _VIDEO_MEDIA_TYPES.get(source.suffix.lower(), "application/octet-stream")
        return FileResponse(source, media_type=media_type)

    @app.get("/v1/gif/{job_id}")
    async def get_gif(
        job_id: str,
        fps: float = Query(default=None),
        width: float = Query(default=None),
        start: float = Query(default=0.0),
        end: float | None = Query(default=None),
    ) -> FileResponse:
        """Render (and cache) an animated GIF preview of the recording.

        The client embeds this looping preview wherever a video player is awkward
        (issues, chat, the extension popup). Query params ``fps``/``width``/
        ``start``/``end`` are optional and clamped to safe ranges; results are
        cached on disk per parameter set, so repeat requests are served instantly.
        """
        from framesleuth.pipeline.gif import encode_gif, normalize_options

        job = await state.store.get_job(job_id)
        if job is None or not job.bundle_path:
            raise HTTPException(status_code=404, detail=JobNotFoundError(job_id).to_dict())
        bundle_dir = Path(job.bundle_path).parent
        source = _find_source_video(bundle_dir)
        if source is None:
            raise HTTPException(
                status_code=404, detail={"error": "Video missing", "code": "missing_video"}
            )

        options = normalize_options(
            fps=fps if fps is not None else settings.GIF_FPS,
            width=width if width is not None else settings.GIF_WIDTH,
            start=start,
            end=end,
            max_duration_s=settings.GIF_MAX_DURATION_S,
        )
        gif_path = bundle_dir / f"preview-{options.cache_key()}.gif"
        if not gif_path.exists():
            result = encode_gif(source, gif_path, options=options)
            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail={"error": "GIF encoding failed", "code": "gif_failed"},
                )
        return FileResponse(gif_path, media_type="image/gif")

    @app.post("/v1/render-html")
    async def render_html_endpoint(payload: dict[str, Any]) -> FileResponse:
        """Render an HTML document (CSS / JS / canvas animation) to a clip.

        Body: ``{"html": "...", "format": "mp4|gif|webm", "duration_s": 5,
        "fps": 30, "width": 1280, "height": 720}``. Synchronous — returns the
        encoded file. Requires the optional ``render`` extra (Playwright) and
        ``ffmpeg``; returns ``503`` with an actionable message when unavailable.
        """
        from framesleuth.pipeline.html_render import (
            HtmlRenderError,
            RenderOptions,
            render_html,
        )

        html = payload.get("html")
        if not isinstance(html, str) or not html.strip():
            raise HTTPException(
                status_code=400, detail={"error": "Missing 'html'", "code": "missing_html"}
            )
        try:
            options = RenderOptions.normalized(
                fmt=str(payload.get("format", "mp4")),
                duration_s=float(payload.get("duration_s", 5.0)),
                fps=int(payload.get("fps", 30)),
                width=int(payload.get("width", 1280)),
                height=int(payload.get("height", 720)),
            )
        except (HtmlRenderError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail={"error": str(exc), "code": "bad_options"}
            ) from exc

        out_dir = Path(tempfile.mkdtemp(prefix="fs-render-"))
        try:
            path = await render_html(html, options, out_dir)
        except HtmlRenderError as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise HTTPException(
                status_code=503, detail={"error": str(exc), "code": "render_unavailable"}
            ) from exc

        media = {"mp4": "video/mp4", "gif": "image/gif", "webm": "video/webm"}[options.fmt]
        return FileResponse(
            path,
            media_type=media,
            filename=f"animation.{options.fmt}",
            background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
        )

    return app


app = create_app()


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the HTTP backend on the configured BACKEND_HOST/BACKEND_PORT."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.BACKEND_HOST, port=settings.BACKEND_PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
