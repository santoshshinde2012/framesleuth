"""Render an HTML document (CSS / JS / canvas animation) to a video or GIF.

This is the "design animation → shareable clip" capability: a self-contained
HTML page (e.g. one Claude generated) is loaded in a headless Chromium, its
animation is recorded for a bounded duration, and the recording is encoded to
``mp4`` / ``gif`` / ``webm``.

Unlike the rest of the media layer (which uses PyAV so no system ``ffmpeg`` is
required), this is an **optional, heavier** capability: it needs Playwright's
Chromium and the ``ffmpeg`` binary. Both imports/lookups are lazy and every
failure raises :class:`HtmlRenderError` with an actionable message rather than
crashing the server, so the rest of the agent is unaffected when they're absent.

    uv pip install -e ".[render]"
    playwright install chromium     # one-time
    # ffmpeg must be on PATH (brew install ffmpeg / apt-get install ffmpeg)
"""

from __future__ import annotations

import asyncio
import importlib.metadata as metadata
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framesleuth.logging_config import get_logger

logger = get_logger("pipeline.html_render")

SUPPORTED_FORMATS: tuple[str, ...] = ("mp4", "gif", "webm")

# Guard rails so a caller cannot ask for a 4K, 60fps, ten-minute render.
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 30.0
_MIN_FPS = 5
_MAX_FPS = 60
_GIF_MAX_FPS = 20
_MIN_DIM = 64
_MAX_WIDTH = 1920
_MAX_HEIGHT = 1080


class HtmlRenderError(Exception):
    """Raised when an HTML render cannot be produced (with an actionable message)."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


@dataclass(frozen=True)
class RenderOptions:
    """Normalized, clamped parameters for one HTML render."""

    fmt: str = "mp4"
    duration_s: float = 5.0
    fps: int = 30
    width: int = 1280
    height: int = 720

    @classmethod
    def normalized(
        cls,
        *,
        fmt: str = "mp4",
        duration_s: float = 5.0,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ) -> RenderOptions:
        fmt = (fmt or "mp4").lower()
        if fmt not in SUPPORTED_FORMATS:
            raise HtmlRenderError(f"format must be one of {', '.join(SUPPORTED_FORMATS)}")
        return cls(
            fmt=fmt,
            duration_s=_clamp(float(duration_s), _MIN_DURATION_S, _MAX_DURATION_S),
            fps=int(_clamp(float(fps), _MIN_FPS, _MAX_FPS)),
            width=int(_clamp(float(width), _MIN_DIM, _MAX_WIDTH)),
            height=int(_clamp(float(height), _MIN_DIM, _MAX_HEIGHT)),
        )


def _ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise HtmlRenderError(
            "ffmpeg is not installed (needed to encode mp4/gif). Install it and retry."
        )
    return exe


def _browsers_base_dir() -> Path:
    """Where Playwright stores downloaded browsers (env override or OS default)."""
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env and env not in ("0", "1"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA")
        return (Path(local) if local else Path.home()) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def _chromium_installed() -> bool:
    """Cheap, side-effect-free check that a Chromium build was downloaded."""
    base = _browsers_base_dir()
    try:
        return base.is_dir() and any(p.name.startswith("chromium") for p in base.iterdir())
    except OSError:
        return False


def render_availability() -> dict[str, Any]:
    """Report whether the optional HTML→video capability can run *in this process*.

    Side-effect free (no browser launch), so it's cheap enough to surface from
    ``/v1/healthz``. The common "I followed the README but it doesn't work" case
    is almost always that the server process is a different environment than the
    one the render extra was installed into, or that it wasn't restarted — this
    makes that visible (``playwright``/``chromium``/``ffmpeg`` flags + the exact
    ``python`` running) instead of failing opaquely only at render time.
    """
    info: dict[str, Any] = {
        "playwright": False,
        "playwright_version": None,
        "chromium": False,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "python": sys.executable,
        "ready": False,
        "hint": None,
    }
    try:
        info["playwright_version"] = metadata.version("playwright")
        info["playwright"] = True
    except metadata.PackageNotFoundError:
        info["playwright"] = False

    if info["playwright"]:
        info["chromium"] = _chromium_installed()

    info["ready"] = bool(info["playwright"] and info["chromium"] and info["ffmpeg"])
    if not info["ready"]:
        steps: list[str] = []
        if not info["playwright"]:
            steps.append('uv pip install -e ".[render]"')
        if not info["chromium"]:
            steps.append("playwright install chromium")
        if not info["ffmpeg"]:
            steps.append("install ffmpeg on PATH")
        info["hint"] = (
            "HTML→video is unavailable in this process. Run: "
            + " && ".join(steps)
            + " — then restart the server (it must be the same environment)."
        )
    return info


_MISSING_BROWSER = ("executable doesn't exist", "playwright install")
_MISSING_SYSDEPS = ("missing dependencies", "missing libraries", "error while loading")


def _launch_hint(exc: Exception) -> str:
    """Turn a Chromium launch failure into an actionable, single-line message."""
    raw = str(exc).strip()
    first = raw.splitlines()[0] if raw else repr(exc)
    low = raw.lower()
    if any(s in low for s in _MISSING_BROWSER):
        extra = " Run: playwright install chromium (then restart the server)."
    elif any(s in low for s in _MISSING_SYSDEPS):
        extra = " Run: playwright install --with-deps chromium (Linux libraries are missing)."
    else:
        extra = " Verify: playwright install chromium."
    return f"Could not launch headless Chromium: {first}.{extra}"


_browser_install_lock = asyncio.Lock()


def _auto_install_enabled() -> bool:
    """Whether to auto-download Chromium on first render (default on)."""
    return os.environ.get("FRAMESLEUTH_AUTO_INSTALL_BROWSER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


async def _ensure_chromium() -> None:
    """Download the Chromium build on demand (first render), serialized by a lock."""
    async with _browser_install_lock:
        if _chromium_installed():
            return  # a concurrent render already fetched it
        logger.info("Downloading Chromium for HTML→video (one-time, ~150 MB)…")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "playwright",
            "install",
            "chromium",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()[-300:]
            raise HtmlRenderError(
                "Could not auto-install Chromium — run `playwright install chromium` "
                f"manually (or set FRAMESLEUTH_AUTO_INSTALL_BROWSER=0). Details: {detail}"
            )
        logger.info("Chromium is ready.")


async def _record_webm(html: str, opts: RenderOptions, out_dir: Path) -> Path:
    """Render the HTML in headless Chromium and record a WebM of the animation."""
    try:
        from playwright.async_api import async_playwright  # lazy: optional dep
    except ModuleNotFoundError as exc:
        raise HtmlRenderError(
            "Playwright is not installed in this environment "
            f'(python: {sys.executable}). Run: uv pip install -e ".[render]" '
            "then restart the server (Chromium auto-downloads on the first render)."
        ) from exc
    except ImportError as exc:  # installed but broken (e.g. partial/ABI mismatch)
        raise HtmlRenderError(
            f"Playwright is installed but failed to import: {exc}. "
            'Reinstall with: uv pip install --reinstall -e ".[render]"'
        ) from exc

    size = {"width": opts.width, "height": opts.height}
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(args=["--no-sandbox"])
        except Exception as exc:
            # If only the browser binary is missing, fetch it once and retry —
            # so installing the `render` extra is the only manual step.
            if any(s in str(exc).lower() for s in _MISSING_BROWSER) and _auto_install_enabled():
                await _ensure_chromium()
                try:
                    browser = await pw.chromium.launch(args=["--no-sandbox"])
                except Exception as exc2:
                    raise HtmlRenderError(_launch_hint(exc2)) from exc2
            else:
                raise HtmlRenderError(_launch_hint(exc)) from exc
        context = await browser.new_context(
            viewport=size,
            record_video_dir=str(out_dir),
            record_video_size=size,
        )
        page = await context.new_page()
        try:
            await page.set_content(html, wait_until="load")
            # Let the animation play for the requested window.
            await page.wait_for_timeout(int(opts.duration_s * 1000))
        finally:
            # Closing the context flushes the recording to disk.
            await context.close()
            await browser.close()

    videos = sorted(out_dir.glob("*.webm"))
    if not videos:
        raise HtmlRenderError("The recording produced no video frames.")
    return videos[0]


async def _run_ffmpeg(args: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace")[-400:]
        raise HtmlRenderError(f"ffmpeg failed: {detail}")


async def render_html(html: str, opts: RenderOptions, out_dir: Path) -> Path:
    """Render ``html`` to a clip and return the path to the encoded file.

    The output format is ``opts.fmt`` (``mp4`` / ``gif`` / ``webm``). Raises
    :class:`HtmlRenderError` on any unrecoverable problem (missing Playwright /
    ffmpeg, launch failure, empty recording, encode failure).
    """
    if not html or not html.strip():
        raise HtmlRenderError("No HTML was provided.")
    out_dir.mkdir(parents=True, exist_ok=True)

    webm = await _record_webm(html, opts, out_dir)
    if opts.fmt == "webm":
        return webm

    ffmpeg = _ffmpeg_path()
    if opts.fmt == "mp4":
        out = out_dir / "render.mp4"
        await _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-i",
                str(webm),
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                str(out),
            ]
        )
        return out

    # gif: two-pass palette for clean colors at a sane frame rate.
    out = out_dir / "render.gif"
    gif_fps = min(opts.fps, _GIF_MAX_FPS)
    vf = (
        f"fps={gif_fps},scale={opts.width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    )
    await _run_ffmpeg([ffmpeg, "-y", "-i", str(webm), "-vf", vf, str(out)])
    return out
