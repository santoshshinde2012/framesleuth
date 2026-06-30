"""Render an HTML document (CSS / JS / canvas animation) to a video or GIF.

This is the "design animation → shareable clip" capability: a self-contained
HTML page (e.g. one Claude generated) is loaded in a headless Chromium and
captured **frame-by-frame** under a paused virtual clock — each frame a lossless,
full-resolution PNG at an exact timestamp, so there are no dropped frames and no
color loss (unlike screen recording). The PNG sequence is then encoded to a
color-correct ``mp4`` (H.264, ``yuv420p``+``bt709``, near-lossless), ``webm``
(VP9), or palette ``gif``. If deterministic capture is unavailable on the running
Chromium, it falls back to real-time recording so the feature still works.

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
import base64
import contextlib
import importlib.metadata as metadata
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Optional dependency (the `render` extra). Imported for typing only so the
    # viewport dict matches Playwright's ViewportSize TypedDict without making
    # playwright a hard import at module load.
    from playwright.async_api import ViewportSize

from framesleuth.logging_config import get_logger

logger = get_logger("pipeline.html_render")

SUPPORTED_FORMATS: tuple[str, ...] = ("mp4", "gif", "webm")

# Guard rails. Resolution goes up to 4K so exports are crisp. Duration is no longer
# capped at a tight 30 s — the whole animation is captured (auto-detected when the
# caller omits a duration). The real safety bound is the total frame count
# (duration x fps), since frame-by-frame capture materializes one lossless PNG per
# frame; ``_MAX_FRAMES`` keeps a long/high-fps/high-res render from exhausting disk.
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 300.0
_MAX_FRAMES = 18000
_DEFAULT_DURATION_S = 5.0
_MIN_FPS = 5
_MAX_FPS = 60
_GIF_MAX_FPS = 25
_MIN_DIM = 64
_MAX_WIDTH = 3840
_MAX_HEIGHT = 2160

# Near-lossless H.264 (visually transparent) so exported colors match the source.
_H264_CRF = "16"
# VP9 quality (lower = better); 24 is high quality at a sane size.
_VP9_CRF = "24"


class HtmlRenderError(Exception):
    """Raised when an HTML render cannot be produced (with an actionable message)."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


@dataclass(frozen=True)
class RenderOptions:
    """Normalized, clamped parameters for one HTML render.

    ``duration_s`` is ``None`` to capture the **whole animation** — its length is
    auto-detected at render time. A concrete value records exactly that window.
    ``max_duration_s`` / ``max_frames`` bound the work; ``default_duration_s`` is the
    fallback used only when nothing declarative is detectable.
    """

    fmt: str = "mp4"
    duration_s: float | None = None
    fps: int = 30
    width: int = 1280
    height: int = 720
    max_duration_s: float = _MAX_DURATION_S
    max_frames: int = _MAX_FRAMES
    default_duration_s: float = _DEFAULT_DURATION_S

    @classmethod
    def normalized(
        cls,
        *,
        fmt: str = "mp4",
        duration_s: float | None = None,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        max_duration_s: float = _MAX_DURATION_S,
        max_frames: int = _MAX_FRAMES,
        default_duration_s: float = _DEFAULT_DURATION_S,
    ) -> RenderOptions:
        fmt = (fmt or "mp4").lower()
        if fmt not in SUPPORTED_FORMATS:
            raise HtmlRenderError(f"format must be one of {', '.join(SUPPORTED_FORMATS)}")
        cap = max(_MIN_DURATION_S, float(max_duration_s))
        # None => auto-detect the animation's full length at capture time.
        resolved = None if duration_s is None else _clamp(float(duration_s), _MIN_DURATION_S, cap)
        return cls(
            fmt=fmt,
            duration_s=resolved,
            fps=int(_clamp(float(fps), _MIN_FPS, _MAX_FPS)),
            width=int(_clamp(float(width), _MIN_DIM, _MAX_WIDTH)),
            height=int(_clamp(float(height), _MIN_DIM, _MAX_HEIGHT)),
            max_duration_s=cap,
            max_frames=max(1, int(max_frames)),
            default_duration_s=_clamp(float(default_duration_s), _MIN_DURATION_S, cap),
        )


def _ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise HtmlRenderError(
            "ffmpeg is not installed (needed to encode mp4/gif). Install it and retry."
        )
    return exe


def _frame_count(duration_s: float, fps: int) -> int:
    """Number of lossless frames to capture for a duration at fps (at least 1)."""
    return max(1, round(duration_s * fps))


def _bounded_frame_count(duration_s: float, fps: int, max_frames: int) -> tuple[int, bool]:
    """Frame count for ``duration`` at ``fps``, capped at ``max_frames``.

    Returns ``(n, truncated)`` — ``truncated`` is ``True`` when the cap reduced the
    count, so the caller can warn that the tail of a very long render was dropped.
    """
    n = _frame_count(duration_s, fps)
    if n > max_frames:
        return max(1, max_frames), True
    return n, False


# Detect an animation's full length from the page: an explicit hint
# (``window.__renderDurationMs`` or ``<body data-render-duration-ms>``) wins; else
# the longest CSS animation/transition (one cycle for infinite loops) and Web
# Animations API timeline. Returns milliseconds, or 0 when nothing is detectable
# (e.g. a pure canvas/requestAnimationFrame loop with no hint). Defensive — any
# error yields 0 so the caller falls back to the default duration.
_DURATION_DETECT_JS = r"""
() => {
  try {
    const hint = window.__renderDurationMs;
    if (typeof hint === 'number' && isFinite(hint) && hint > 0) return hint;
    const dataHint = (document.body && document.body.dataset)
      ? parseFloat(document.body.dataset.renderDurationMs) : NaN;
    if (isFinite(dataHint) && dataHint > 0) return dataHint;

    const toMs = (v) => {
      if (!v) return 0;
      v = String(v).trim();
      if (v.endsWith('ms')) return parseFloat(v) || 0;
      if (v.endsWith('s')) return (parseFloat(v) || 0) * 1000;
      return parseFloat(v) || 0;
    };
    let max = 0;
    const at = (arr, i) => (arr[i] !== undefined ? arr[i] : arr[0]);

    for (const el of document.querySelectorAll('*')) {
      const cs = getComputedStyle(el);
      const aDur = (cs.animationDuration || '').split(',');
      const aDel = (cs.animationDelay || '').split(',');
      const aIt = (cs.animationIterationCount || '1').split(',');
      for (let i = 0; i < aDur.length; i++) {
        const d = toMs(aDur[i]);
        if (d <= 0) continue;
        const delay = toMs(at(aDel, i));
        const itRaw = String(at(aIt, i) || '1').trim();
        const it = itRaw === 'infinite' ? 1 : (parseFloat(itRaw) || 1);
        max = Math.max(max, delay + d * it);
      }
      const tDur = (cs.transitionDuration || '').split(',');
      const tDel = (cs.transitionDelay || '').split(',');
      for (let i = 0; i < tDur.length; i++) {
        const d = toMs(tDur[i]);
        if (d <= 0) continue;
        max = Math.max(max, toMs(at(tDel, i)) + d);
      }
    }

    if (document.getAnimations) {
      for (const a of document.getAnimations()) {
        const t = (a.effect && a.effect.getComputedTiming) ? a.effect.getComputedTiming() : null;
        if (!t) continue;
        const dur = (typeof t.duration === 'number' && isFinite(t.duration)) ? t.duration : 0;
        const inf = (t.iterations === Infinity || !isFinite(t.iterations));
        const iter = inf ? 1 : (t.iterations || 1);
        max = Math.max(max, (t.delay || 0) + dur * iter + (t.endDelay || 0));
      }
    }
    return max;
  } catch (e) {
    return 0;
  }
}
"""


async def _detect_duration_ms(page: Any) -> float:
    """Evaluate the detection script on the page; return milliseconds (0 if none)."""
    try:
        value = await page.evaluate(_DURATION_DETECT_JS)
        ms = float(value)
    except Exception as exc:  # any page/eval error → fall back to the default
        logger.debug("Animation duration detection failed: %s", exc)
        return 0.0
    return ms if ms > 0 and math.isfinite(ms) else 0.0


async def _effective_duration_s(page: Any, opts: RenderOptions) -> float:
    """Resolve the capture window: the caller's value, or the detected animation length."""
    if opts.duration_s is not None:
        return opts.duration_s
    detected = (await _detect_duration_ms(page)) / 1000.0
    duration = detected if detected > 0 else opts.default_duration_s
    if detected > 0:
        logger.info("Auto-detected animation length: %.2fs", detected)
    return _clamp(duration, _MIN_DURATION_S, opts.max_duration_s)


def _frames_glob(frames_dir: Path) -> str:
    """Input pattern (for ffmpeg) over the captured PNG sequence."""
    return str(frames_dir / "%05d.png")


def _mp4_args(ffmpeg: str, frames_glob: str, fps: int, out: Path) -> list[str]:
    """Encode the PNG sequence to web-ready, color-correct H.264 MP4.

    ``-crf 16`` is visually transparent (no banding/quality loss), ``yuv420p`` +
    ``bt709`` tags keep colors correct across players, ``+faststart`` makes it
    stream/seek immediately. Exact ``-framerate`` because frames are deterministic.
    """
    return [
        ffmpeg, "-y", "-framerate", str(fps), "-i", frames_glob,
        "-c:v", "libx264", "-preset", "medium", "-crf", _H264_CRF,
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def _webm_args(ffmpeg: str, frames_glob: str, fps: int, out: Path) -> list[str]:
    """Encode the PNG sequence to high-quality VP9 WebM."""
    return [
        ffmpeg, "-y", "-framerate", str(fps), "-i", frames_glob,
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
        "-b:v", "0", "-crf", _VP9_CRF, "-row-mt", "1", str(out),
    ]  # fmt: skip


def _gif_args(ffmpeg: str, frames_glob: str, fps: int, width: int, out: Path) -> list[str]:
    """Encode the PNG sequence to a GIF via a per-clip palette (clean colors)."""
    gif_fps = min(fps, _GIF_MAX_FPS)
    vf = (
        f"fps={gif_fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    return [ffmpeg, "-y", "-framerate", str(fps), "-i", frames_glob, "-vf", vf, str(out)]


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


def _import_playwright() -> Any:
    """Lazily import ``async_playwright`` with actionable error messages."""
    try:
        from playwright.async_api import async_playwright  # lazy: optional dep

        return async_playwright
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


async def _launch_browser(pw: Any) -> Any:
    """Launch headless Chromium, auto-installing it once if the binary is missing."""
    try:
        return await pw.chromium.launch(args=["--no-sandbox"])
    except Exception as exc:
        if any(s in str(exc).lower() for s in _MISSING_BROWSER) and _auto_install_enabled():
            await _ensure_chromium()
            try:
                return await pw.chromium.launch(args=["--no-sandbox"])
            except Exception as exc2:
                raise HtmlRenderError(_launch_hint(exc2)) from exc2
        raise HtmlRenderError(_launch_hint(exc)) from exc


async def _advance_virtual_time(cdp: Any, budget_ms: float) -> None:
    """Advance the renderer's virtual clock by ``budget_ms`` and wait for it to drain.

    Under a paused virtual-time policy the whole renderer clock — ``requestAnimation
    Frame``, ``setTimeout``, CSS animations, Web Animations — is frozen, so advancing
    a fixed budget per frame produces deterministic, evenly spaced frames regardless
    of how heavy the animation is (no dropped or duplicated frames).
    """
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[None] = loop.create_future()

    def _on_expired(_: Any) -> None:
        if not fut.done():
            fut.set_result(None)

    cdp.on("Emulation.virtualTimeBudgetExpired", _on_expired)
    try:
        await cdp.send(
            "Emulation.setVirtualTimePolicy",
            {"policy": "advance", "budget": budget_ms},
        )
        await asyncio.wait_for(fut, timeout=15.0)
    finally:
        # Listener-API differences across Playwright versions must never break a render.
        with contextlib.suppress(Exception):
            cdp.remove_listener("Emulation.virtualTimeBudgetExpired", _on_expired)


async def _capture_frames(html: str, opts: RenderOptions, frames_dir: Path) -> int:
    """Capture a deterministic, lossless PNG per frame via CDP virtual time.

    This is the high-fidelity path (like a frame-by-frame exporter): each frame is
    a full-quality PNG screenshot taken at an exact virtual timestamp, so the result
    has no dropped frames and no color loss — the encoder just muxes them at the
    requested fps. Returns the number of frames written.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    async_playwright = _import_playwright()
    frame_ms = 1000.0 / opts.fps
    size: ViewportSize = {"width": opts.width, "height": opts.height}

    async with async_playwright() as pw:
        browser = await _launch_browser(pw)
        context = await browser.new_context(viewport=size, device_scale_factor=1)
        page = await context.new_page()
        cdp = await context.new_cdp_session(page)
        try:
            await page.set_content(html, wait_until="load")
            # Let fonts/first paint settle before the clock is frozen.
            with contextlib.suppress(Exception):
                await page.evaluate("document.fonts && document.fonts.ready")
            await page.wait_for_timeout(60)
            # Resolve how long to record — auto-detect the whole animation when the
            # caller didn't pin a duration — then bound the frame count for safety.
            duration_s = await _effective_duration_s(page, opts)
            n, truncated = _bounded_frame_count(duration_s, opts.fps, opts.max_frames)
            if truncated:
                logger.warning(
                    "Render capped at %d frames (%.1fs @ %dfps exceeds RENDER_MAX_FRAMES=%d); "
                    "raise the cap or lower fps/resolution to capture the full tail.",
                    n,
                    duration_s,
                    opts.fps,
                    opts.max_frames,
                )
            # Freeze the clock, then step one frame budget at a time.
            await cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
            for i in range(n):
                if i > 0:
                    await _advance_virtual_time(cdp, frame_ms)
                shot = await cdp.send(
                    "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
                )
                (frames_dir / f"{i:05d}.png").write_bytes(base64.b64decode(shot["data"]))
        finally:
            await context.close()
            await browser.close()

    if not any(frames_dir.glob("*.png")):
        raise HtmlRenderError("Frame capture produced no images.")
    return n


async def _record_webm(html: str, opts: RenderOptions, out_dir: Path) -> Path:
    """Fallback: record a real-time WebM of the animation in headless Chromium.

    Used only when deterministic frame capture is unavailable (older Chromium /
    CDP virtual-time failure). Lower fidelity than frame-by-frame because the
    recorder samples in real time, but keeps the feature working.
    """
    async_playwright = _import_playwright()
    size: ViewportSize = {"width": opts.width, "height": opts.height}
    async with async_playwright() as pw:
        browser = await _launch_browser(pw)
        context = await browser.new_context(
            viewport=size,
            record_video_dir=str(out_dir),
            record_video_size=size,
        )
        page = await context.new_page()
        try:
            await page.set_content(html, wait_until="load")
            # Let the animation play for its full (auto-detected or requested) window.
            duration_s = await _effective_duration_s(page, opts)
            await page.wait_for_timeout(int(duration_s * 1000))
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


def _force_realtime() -> bool:
    """Opt-in escape hatch to skip frame-by-frame capture (debug/diagnostics)."""
    return os.environ.get("FRAMESLEUTH_RENDER_MODE", "").strip().lower() == "realtime"


async def _encode_from_frames(opts: RenderOptions, frames_dir: Path, out_dir: Path) -> Path:
    """Encode the captured PNG sequence to the requested format (color-correct)."""
    ffmpeg = _ffmpeg_path()
    glob = _frames_glob(frames_dir)
    if opts.fmt == "mp4":
        out = out_dir / "render.mp4"
        await _run_ffmpeg(_mp4_args(ffmpeg, glob, opts.fps, out))
    elif opts.fmt == "webm":
        out = out_dir / "render.webm"
        await _run_ffmpeg(_webm_args(ffmpeg, glob, opts.fps, out))
    else:  # gif
        out = out_dir / "render.gif"
        await _run_ffmpeg(_gif_args(ffmpeg, glob, opts.fps, opts.width, out))
    return out


async def _render_realtime(html: str, opts: RenderOptions, out_dir: Path) -> Path:
    """Real-time recording fallback: record WebM, then transcode if needed."""
    webm = await _record_webm(html, opts, out_dir)
    if opts.fmt == "webm":
        return webm
    ffmpeg = _ffmpeg_path()
    if opts.fmt == "mp4":
        out = out_dir / "render.mp4"
        await _run_ffmpeg(
            [
                ffmpeg, "-y", "-i", str(webm),
                "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out),
            ]  # fmt: skip
        )
        return out
    out = out_dir / "render.gif"
    await _run_ffmpeg(_gif_args(ffmpeg, str(webm), opts.fps, opts.width, out))
    return out


async def render_html(html: str, opts: RenderOptions, out_dir: Path) -> Path:
    """Render ``html`` to a clip and return the path to the encoded file.

    Primary path: **frame-by-frame** — capture a lossless PNG per frame at an exact
    virtual timestamp, then encode at the requested fps. This preserves every color
    and drops no frames (matching dedicated animation exporters). If deterministic
    capture is unavailable (older Chromium / CDP virtual-time failure) it falls back
    to real-time recording so the feature still works.

    The output format is ``opts.fmt`` (``mp4`` / ``gif`` / ``webm``). Raises
    :class:`HtmlRenderError` on missing Playwright/ffmpeg or an unrecoverable encode.
    """
    if not html or not html.strip():
        raise HtmlRenderError("No HTML was provided.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _force_realtime():
        frames_dir = out_dir / "frames"
        try:
            await _capture_frames(html, opts, frames_dir)
            return await _encode_from_frames(opts, frames_dir, out_dir)
        except HtmlRenderError:
            raise  # missing deps / encode failure — surface the actionable message
        except Exception as exc:  # capture-specific failure → degrade, don't fail
            logger.warning("Frame-by-frame capture failed (%s); using real-time recording.", exc)

    return await _render_realtime(html, opts, out_dir)
