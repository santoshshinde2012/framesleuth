"""Encode an animated GIF preview from a stored recording.

A GIF is the most portable way for a client (a capture extension, a chat
surface, a GitHub issue) to embed a short looping preview of the bug without
shipping a video player. Like the rest of the media layer this uses PyAV only —
which bundles its own ffmpeg libraries, so no system ``ffmpeg`` binary or extra
imaging dependency (Pillow/imageio) is required. Every call degrades to ``None``
on a missing wheel, an unreadable container, or a decode error rather than
raising, mirroring :mod:`framesleuth.pipeline.preprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framesleuth.logging_config import get_logger

logger = get_logger("pipeline.gif")

# Guard rails so a caller (or a hostile query string) cannot ask for a 4K, 60fps,
# ten-minute GIF that would balloon to hundreds of megabytes.
_MIN_FPS = 1
_MAX_FPS = 30
_MIN_WIDTH = 64
_MAX_WIDTH = 1280


@dataclass(frozen=True)
class GifOptions:
    """Normalized, clamped parameters for one GIF render."""

    fps: int
    width: int
    start: float
    end: float  # exclusive upper bound in source-time seconds

    def cache_key(self) -> str:
        """Stable filename stem encoding the parameters, for on-disk caching."""
        return f"{self.fps}fps-{self.width}w-{self.start:.3f}-{self.end:.3f}"


def normalize_options(
    *,
    fps: float,
    width: float,
    start: float,
    end: float | None,
    max_duration_s: float,
) -> GifOptions:
    """Clamp raw request parameters into a safe :class:`GifOptions`.

    ``start``/``end`` are in source-time seconds; the rendered window is capped at
    ``max_duration_s`` so the output stays small regardless of clip length. A
    missing or non-positive ``end`` means "from ``start`` for ``max_duration_s``".
    """
    safe_fps = int(min(_MAX_FPS, max(_MIN_FPS, round(fps))))
    safe_width = int(min(_MAX_WIDTH, max(_MIN_WIDTH, round(width))))
    safe_start = max(0.0, float(start))
    if end is None or end <= safe_start:
        safe_end = safe_start + max_duration_s
    else:
        safe_end = min(float(end), safe_start + max_duration_s)
    return GifOptions(fps=safe_fps, width=safe_width, start=safe_start, end=safe_end)


def _decode_window(container: Any, vstream: Any, opts: GifOptions) -> list[Any]:
    """Decode RGB frames within ``[start, end)`` subsampled to ``opts.fps``.

    Seeks to the keyframe at/before ``start`` then walks forward, keeping at most
    one frame per ``1/fps`` slot so the GIF plays at the requested rate without
    re-encoding every source frame.
    """
    import cv2

    time_base = vstream.time_base
    if time_base:
        container.seek(int(opts.start / time_base), stream=vstream, backward=True)

    step = 1.0 / opts.fps
    next_t = opts.start
    frames: list[Any] = []
    for frame in container.decode(vstream):
        if frame.pts is None or not time_base:
            # No usable timestamps: fall back to taking every decoded frame.
            arr = frame.to_ndarray(format="rgb24")
            frames.append(_resize(arr, opts.width, cv2))
            if len(frames) >= int((opts.end - opts.start) * opts.fps) + 1:
                break
            continue
        t = float(frame.pts * time_base)
        if t < opts.start - 1e-3:
            continue
        if t >= opts.end:
            break
        if t + 1e-6 < next_t:
            continue  # still inside the current fps slot; skip
        frames.append(_resize(frame.to_ndarray(format="rgb24"), opts.width, cv2))
        next_t += step
    return frames


def _resize(arr: Any, width: int, cv2: Any) -> Any:
    """Downscale ``arr`` to ``width`` (preserving aspect); never upscale."""
    h, w = arr.shape[:2]
    if w > width:
        arr = cv2.resize(arr, (width, max(1, int(h * width / w))))
    return arr


def _mux_gif(out_path: Path, frames: list[Any], fps: int) -> None:
    """Write ``frames`` to a GIF container at ``out_path`` via PyAV."""
    import av

    height, width = frames[0].shape[:2]
    with av.open(str(out_path), mode="w", format="gif") as out_container:
        out_stream = out_container.add_stream("gif", rate=fps)
        # rgb8 is ffmpeg's built-in fixed palette: good enough for a UI preview
        # and avoids a palettegen/paletteuse filter graph (and its extra deps).
        out_stream.pix_fmt = "rgb8"
        out_stream.width = width
        out_stream.height = height
        for arr in frames:
            vframe = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in out_stream.encode(vframe):
                out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)


def encode_gif(
    video_path: Path,
    out_path: Path,
    *,
    options: GifOptions,
) -> Path | None:
    """Render ``video_path`` to an animated GIF at ``out_path``.

    Returns ``out_path`` on success, or ``None`` if PyAV/OpenCV are unavailable,
    the source has no decodable video, or any decode/encode step fails. The output
    is written atomically (to a sibling ``.tmp`` then renamed) so a partial file is
    never served from cache.
    """
    if not video_path.exists():
        logger.warning("GIF source missing: %s", video_path)
        return None
    try:
        import av  # lazy: heavy media deps only present in the full stack
    except Exception as exc:  # pragma: no cover - exercised only without media deps
        logger.warning("GIF encoding unavailable (av/cv2 missing): %s", exc)
        return None

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        logger.warning("Could not open %s for GIF encoding: %s", video_path, exc)
        return None

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        if not container.streams.video:
            logger.warning("No video stream in %s; cannot build GIF", video_path)
            return None
        vstream = container.streams.video[0]
        frames = _decode_window(container, vstream, options)
        if not frames:
            logger.warning("No frames decoded from %s for GIF window", video_path)
            return None

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _mux_gif(tmp_path, frames, options.fps)
    except Exception as exc:
        logger.warning("GIF encoding failed for %s: %s", video_path, exc)
        tmp_path.unlink(missing_ok=True)
        return None
    finally:
        container.close()

    try:
        tmp_path.replace(out_path)
    except OSError as exc:
        logger.warning("Could not finalize GIF at %s: %s", out_path, exc)
        tmp_path.unlink(missing_ok=True)
        return None
    logger.info("Encoded GIF (%d frames) to %s", len(frames), out_path)
    return out_path
