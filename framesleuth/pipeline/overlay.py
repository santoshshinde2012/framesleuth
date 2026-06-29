"""Render click/cursor interaction markers onto keyframes before VLM analysis.

A screen recording's clicks live in the browser sidecars, not the pixels — so the
VLM (and a human reviewer) can't see *where* the user acted. When a click/cursor
event with usable coordinates lands near a keyframe's timestamp, this draws a
translucent marker on that frame in place, so "the user clicked here" becomes
visible evidence.

It is deliberately defensive: coordinates are only used when the event provides
them in a recognized form (normalized ``nx``/``ny`` in ``[0,1]``, or absolute
``x``/``y`` together with a viewport size from the env snapshot). When no reliable
mapping exists, the frame is left untouched — never guessed. No text is drawn, so
the OCR pass is not polluted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framesleuth.logging_config import get_logger
from framesleuth.schemas import KeyframeRef

logger = get_logger("pipeline.overlay")


def _viewport(env: dict[str, Any]) -> tuple[float, float] | None:
    """Extract a (width, height) viewport from the env snapshot, if present."""
    viewport = env.get("viewport")
    if isinstance(viewport, dict):
        width, height = viewport.get("width"), viewport.get("height")
    else:
        width = env.get("innerWidth") or env.get("width")
        height = env.get("innerHeight") or env.get("height")
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        return None
    w, h = float(width), float(height)
    return (w, h) if w > 0 and h > 0 else None


def _normalized_xy(event: dict[str, Any], env: dict[str, Any]) -> tuple[float, float] | None:
    """Return click position as fractions of frame size, or ``None`` if unmappable."""
    nx, ny = event.get("nx"), event.get("ny")
    if (
        isinstance(nx, int | float)
        and isinstance(ny, int | float)
        and 0.0 <= nx <= 1.0
        and 0.0 <= ny <= 1.0
    ):
        return (float(nx), float(ny))
    x = event.get("x", event.get("clientX"))
    y = event.get("y", event.get("clientY"))
    viewport = _viewport(env)
    if isinstance(x, int | float) and isinstance(y, int | float) and viewport is not None:
        vw, vh = viewport
        return (max(0.0, min(1.0, x / vw)), max(0.0, min(1.0, y / vh)))
    return None


def overlay_interactions(
    keyframes: list[KeyframeRef],
    frames_dir: Path,
    parsed: Any,
    *,
    window_s: float = 0.6,
) -> int:
    """Draw interaction markers on keyframes near a click/cursor event, in place.

    Args:
        keyframes: Keyframes whose images live under ``frames_dir``.
        frames_dir: Directory holding each keyframe's ``file``.
        parsed: Parsed sidecars (``clicks``/``cursor``/``env``).
        window_s: Max time gap between an event and a keyframe to mark it.

    Returns:
        The number of frames a marker was drawn on (0 if unavailable/no coords).
    """
    events = list(getattr(parsed, "clicks", []) or []) + list(getattr(parsed, "cursor", []) or [])
    if not keyframes or not events:
        return 0
    try:
        import cv2  # lazy: heavy media dep, only in the full stack
    except Exception:  # pragma: no cover - exercised only without cv2
        return 0

    env = getattr(parsed, "env", {}) or {}
    drawn = 0
    for keyframe in keyframes:
        nearest = _nearest_event(events, keyframe.t, window_s)
        if nearest is None:
            continue
        frac = _normalized_xy(nearest, env)
        if frac is None:
            continue
        if _draw_marker(cv2, frames_dir / keyframe.file, frac):
            drawn += 1
    if drawn:
        logger.info("Overlaid interaction markers on %d keyframe(s)", drawn)
    return drawn


def _nearest_event(
    events: list[dict[str, Any]], t: float, window_s: float
) -> dict[str, Any] | None:
    """Return the event closest in time to ``t`` within ``window_s``, if any."""
    best: dict[str, Any] | None = None
    best_gap = window_s
    for event in events:
        try:
            gap = abs(float(event.get("t", 0.0)) - t)
        except (TypeError, ValueError):
            continue
        if gap <= best_gap:
            best, best_gap = event, gap
    return best


def _draw_marker(cv2: Any, image_path: Path, frac: tuple[float, float]) -> bool:
    """Draw a translucent ring + crosshair at ``frac`` of the image; return success."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        h, w = image.shape[:2]
        cx, cy = int(frac[0] * w), int(frac[1] * h)
        radius = max(8, int(0.03 * max(w, h)))
        overlay = image.copy()
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), thickness=3)
        cv2.line(overlay, (cx - radius, cy), (cx + radius, cy), (0, 0, 255), 1)
        cv2.line(overlay, (cx, cy - radius), (cx, cy + radius), (0, 0, 255), 1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        return bool(cv2.imwrite(str(image_path), image))
    except Exception as exc:  # pragma: no cover - defensive, never abort analysis
        logger.debug("Overlay draw skipped for %s: %s", image_path, exc)
        return False
