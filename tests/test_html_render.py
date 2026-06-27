"""Tests for the HTML -> video/gif render options and input guards.

These cover the deterministic, dependency-free surface: option normalization
(clamping + format validation) and the empty-input guard. The actual Chromium
recording / ffmpeg encode is an optional capability exercised in integration,
not here, so the unit suite stays fast and runs without Playwright or ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from framesleuth.pipeline.html_render import (
    SUPPORTED_FORMATS,
    HtmlRenderError,
    RenderOptions,
    _auto_install_enabled,
    render_availability,
    render_html,
)


def test_auto_install_browser_defaults_on_and_is_opt_out(monkeypatch) -> None:
    """Chromium auto-downloads on first render unless explicitly disabled."""
    monkeypatch.delenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", raising=False)
    assert _auto_install_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", off)
        assert _auto_install_enabled() is False
    monkeypatch.setenv("FRAMESLEUTH_AUTO_INSTALL_BROWSER", "1")
    assert _auto_install_enabled() is True


def test_defaults_are_sane() -> None:
    opts = RenderOptions.normalized()
    assert opts.fmt == "mp4"
    assert opts.duration_s == 5.0
    assert opts.fps == 30
    assert opts.width == 1280
    assert opts.height == 720


def test_normalized_clamps_out_of_range_values() -> None:
    """A caller cannot ask for a 4K, 120fps, ten-minute render."""
    opts = RenderOptions.normalized(
        fmt="webm", duration_s=600.0, fps=240, width=99999, height=99999
    )
    assert opts.duration_s == 30.0  # capped at _MAX_DURATION_S
    assert opts.fps == 60  # capped at _MAX_FPS
    assert opts.width == 1920  # capped at _MAX_WIDTH
    assert opts.height == 1080  # capped at _MAX_HEIGHT


def test_normalized_clamps_below_minimums() -> None:
    opts = RenderOptions.normalized(duration_s=0.0, fps=1, width=1, height=1)
    assert opts.duration_s == 0.5
    assert opts.fps == 5
    assert opts.width == 64
    assert opts.height == 64


def test_normalized_is_case_insensitive_on_format() -> None:
    assert RenderOptions.normalized(fmt="MP4").fmt == "mp4"
    assert RenderOptions.normalized(fmt="GIF").fmt == "gif"


@pytest.mark.parametrize("bad", ["tiff", "avi", "mov", "mp5"])
def test_normalized_rejects_unsupported_format(bad: str) -> None:
    with pytest.raises(HtmlRenderError):
        RenderOptions.normalized(fmt=bad)


def test_normalized_empty_format_falls_back_to_default() -> None:
    """An empty/None format is treated as the default (mp4), not an error."""
    assert RenderOptions.normalized(fmt="").fmt == "mp4"


def test_supported_formats_are_the_three_advertised() -> None:
    assert set(SUPPORTED_FORMATS) == {"mp4", "gif", "webm"}


def test_options_are_immutable() -> None:
    opts = RenderOptions.normalized()
    with pytest.raises((AttributeError, TypeError)):
        opts.fps = 99  # type: ignore[misc]


async def test_render_html_rejects_empty_html(tmp_path: Path) -> None:
    """Empty/whitespace HTML fails fast before any Chromium/ffmpeg work."""
    opts = RenderOptions.normalized()
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(HtmlRenderError):
            await render_html(bad, opts, tmp_path / "out")


def test_render_availability_reports_a_stable_shape() -> None:
    """Availability probe is side-effect free and always returns the full shape."""
    info = render_availability()
    for key in ("playwright", "chromium", "ffmpeg", "python", "ready", "hint"):
        assert key in info
    assert isinstance(info["playwright"], bool)
    assert isinstance(info["chromium"], bool)
    assert isinstance(info["ffmpeg"], bool)
    assert isinstance(info["ready"], bool)
    assert isinstance(info["python"], str) and info["python"]
    # `ready` is only true when every dependency is present; otherwise a hint exists.
    assert info["ready"] == (info["playwright"] and info["chromium"] and info["ffmpeg"])
    if not info["ready"]:
        assert isinstance(info["hint"], str) and info["hint"]
