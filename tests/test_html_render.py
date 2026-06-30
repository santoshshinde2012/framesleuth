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
    _bounded_frame_count,
    _detect_duration_ms,
    _effective_duration_s,
    _frame_count,
    _gif_args,
    _mp4_args,
    _webm_args,
    render_availability,
    render_html,
)


class _FakePage:
    """Minimal page double whose ``evaluate`` returns (or raises) a canned value."""

    def __init__(self, value: object) -> None:
        self._value = value
        self.evaluated = False

    async def evaluate(self, _script: str) -> object:
        self.evaluated = True
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


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
    assert opts.duration_s is None  # None = capture the whole animation (auto-detect)
    assert opts.fps == 30
    assert opts.width == 1280
    assert opts.height == 720


def test_omitting_duration_means_auto_detect() -> None:
    """No duration → auto-detect the whole animation; a value pins the window."""
    assert RenderOptions.normalized(duration_s=None).duration_s is None
    assert RenderOptions.normalized(duration_s=12.0).duration_s == 12.0


def test_normalized_clamps_out_of_range_values() -> None:
    """fps/resolution are capped; duration is capped at the (configurable) max."""
    opts = RenderOptions.normalized(
        fmt="webm", duration_s=99999.0, fps=240, width=99999, height=99999
    )
    assert opts.duration_s == 300.0  # capped at the default _MAX_DURATION_S (was 30)
    assert opts.fps == 60  # capped at _MAX_FPS
    assert opts.width == 3840  # capped at _MAX_WIDTH (4K)
    assert opts.height == 2160  # capped at _MAX_HEIGHT (4K)


def test_duration_cap_is_configurable() -> None:
    """A longer animation is allowed when the caller raises the cap."""
    opts = RenderOptions.normalized(duration_s=120.0, max_duration_s=600.0)
    assert opts.duration_s == 120.0  # no longer clamped to 30
    capped = RenderOptions.normalized(duration_s=1000.0, max_duration_s=600.0)
    assert capped.duration_s == 600.0


def test_bounded_frame_count_caps_total_frames() -> None:
    """The frame-count guard bounds disk use regardless of duration x fps."""
    n, truncated = _bounded_frame_count(10.0, 30, max_frames=18000)
    assert (n, truncated) == (300, False)
    n, truncated = _bounded_frame_count(600.0, 60, max_frames=18000)  # 36000 requested
    assert n == 18000 and truncated is True


# --------------------------------------------------------------------------- #
# Auto-duration resolution (the "capture the whole animation" logic)
# --------------------------------------------------------------------------- #


async def test_detect_duration_ms_parses_a_number() -> None:
    assert await _detect_duration_ms(_FakePage(1200)) == 1200.0


async def test_detect_duration_ms_is_zero_on_bad_or_error_value() -> None:
    """A non-number, non-finite, zero, or thrown error all mean 'undetectable'."""
    assert await _detect_duration_ms(_FakePage("nope")) == 0.0
    assert await _detect_duration_ms(_FakePage(float("inf"))) == 0.0
    assert await _detect_duration_ms(_FakePage(0)) == 0.0
    assert await _detect_duration_ms(_FakePage(RuntimeError("boom"))) == 0.0


async def test_effective_duration_uses_explicit_value_without_probing() -> None:
    """A pinned duration is used verbatim — the page is never evaluated."""
    opts = RenderOptions.normalized(duration_s=7.0)
    page = _FakePage(RuntimeError("evaluate must not be called"))
    assert await _effective_duration_s(page, opts) == 7.0
    assert page.evaluated is False


async def test_effective_duration_auto_detects_full_length() -> None:
    """With no pinned duration, the detected animation length (ms→s) is used."""
    opts = RenderOptions.normalized(duration_s=None, default_duration_s=5.0)
    assert await _effective_duration_s(_FakePage(2500), opts) == 2.5


async def test_effective_duration_falls_back_to_default_when_undetectable() -> None:
    """A pure canvas loop (nothing detected) falls back to the configured default."""
    opts = RenderOptions.normalized(duration_s=None, default_duration_s=4.0)
    assert await _effective_duration_s(_FakePage(0), opts) == 4.0


async def test_effective_duration_clamps_detected_to_max() -> None:
    opts = RenderOptions.normalized(duration_s=None, max_duration_s=3.0)
    assert await _effective_duration_s(_FakePage(99_999), opts) == 3.0


@pytest.mark.integration
async def test_render_captures_whole_animation_via_auto_duration(tmp_path: Path) -> None:
    """End-to-end: a 1 s CSS animation auto-detects to ~1 s and captures all frames.

    Skipped unless the optional ``render`` extra (Playwright + Chromium) is present,
    so the unit suite stays dependency-free while CI with the extra exercises the
    in-browser duration detection and frame capture for real.
    """
    info = render_availability()
    if not (info["playwright"] and info["chromium"]):
        pytest.skip("render extra (Playwright + Chromium) not installed")

    from framesleuth.pipeline.html_render import _capture_frames

    html = (
        "<!doctype html><html><head><style>"
        "@keyframes f{from{opacity:1}to{opacity:0}}"
        ".b{width:100px;height:100px;background:#ec5b2a;animation:f 1s linear 1}"
        "</style></head><body><div class='b'></div></body></html>"
    )
    opts = RenderOptions.normalized(duration_s=None, fps=10)  # auto → ~1 s
    n = await _capture_frames(html, opts, tmp_path / "frames")

    # ~1 s at 10 fps ≈ 10 frames (slack for the first frame + rounding).
    assert 8 <= n <= 13
    assert len(list((tmp_path / "frames").glob("*.png"))) == n


def test_normalized_allows_1080p() -> None:
    """Full-quality 1080p exports pass through unclamped."""
    opts = RenderOptions.normalized(width=1920, height=1080)
    assert (opts.width, opts.height) == (1920, 1080)


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


def test_frame_count_is_duration_times_fps_min_one() -> None:
    """Frame-by-frame capture materializes duration x fps frames (at least one)."""
    assert _frame_count(5.0, 30) == 150
    assert _frame_count(2.0, 60) == 120
    assert _frame_count(0.0, 30) == 1  # never zero frames
    assert _frame_count(0.5, 24) == 12


def test_mp4_args_are_color_correct_and_frame_accurate() -> None:
    """The MP4 encode preserves color (yuv420p/bt709), is near-lossless, web-ready."""
    args = _mp4_args("ffmpeg", "/f/%05d.png", 30, Path("/out/render.mp4"))
    assert args[:5] == ["ffmpeg", "-y", "-framerate", "30", "-i"]
    assert "libx264" in args
    assert "yuv420p" in args  # broad-compatibility pixel format
    assert "bt709" in args  # correct color primaries/transfer/space
    assert "+faststart" in args  # immediate streaming/seeking
    # Near-lossless quality so exported colors match the source.
    assert args[args.index("-crf") + 1] == "16"


def test_webm_args_use_vp9_lossless_ish() -> None:
    args = _webm_args("ffmpeg", "/f/%05d.png", 24, Path("/out/render.webm"))
    assert "libvpx-vp9" in args
    assert args[args.index("-crf") + 1] == "24"


def test_gif_args_use_per_clip_palette() -> None:
    args = _gif_args("ffmpeg", "/f/%05d.png", 30, 1280, Path("/out/render.gif"))
    vf = args[args.index("-vf") + 1]
    assert "palettegen" in vf and "paletteuse" in vf
    assert "fps=25" in vf  # clamped to the GIF fps ceiling


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
