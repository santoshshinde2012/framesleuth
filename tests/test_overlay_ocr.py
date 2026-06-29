"""Tests for interaction overlay and the OCR backstop (fallback paths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from framesleuth.pipeline.ocr import ocr_available, ocr_image
from framesleuth.pipeline.overlay import _normalized_xy, overlay_interactions
from framesleuth.pipeline.sidecars import ParsedSidecars
from framesleuth.schemas import KeyframeRef

cv2 = pytest.importorskip("cv2")


def _frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((80, 120, 3), 128, dtype=np.uint8))


def test_normalized_xy_from_fraction() -> None:
    assert _normalized_xy({"nx": 0.5, "ny": 0.25}, {}) == (0.5, 0.25)


def test_normalized_xy_from_absolute_with_viewport() -> None:
    coords = _normalized_xy({"x": 200, "y": 100}, {"innerWidth": 400, "innerHeight": 200})
    assert coords == (0.5, 0.5)


def test_normalized_xy_none_without_mapping() -> None:
    assert _normalized_xy({"x": 200, "y": 100}, {}) is None  # no viewport
    assert _normalized_xy({"selector": "button"}, {}) is None


def test_overlay_draws_marker_when_coords_present(tmp_path: Path) -> None:
    _frame(tmp_path / "frames" / "0.png")
    before = cv2.imread(str(tmp_path / "frames" / "0.png")).copy()
    parsed = ParsedSidecars(clicks=[{"t": 1.0, "nx": 0.5, "ny": 0.5}])
    kfs = [KeyframeRef(index=0, t=1.0, shows="scene", file="frames/0.png")]

    drawn = overlay_interactions(kfs, tmp_path, parsed, window_s=0.6)

    assert drawn == 1
    after = cv2.imread(str(tmp_path / "frames" / "0.png"))
    assert not np.array_equal(before, after)  # the frame was modified


def test_overlay_noop_without_coordinates(tmp_path: Path) -> None:
    _frame(tmp_path / "frames" / "0.png")
    parsed = ParsedSidecars(clicks=[{"t": 1.0, "selector": "button"}])  # no coords
    kfs = [KeyframeRef(index=0, t=1.0, shows="scene", file="frames/0.png")]
    assert overlay_interactions(kfs, tmp_path, parsed) == 0


def test_overlay_noop_without_events(tmp_path: Path) -> None:
    _frame(tmp_path / "frames" / "0.png")
    kfs = [KeyframeRef(index=0, t=1.0, shows="scene", file="frames/0.png")]
    assert overlay_interactions(kfs, tmp_path, ParsedSidecars()) == 0


def test_ocr_image_fallback_returns_none_without_extra(tmp_path: Path) -> None:
    """Without the optional pytesseract extra, OCR is a clean no-op (None)."""
    _frame(tmp_path / "f.png")
    if ocr_available():
        pytest.skip("pytesseract installed; fallback path not exercised")
    assert ocr_image(str(tmp_path / "f.png")) is None


def test_augment_ocr_keeps_vlm_text_without_backstop(tmp_path: Path) -> None:
    """The OCR augmentation keeps the VLM's text when the backstop is unavailable."""
    from framesleuth.pipeline.understand import _augment_ocr

    _frame(tmp_path / "f.png")
    if ocr_available():
        pytest.skip("pytesseract installed; fallback path not exercised")
    assert _augment_ocr("VLM OCR TEXT", str(tmp_path / "f.png")) == "VLM OCR TEXT"
