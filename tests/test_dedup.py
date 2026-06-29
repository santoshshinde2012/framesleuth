"""Tests for perceptual-hash keyframe deduplication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from framesleuth.pipeline.dedup import dedupe_keyframes, dhash, hamming
from framesleuth.schemas import KeyframeRef

cv2 = pytest.importorskip("cv2")


def _write_frame(path: Path, value: int, *, pattern: bool = False) -> None:
    """Write a 64x64 grayscale PNG.

    ``pattern=True`` paints high-contrast alternating columns, which produces a very
    different difference-hash from a flat-color frame (whose dHash is all zeros).
    """
    img = np.full((64, 64), value, dtype=np.uint8)
    if pattern:
        img[:, ::2] = 0
        img[:, 1::2] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _kf(i: int) -> KeyframeRef:
    return KeyframeRef(index=i, t=float(i), shows="scene", file=f"frames/{i}.png")


def test_dhash_and_hamming_basics(tmp_path: Path) -> None:
    _write_frame(tmp_path / "a.png", 100)
    _write_frame(tmp_path / "b.png", 100)
    _write_frame(tmp_path / "c.png", 100, pattern=True)
    ha, hb, hc = (dhash(str(tmp_path / n)) for n in ("a.png", "b.png", "c.png"))
    assert ha is not None and hb is not None and hc is not None
    assert hamming(ha, hb) == 0  # identical frames
    assert hamming(ha, hc) > 4  # a real structural change


def test_dhash_returns_none_for_unreadable(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert dhash(str(tmp_path / "bad.png")) is None


def test_dedupe_collapses_near_duplicates(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    # 0 and 1 identical; 2 distinct.
    _write_frame(frames / "0.png", 100)
    _write_frame(frames / "1.png", 100)
    _write_frame(frames / "2.png", 100, pattern=True)
    kept, dropped = dedupe_keyframes([_kf(0), _kf(1), _kf(2)], tmp_path, max_hamming=4)
    kept_files = {k.file for k in kept}
    assert dropped == 1
    assert "frames/2.png" in kept_files
    assert len(kept) == 2


def test_dedupe_keeps_distinct_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_frame(frames / "0.png", 30)
    _write_frame(frames / "1.png", 200, pattern=True)
    kept, dropped = dedupe_keyframes([_kf(0), _kf(1)], tmp_path, max_hamming=4)
    assert dropped == 0
    assert len(kept) == 2


def test_dedupe_single_frame_is_noop(tmp_path: Path) -> None:
    kept, dropped = dedupe_keyframes([_kf(0)], tmp_path)
    assert dropped == 0 and len(kept) == 1
