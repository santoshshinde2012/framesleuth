"""Perceptual-hash deduplication of keyframes.

A coverage-binned keyframe set still contains near-identical frames — a held
loading spinner, a static title card, the same screen revisited — and each one
costs a full VLM round-trip. This collapses runs of perceptually-identical frames
*before* the VLM sees them, so the budget is spent on distinct content.

It is deliberately conservative: two frames are merged only when their difference
hashes are within ``max_hamming`` bits (out of 64), so a subtle but real on-screen
change (an error appearing, a value updating) is never silently dropped. If OpenCV
is unavailable or a frame can't be decoded, dedup is a no-op — never a failure.
"""

from __future__ import annotations

from pathlib import Path

from framesleuth.logging_config import get_logger
from framesleuth.schemas import KeyframeRef

logger = get_logger("pipeline.dedup")


def dhash(image_path: str, *, hash_size: int = 8) -> int | None:
    """Compute a 64-bit difference hash for an image, or ``None`` if it can't be read.

    The difference hash (dHash) downsamples to ``(hash_size+1) x hash_size`` grayscale
    and encodes whether each pixel is brighter than its right neighbor. It is robust
    to scale/compression yet sensitive to genuine layout/content change.
    """
    try:
        import cv2  # lazy: heavy media dep, only in the full stack
        import numpy as np
    except Exception:  # pragma: no cover - exercised only without cv2
        return None
    try:
        raw = Path(image_path).read_bytes()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        resized = cv2.resize(arr, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = 0
        for value in diff.flatten():
            bits = (bits << 1) | int(value)
        return bits
    except Exception as exc:  # pragma: no cover - defensive, never abort analysis
        logger.debug("dHash skipped for %s: %s", image_path, exc)
        return None


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def dedupe_keyframes(
    keyframes: list[KeyframeRef], frames_dir: Path, *, max_hamming: int = 4
) -> tuple[list[KeyframeRef], int]:
    """Drop near-duplicate keyframes, returning ``(kept, dropped_count)``.

    Frames are compared in time order; a frame within ``max_hamming`` bits of an
    already-kept frame is dropped. Frames whose hash can't be computed are always
    kept (fail-open). At least one keyframe is always returned for non-empty input.
    """
    if len(keyframes) <= 1:
        return keyframes, 0

    kept: list[KeyframeRef] = []
    kept_hashes: list[int] = []
    dropped = 0
    for keyframe in keyframes:
        digest = dhash(str(frames_dir / keyframe.file))
        if digest is None:
            kept.append(keyframe)
            continue
        if any(hamming(digest, prior) <= max_hamming for prior in kept_hashes):
            dropped += 1
            continue
        kept.append(keyframe)
        kept_hashes.append(digest)

    # Guarantee at least one frame survives even in a pathological all-duplicate run.
    if not kept:
        return keyframes[:1], len(keyframes) - 1
    return kept, dropped
