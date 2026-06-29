"""Scene and keyframe selection with deterministic fallback guarantees."""

from __future__ import annotations

from collections.abc import Iterable

from framesleuth.schemas import KeyframeRef


def detect_scene_cuts(change_scores: Iterable[float], threshold: float = 0.35) -> list[int]:
    """Return frame indices considered scene cuts from visual delta scores."""
    return [idx for idx, score in enumerate(change_scores) if score >= threshold]


def select_keyframes(
    frame_times: list[float],
    frame_files: list[str],
    *,
    change_scores: list[float] | None = None,
    error_hints: list[bool] | None = None,
    max_keyframes: int = 12,
    cut_threshold: float = 0.35,
) -> list[KeyframeRef]:
    """Select keyframes using cuts, error hints, and a mandatory fallback.

    Guarantees at least one keyframe for non-empty input. ``cut_threshold`` tunes
    how large a visual delta counts as a scene cut.
    """
    if not frame_times or not frame_files:
        return []

    if len(frame_times) != len(frame_files):
        raise ValueError("frame_times and frame_files length mismatch")

    scores = change_scores or [0.0] * len(frame_times)
    hints = error_hints or [False] * len(frame_times)

    candidate_indices = set(detect_scene_cuts(scores, threshold=cut_threshold))
    candidate_indices.update(idx for idx, is_hint in enumerate(hints) if is_hint)

    # Fallback for zero-cuts videos: choose midpoint frame.
    if not candidate_indices:
        candidate_indices.add(len(frame_times) // 2)

    selected = sorted(candidate_indices)[:max_keyframes]

    keyframes: list[KeyframeRef] = []
    for index in selected:
        is_error = hints[index] if index < len(hints) else False
        shows = "error_state_candidate" if is_error else "scene_transition"
        keyframes.append(
            KeyframeRef(
                index=index,
                t=frame_times[index],
                shows=shows,
                file=frame_files[index],
            )
        )

    return keyframes
