"""Visual understanding over selected keyframes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from framesleuth.clients.vlm import FrameAnalysisResponse
from framesleuth.schemas import KeyframeRef, SceneRecord

# Words that betray an error frame even when the model forgot to set
# ``is_error_state`` — used to still trigger the focused OCR retry.
_ERROR_HINTS = ("error", "exception", "failed", "failure", "traceback", "stack trace", "500")

# Given a timestamp, return a path to that frame re-decoded at full resolution
# (or ``None`` if it can't be produced). Lets the error retry read tiny text that
# a 480p downscale would smear — without coupling this stage to video decoding.
RescueFrame = Callable[[float], str | None]


class SupportsAnalyzeFrame(Protocol):
    """Protocol abstraction for VLM client dependency inversion."""

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        """Analyze a frame and return structured response."""


def _likely_error(response: FrameAnalysisResponse) -> bool:
    """Whether a frame looks like an error even if ``is_error_state`` is unset."""
    if response.is_error_state:
        return True
    blob = f"{response.caption} {response.reason or ''}".lower()
    return any(hint in blob for hint in _ERROR_HINTS)


def _augment_ocr(vlm_text: str, image_path: str) -> str:
    """Merge the VLM's OCR with a dedicated OCR read, preferring the richer text.

    The two readers fail differently; keeping the longer reading recovers small
    error text the VLM smeared while never losing what the VLM already captured.
    A no-op when the optional OCR backstop is unavailable.
    """
    from framesleuth.pipeline.ocr import ocr_image

    backstop = ocr_image(image_path)
    if not backstop:
        return vlm_text
    if len(backstop) > len(vlm_text.strip()):
        return backstop
    return vlm_text


async def _analyze_one(
    keyframe: KeyframeRef,
    frames_dir: Path,
    vlm_client: SupportsAnalyzeFrame,
    *,
    min_ocr_len: int,
    error_max_tokens: int | None,
    rescue_frame: RescueFrame | None,
    build_aware: bool,
    ocr_backstop: bool,
) -> SceneRecord:
    """Analyze a single keyframe, retrying once for sparse error-frame OCR.

    When ``build_aware`` is set (feature/demo/build videos), the first pass uses the
    build prompt that additionally extracts structured UI elements, layout, screen
    name, and design notes — the inputs a coding agent needs to *build* what was
    shown. The error re-OCR retry still uses the focused error template. When
    ``ocr_backstop`` is set, a dedicated OCR engine reads the high-res error frame
    as an independent check and its text is used when it beats the VLM's OCR.
    """
    from framesleuth.prompts import VLMPrompts

    image_path = str(frames_dir / keyframe.file)
    first_prompt = VLMPrompts.frame_analysis_build(keyframe.t) if build_aware else None
    response = await vlm_client.analyze_frame(image_path, keyframe.t, prompt_override=first_prompt)

    # Re-prompt with the tuned error template when the frame looks like a failure
    # but the OCR came back too sparse to be useful — a weak model often misses
    # the small error text on the first, general-purpose pass. The retry reads the
    # frame at full resolution and *uncompressed* (never compress the evidence),
    # with a larger token budget so long stack traces are not truncated.
    if _likely_error(response) and len(response.ocr_text.strip()) < min_ocr_len:
        hires = rescue_frame(keyframe.t) if rescue_frame else None
        response = await vlm_client.analyze_frame(
            hires or image_path,
            keyframe.t,
            prompt_override=VLMPrompts.error_frame_analysis(keyframe.t),
            max_tokens=error_max_tokens,
            send_jpeg=False,
        )
        if ocr_backstop:
            response.ocr_text = _augment_ocr(response.ocr_text, hires or image_path)

    return SceneRecord(
        t0=keyframe.t,
        t1=keyframe.t,
        caption=response.caption,
        ocr_text=response.ocr_text,
        ui_action=response.ui_action,
        is_error_state=response.is_error_state,
        reason=response.reason,
        ui_elements=response.ui_elements,
        layout=response.layout,
        screen_name=response.screen_name,
        design_notes=response.design_notes,
        data_shown=response.data_shown,
    )


async def analyze_keyframes(
    keyframes: list[KeyframeRef],
    frames_dir: Path,
    vlm_client: SupportsAnalyzeFrame,
    *,
    min_ocr_len: int = 4,
    max_concurrency: int = 3,
    error_max_tokens: int | None = None,
    rescue_frame: RescueFrame | None = None,
    build_aware: bool = False,
    ocr_backstop: bool = False,
) -> list[SceneRecord]:
    """Analyze keyframes concurrently and return scene records in keyframe order.

    Frames are independent, so they are analyzed with bounded concurrency rather
    than strictly one-at-a-time; this overlaps the per-frame VLM round-trips
    instead of summing them. ``max_concurrency`` bounds in-flight requests so a
    single local model server is not overwhelmed. ``rescue_frame`` (optional)
    supplies a full-resolution copy of a frame for the error re-OCR.
    """
    if not keyframes:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _guarded(keyframe: KeyframeRef) -> SceneRecord:
        async with semaphore:
            return await _analyze_one(
                keyframe,
                frames_dir,
                vlm_client,
                min_ocr_len=min_ocr_len,
                error_max_tokens=error_max_tokens,
                rescue_frame=rescue_frame,
                build_aware=build_aware,
                ocr_backstop=ocr_backstop,
            )

    # gather preserves input order, keeping scenes aligned with their keyframes.
    return list(await asyncio.gather(*(_guarded(kf) for kf in keyframes)))
