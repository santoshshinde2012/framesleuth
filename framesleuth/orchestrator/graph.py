"""End-to-end orchestration pipeline (stage-driven flow).

Designed to *fail loud, degrade gracefully*: when the vision model or ffmpeg is
unavailable, the orchestrator still produces a Context Bundle from the
browser sidecars (console errors, failed requests, clicks) rather than aborting.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from framesleuth.actions import resolve_action, suggest_actions
from framesleuth.clients.coder import CoderClient
from framesleuth.clients.vlm import VLMClient
from framesleuth.config import Settings
from framesleuth.errors import FramesleutheException, JobCancelledError
from framesleuth.jobs.store import JobStore
from framesleuth.logging_config import get_logger, set_job_id
from framesleuth.pipeline.asr import ASRPipeline
from framesleuth.pipeline.bug_extract import extract_bug_context_bundle
from framesleuth.pipeline.build_context import build_build_context
from framesleuth.pipeline.classify import (
    classify_video,
    is_ambiguous,
    looks_like_build_intent,
    refine_classification_with_model,
)
from framesleuth.pipeline.confidence import assess_actionability, compute_field_confidence
from framesleuth.pipeline.dedup import dedupe_keyframes
from framesleuth.pipeline.fusion import build_timeline
from framesleuth.pipeline.grounding import locate_in_code
from framesleuth.pipeline.overlay import overlay_interactions
from framesleuth.pipeline.preprocess import (
    ExtractedFrame,
    extract_audio,
    extract_frames,
    preprocess_video,
)
from framesleuth.pipeline.redact import redact_text
from framesleuth.pipeline.scenes import select_keyframes
from framesleuth.pipeline.sidecars import (
    ParsedSidecars,
    derive_error_evidence,
    derive_repro_steps,
    environment_from,
    parse_sidecars,
)
from framesleuth.pipeline.summarize import build_summary_input, summarize_recording
from framesleuth.pipeline.understand import analyze_keyframes
from framesleuth.schemas import (
    ErrorEvidenceItem,
    JobState,
    KeyframeRef,
    Redaction,
    SceneRecord,
    Transcript,
)
from framesleuth.skills import resolve_skill

logger = get_logger("orchestrator.graph")

# Identifiers worth searching for in the workspace (function/symbol names,
# file references), extracted from error text and stack frames.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_STOPWORDS = {
    "error",
    "errors",
    "exception",
    "cannot",
    "property",
    "undefined",
    "null",
    "failed",
    "internal",
    "server",
    "typeerror",
    "valueerror",
    "traceback",
}


def _grounding_queries(evidence: list[ErrorEvidenceItem]) -> list[str]:
    """Extract candidate symbols and full lines to search the workspace for."""
    queries: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        for line in item.text.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                queries.append(stripped)
            for token in _IDENTIFIER_RE.findall(line):
                if token.lower() in _STOPWORDS or token in seen:
                    continue
                seen.add(token)
                queries.append(token)
    return queries


def _intent_queries(user_intent: str | None, scenes: list[SceneRecord]) -> list[str]:
    """Grounding queries for build/feature work, where there is no error text.

    Derives nouns from the user's request and the on-screen UI labels / screen
    names so a feature like "add a dark-mode toggle" still grounds to the files
    that mention "toggle"/"theme"/etc. — instead of returning nothing.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        token = token.strip()
        if len(token) >= 4 and token.lower() not in _STOPWORDS and token.lower() not in seen:
            seen.add(token.lower())
            queries.append(token)

    for token in _IDENTIFIER_RE.findall(user_intent or ""):
        _add(token)
    for scene in scenes:
        if scene.screen_name:
            _add(scene.screen_name)
        for el in scene.ui_elements:
            for token in _IDENTIFIER_RE.findall(el.label):
                _add(token)
    return queries


class AnalysisOrchestrator:
    """Runs the analysis stages and persists lifecycle state."""

    def __init__(self, settings: Settings, store: JobStore, vlm_client: VLMClient) -> None:
        """Initialize the orchestrator with its settings, store, and VLM client."""
        self.settings = settings
        self.store = store
        self.vlm_client = vlm_client

    async def run(
        self,
        job_id: str,
        video_path: Path,
        source_video: str,
        *,
        sidecars: Any = None,
        workspace_root: Path | None = None,
        user_intent: str | None = None,
        skill: str | None = None,
        system_prompt: str | None = None,
        action: str | None = None,
        action_prompt: str | None = None,
    ) -> Path:
        """Execute the pipeline from preprocess to bundle persistence.

        Args:
            job_id: Unique job identifier.
            video_path: Path to the uploaded video on disk.
            source_video: Original filename for provenance.
            sidecars: Raw browser sidecars (flat event list or structured dict).
            workspace_root: Optional repo root for code grounding (Surface A).
            user_intent: The user's natural-language request to act on, recorded
                on the bundle and surfaced in the generated fix prompt.
            skill: Built-in skill name selecting the summary output style
                (e.g. ``summary``, ``bug_report``, ``tutorial``).
            system_prompt: Custom system prompt that overrides ``skill`` entirely.
            action: Built-in action mode shaping the fix-prompt task
                (e.g. ``fix``, ``explain``, ``triage``, ``test``). Auto-picked
                from the classification when not provided.
            action_prompt: Custom action task text that overrides ``action``.

        Returns:
            Path to the persisted ``bundle.json``.
        """
        set_job_id(job_id)
        metrics: dict[str, Any] = {"stages": {}, "degraded": []}
        # Job-scoped scratch space for frames/audio, removed regardless of outcome
        # so working files never accumulate — and never land next to the caller's
        # input video (the MCP path passes a real workspace file).
        work_dir = self.settings.BUNDLE_DIR / job_id / ".work"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            return await self._run(
                job_id,
                video_path,
                source_video,
                sidecars=sidecars,
                workspace_root=workspace_root,
                user_intent=user_intent,
                skill=skill,
                system_prompt=system_prompt,
                action=action,
                action_prompt=action_prompt,
                work_dir=work_dir,
                metrics=metrics,
            )
        except JobCancelledError:
            logger.info("Analysis cancelled for job %s", job_id)
            await self.store.update_job(job_id, state=JobState.CANCELLED)
            raise
        except FramesleutheException:
            raise
        except Exception as exc:  # mark the job failed before surfacing the error
            logger.exception("Analysis failed for job %s", job_id)
            await self.store.update_job(
                job_id,
                state=JobState.FAILED,
                error_json={"error": str(exc), "type": type(exc).__name__},
            )
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _run(
        self,
        job_id: str,
        video_path: Path,
        source_video: str,
        *,
        sidecars: Any,
        workspace_root: Path | None,
        user_intent: str | None,
        skill: str | None,
        system_prompt: str | None,
        action: str | None,
        action_prompt: str | None,
        work_dir: Path,
        metrics: dict[str, Any],
    ) -> Path:
        parsed = parse_sidecars(sidecars)

        await self._check_cancelled(job_id)
        await self.store.update_job(job_id, state=JobState.PREPROCESSING, progress_pct=10)
        duration_s = self._preprocess(video_path, metrics)
        transcript = self._transcribe(video_path, work_dir, metrics)

        # Decide build-aware visual extraction up front (from intent + narration),
        # because the understanding stage runs before classification.
        transcript_text = " ".join(seg.text for seg in transcript.segments)
        build_aware = looks_like_build_intent(user_intent, transcript_text)

        await self._check_cancelled(job_id)
        await self.store.update_job(job_id, state=JobState.UNDERSTANDING, progress_pct=40)
        scenes, analyzed_frames = await self._understand(
            video_path, work_dir, duration_s, metrics, parsed=parsed, build_aware=build_aware
        )

        await self._check_cancelled(job_id)
        await self.store.update_job(job_id, state=JobState.CLASSIFYING, progress_pct=65)
        sidecar_evidence, evidence_redactions = self._sidecar_evidence(parsed)
        sidecar_steps = derive_repro_steps(parsed)
        environment = environment_from(parsed)
        error_signals = [item.text for item in sidecar_evidence]

        classification = classify_video(
            transcript,
            scenes,
            settings=self.settings,
            error_signals=error_signals,
            user_intent=user_intent,
        )
        # On an ambiguous classification with visual evidence, zoom into the
        # failure window (bounded resample), then break any residual tie with a
        # model classification. Both no-op on confident/degraded runs.
        classification = await self._maybe_resample(
            video_path,
            work_dir,
            duration_s,
            scenes,
            analyzed_frames,
            transcript,
            error_signals,
            classification,
            metrics,
            user_intent=user_intent,
        )
        classification = await self._refine_classification(
            classification, scenes, transcript, error_signals, metrics, user_intent=user_intent
        )

        # Redact OCR text from all (incl. resampled) scenes before persistence.
        redactions: list[Redaction] = list(evidence_redactions)
        for scene in scenes:
            redacted_text, applied = redact_text(
                scene.ocr_text, timestamp=scene.t0, redact_pii=self.settings.REDACT_PII
            )
            scene.ocr_text = redacted_text
            redactions.extend(applied)

        # Synthesize the summary/analysis from the (redacted) scenes + transcript,
        # shaped by the caller's skill/system prompt, BEFORE extraction so the
        # bundle can lead with it (the summary is the deliverable for general
        # videos, and seeds the title when there is no error/caption). A model
        # failure degrades to an empty string and never affects analysis_quality.
        skill_label, skill_prompt = resolve_skill(skill, system_prompt)
        summary = await self._summarize(scenes, transcript, skill_prompt, user_intent, metrics)

        await self._check_cancelled(job_id)
        await self.store.update_job(job_id, state=JobState.EXTRACTING, progress_pct=80)
        keyframes = self._keyframes(scenes)
        bundle = extract_bug_context_bundle(
            job_id=job_id,
            source_video=source_video,
            duration_s=duration_s,
            classification=classification,
            transcript=transcript,
            scenes=scenes,
            keyframes=keyframes,
            environment=environment,
            sidecar_steps=sidecar_steps,
            sidecar_evidence=sidecar_evidence,
            degraded_stages=metrics["degraded"],
            summary=summary,
        )
        bundle.redactions = redactions
        bundle.transcript_path = "transcript.json"
        bundle.timeline_path = "timeline.json"
        bundle.user_intent = user_intent
        bundle.skill = skill_label

        # Resolve the action mode that shapes the fix-prompt. With no explicit
        # action it is auto-picked from the classification (bug -> fix, etc.).
        action_label, custom_action, _ = resolve_action(
            action, action_prompt, classification.label.value
        )
        bundle.action = action_label
        bundle.action_prompt = custom_action

        await self.store.update_job(job_id, state=JobState.GROUNDING, progress_pct=90)
        bundle.code_candidates = self._ground(
            bundle.error_evidence, workspace_root, metrics, user_intent=user_intent, scenes=scenes
        )

        # Assemble the build/feature context (screens, components, flow, design) and
        # where to implement it — null for pure bug reports. Then derive per-field
        # confidence and task-aware actionability so consumers know what to trust.
        bundle.build_context = build_build_context(scenes, classification, bundle.code_candidates)
        bundle.field_confidence = compute_field_confidence(bundle)
        bundle.analysis_quality.actionability = assess_actionability(bundle)

        # Derive the next-step menu last, so it reflects grounded code candidates,
        # the build context, and the final analysis quality.
        bundle.suggested_actions = suggest_actions(bundle.model_dump(mode="json"))

        # Surface per-stage timings on the bundle so consumers (and the report UI)
        # can see where the analysis time went, not just that it finished.
        bundle.stage_timings = {k: float(v) for k, v in metrics["stages"].items()}

        bundle_dir = self.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        self._persist_keyframes(analyzed_frames, work_dir, bundle_dir)
        self._write_artifacts(bundle_dir, parsed, transcript, scenes, metrics)
        self._store_source_video(video_path, bundle_dir)

        await self.store.update_job(
            job_id,
            state=JobState.DONE,
            progress_pct=100,
            bundle_path=str(bundle_path),
        )
        logger.info("Analysis complete: degraded_stages=%s", metrics["degraded"])
        return bundle_path

    async def _check_cancelled(self, job_id: str) -> None:
        """Abort the run cooperatively if the caller requested cancellation.

        Checked at each stage boundary so a cancel takes effect promptly without
        killing a thread mid-decode. Raises :class:`JobCancelledError`, which the
        ``run`` wrapper turns into a terminal ``CANCELLED`` state.
        """
        if await self.store.is_cancel_requested(job_id):
            raise JobCancelledError(job_id)

    def _preprocess(self, video_path: Path, metrics: dict[str, Any]) -> float:
        """Probe the video; degrade to a zero-duration bundle on failure."""
        start = time.perf_counter()
        try:
            pre = preprocess_video(video_path, settings=self.settings)
            metrics["stages"]["preprocess"] = round(time.perf_counter() - start, 3)
            if pre.duration_s <= 0:
                # Duration could not be determined (even after the packet-timestamp
                # scan), so only a single frame at t=0 can be sampled. Flag the run
                # as degraded rather than reporting a confident, near-empty bundle.
                logger.warning("Preprocess degraded: duration unknown; sampling single-frame")
                metrics["degraded"].append("preprocess")
            return pre.duration_s
        except FramesleutheException as exc:
            logger.warning("Preprocess degraded (%s); continuing sidecar-only", exc.code.value)
            metrics["degraded"].append("preprocess")
            return 0.0

    def _transcribe(self, video_path: Path, work_dir: Path, metrics: dict[str, Any]) -> Transcript:
        """Extract audio and transcribe it; degrade to an empty transcript."""
        start = time.perf_counter()
        try:
            audio_path = extract_audio(video_path, work_dir)
            transcript = ASRPipeline(
                min_confidence=self.settings.ASR_MIN_CONFIDENCE,
                vad_filter=self.settings.ASR_VAD_FILTER,
                language=self.settings.ASR_LANGUAGE or None,
            ).transcribe(audio_path, has_audio=audio_path is not None)
            if transcript.segments:
                metrics["stages"]["asr"] = round(time.perf_counter() - start, 3)
            else:
                metrics["degraded"].append("asr")
            return transcript
        except Exception as exc:  # ASR is advisory: never let it abort the run
            logger.warning("ASR degraded: %s", exc)
            metrics["degraded"].append("asr")
            return Transcript(segments=[], words=[])

    async def _summarize(
        self,
        scenes: list[SceneRecord],
        transcript: Transcript,
        system_prompt: str,
        user_intent: str | None,
        metrics: dict[str, Any],
    ) -> str:
        """Generate the recording summary; degrade to an empty string on failure.

        Uses the configured summary model, falling back to the VLM endpoint when
        ``SUMMARY_URL``/``SUMMARY_MODEL`` are blank (a vision-language model
        summarizes the timeline text fine, so no extra model server is needed).
        """
        start = time.perf_counter()
        url = self.settings.SUMMARY_URL or self.settings.VLM_URL
        model = self.settings.SUMMARY_MODEL or self.settings.VLM_MODEL
        async with CoderClient(
            url, model, timeout_s=self.settings.SUMMARY_TIMEOUT_S, max_retries=2
        ) as client:
            summary = await summarize_recording(
                scenes,
                transcript,
                system_prompt=system_prompt,
                user_intent=user_intent,
                client=client,
                max_tokens=self.settings.SUMMARY_MAX_TOKENS,
            )
        if summary:
            metrics["stages"]["summarize"] = round(time.perf_counter() - start, 3)
        else:
            # Empty summary = nothing to summarize or the model was unavailable.
            metrics["degraded"].append("summarize")
        return summary

    async def _understand(
        self,
        video_path: Path,
        work_dir: Path,
        duration_s: float,
        metrics: dict[str, Any],
        *,
        parsed: ParsedSidecars | None = None,
        build_aware: bool = False,
    ) -> tuple[list[SceneRecord], list[KeyframeRef]]:
        """Run visual understanding when frames and the VLM are available.

        Returns the analyzed scenes alongside the keyframes they came from, so the
        source images can be persisted next to the bundle. ``build_aware`` raises the
        keyframe budget (to capture more distinct screens) and switches the per-frame
        prompt to the structured build prompt. ``parsed`` supplies click/cursor
        sidecars used to overlay interaction markers onto the analyzed frames.
        """
        start = time.perf_counter()
        sampled = self._sample_times(duration_s, build_aware=build_aware)
        keyframes = self._select_keyframes(video_path, sampled, work_dir, build_aware=build_aware)
        # Only call the VLM when the keyframe images actually exist on disk.
        available = [kf for kf in keyframes if (work_dir / kf.file).exists()]
        if not available:
            logger.info("Visual analysis skipped: no extracted frames available")
            metrics["degraded"].append("understand")
            return [], []
        # Collapse near-identical frames (held spinners, static cards, repeated
        # screens) so the VLM budget is spent on distinct content.
        if self.settings.KEYFRAME_DEDUP and len(available) > 1:
            available, dropped = dedupe_keyframes(
                available, work_dir, max_hamming=self.settings.KEYFRAME_PHASH_HAMMING_MAX
            )
            if dropped:
                metrics["keyframes_deduped"] = dropped
                logger.info("Keyframe dedup dropped %d near-duplicate frame(s)", dropped)
        # Draw click/cursor markers onto the frames the VLM is about to read, so a
        # user interaction becomes visible evidence (no-op without coordinates).
        if self.settings.OVERLAY_INTERACTIONS and parsed is not None:
            marked = overlay_interactions(available, work_dir, parsed)
            if marked:
                metrics["interactions_overlaid"] = marked
        try:
            scenes = await analyze_keyframes(
                available,
                work_dir,
                self.vlm_client,
                max_concurrency=self.settings.VLM_MAX_CONCURRENCY,
                error_max_tokens=self.settings.VLM_ERROR_MAX_TOKENS,
                rescue_frame=lambda t: self._rescue_frame(video_path, work_dir, t),
                build_aware=build_aware,
                ocr_backstop=self.settings.OCR_BACKSTOP,
            )
            metrics["stages"]["understand"] = round(time.perf_counter() - start, 3)
            return scenes, available
        except Exception as exc:
            # VLM/network/file failures must degrade to sidecar-only, not abort.
            logger.warning("Visual analysis degraded: %s", exc)
            metrics["degraded"].append("understand")
            return [], []

    def _rescue_frame(self, video_path: Path, work_dir: Path, t: float) -> str | None:
        """Re-decode one frame at full resolution for the error re-OCR.

        Wires ``FRAME_HIGHRES_HEIGHT``: a stack trace rendered a few pixels tall in
        a 480p downscale is unreadable, so the focused error pass reads the frame at
        high resolution instead. Each timestamp gets its own output dir so
        concurrent rescues never collide. Returns ``None`` if it can't be produced.
        """
        out_dir = work_dir / "frames_hires" / str(round(t * 1000))
        extracted = extract_frames(
            video_path, [t], out_dir, height=self.settings.FRAME_HIGHRES_HEIGHT
        )
        if not extracted:
            return None
        return str(out_dir / "0.png")

    async def _maybe_resample(
        self,
        video_path: Path,
        work_dir: Path,
        duration_s: float,
        scenes: list[SceneRecord],
        analyzed: list[KeyframeRef],
        transcript: Transcript,
        error_signals: list[str],
        classification: Any,
        metrics: dict[str, Any],
        user_intent: str | None = None,
    ) -> Any:
        """Resample extra frames around the failure window when classification is uncertain.

        A bounded agentic step (``MAX_RESAMPLE_RETRIES``): only fires when the
        deterministic label is ambiguous *and* there is a visual error anchor to
        zoom into, extends ``scenes``/``analyzed`` in lockstep, and re-classifies.
        Inert on confident/degraded runs and when no new frames can be decoded.
        """
        settings = self.settings
        if settings.MAX_RESAMPLE_RETRIES <= 0 or duration_s <= 0 or not scenes:
            return classification

        start = time.perf_counter()
        attempts = 0
        while attempts < settings.MAX_RESAMPLE_RETRIES and is_ambiguous(classification, settings):
            windows = [scene.t0 for scene in scenes if scene.is_error_state]
            extra_times = self._resample_times(
                windows, duration_s, existing=[kf.t for kf in analyzed]
            )
            if not extra_times:
                break
            new_scenes, new_frames = await self._analyze_extra(
                video_path, work_dir, extra_times, len(analyzed)
            )
            if not new_scenes:
                break
            scenes.extend(new_scenes)
            analyzed.extend(new_frames)
            classification = classify_video(
                transcript,
                scenes,
                settings=settings,
                error_signals=error_signals,
                user_intent=user_intent,
            )
            attempts += 1

        if attempts:
            metrics["stages"]["resample"] = round(time.perf_counter() - start, 3)
            metrics["resample_attempts"] = attempts
        return classification

    def _resample_times(
        self,
        windows: list[float],
        duration_s: float,
        existing: list[float],
        *,
        span: float = 1.0,
        cap: int = 4,
    ) -> list[float]:
        """Pick new timestamps just before/after each error window, deduped.

        Skips timestamps within 0.1s of one already analyzed so a second pass
        cannot re-decode the same frame (which would loop without new evidence).
        """
        seen = {round(t, 1) for t in existing}
        out: list[float] = []
        for window in windows:
            for offset in (-span / 2, span / 2):
                t = round(min(max(window + offset, 0.0), duration_s), 3)
                key = round(t, 1)
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
                if len(out) >= cap:
                    return out
        return out

    async def _analyze_extra(
        self, video_path: Path, work_dir: Path, times: list[float], start_index: int
    ) -> tuple[list[SceneRecord], list[KeyframeRef]]:
        """Decode and analyze extra frames for resampling; degrade to empty lists."""
        out_dir = work_dir / f"frames_resample_{start_index}"
        extracted = extract_frames(
            video_path, times, out_dir, height=self.settings.FRAME_LOWRES_HEIGHT
        )
        keyframes = [
            KeyframeRef(index=start_index + offset, t=frame.t, shows="scene", file=frame.file)
            for offset, frame in enumerate(extracted)
        ]
        available = [kf for kf in keyframes if (work_dir / kf.file).exists()]
        if not available:
            return [], []
        try:
            scenes = await analyze_keyframes(
                available,
                work_dir,
                self.vlm_client,
                max_concurrency=self.settings.VLM_MAX_CONCURRENCY,
                error_max_tokens=self.settings.VLM_ERROR_MAX_TOKENS,
                rescue_frame=lambda t: self._rescue_frame(video_path, work_dir, t),
                ocr_backstop=self.settings.OCR_BACKSTOP,
            )
        except Exception as exc:
            logger.warning("Resample understanding degraded: %s", exc)
            return [], []
        return scenes, available

    async def _refine_classification(
        self,
        classification: Any,
        scenes: list[SceneRecord],
        transcript: Transcript,
        error_signals: list[str],
        metrics: dict[str, Any],
        user_intent: str | None = None,
    ) -> Any:
        """Break a residual ambiguous-band tie with a model classification."""
        settings = self.settings
        if not settings.CLASSIFY_USE_MODEL or not is_ambiguous(classification, settings):
            return classification

        summary_text = build_summary_input(scenes, transcript, None)
        start = time.perf_counter()
        url = settings.SUMMARY_URL or settings.VLM_URL
        model = settings.SUMMARY_MODEL or settings.VLM_MODEL
        async with CoderClient(
            url, model, timeout_s=settings.SUMMARY_TIMEOUT_S, max_retries=2
        ) as client:
            refined = await refine_classification_with_model(
                classification,
                summary_text=summary_text,
                scenes=scenes,
                error_signals=error_signals,
                client=client,
                settings=settings,
                user_intent=user_intent,
            )
        if refined is not classification:
            metrics["stages"]["classify_refine"] = round(time.perf_counter() - start, 3)
        return refined

    def _select_keyframes(
        self, video_path: Path, sampled: list[float], frames_dir: Path, *, build_aware: bool = False
    ) -> list[KeyframeRef]:
        """Extract frames to disk and pick keyframes from real visual deltas.

        Falls back to any pre-extracted ``frames/{i}.png`` already on disk (the
        path used by deterministic tests) when live extraction yields nothing.
        """
        extracted = extract_frames(
            video_path,
            sampled,
            frames_dir / "frames",
            height=self.settings.FRAME_LOWRES_HEIGHT,
        )
        if extracted:
            return self._coverage_keyframes(extracted, max_keyframes=12 if build_aware else 8)
        times: list[float] = []
        files: list[str] = []
        for index, t in enumerate(sampled):
            rel = f"frames/{index}.png"
            if (frames_dir / rel).exists():
                times.append(t)
                files.append(rel)
        if not files:
            return []
        return select_keyframes(
            frame_times=times,
            frame_files=files,
            error_hints=[False] * len(files),
            cut_threshold=self.settings.SCENE_CUT_THRESHOLD,
        )

    def _coverage_keyframes(
        self, extracted: list[ExtractedFrame], *, max_keyframes: int = 8
    ) -> list[KeyframeRef]:
        """Choose keyframes with adaptive, coverage-enforcing selection (AKS-lite).

        Uniform sampling wastes the VLM budget on near-duplicate frames and misses
        the salient ones. Following adaptive keyframe sampling (AKS, CVPR 2025), we
        split the timeline into ``max_keyframes`` temporal bins (guaranteeing
        COVERAGE) and pick the most visually salient frame — highest
        ``change_score``, our pre-VLM RELEVANCE proxy — within each bin. Endpoints
        are always kept so the full span is represented.
        """
        n = len(extracted)
        if n <= max_keyframes:
            chosen = set(range(n))
        else:
            chosen = {0, n - 1}
            for b in range(max_keyframes):
                lo = round(b * n / max_keyframes)
                hi = round((b + 1) * n / max_keyframes)
                if hi <= lo:
                    continue
                best = max(range(lo, hi), key=lambda i: extracted[i].change_score)
                chosen.add(best)
                if len(chosen) >= max_keyframes:
                    break
        return [
            KeyframeRef(index=idx, t=extracted[idx].t, shows="scene", file=extracted[idx].file)
            for idx in sorted(chosen)[:max_keyframes]
        ]

    def _persist_keyframes(
        self, analyzed: list[KeyframeRef], frames_dir: Path, bundle_dir: Path
    ) -> None:
        """Copy each analyzed source frame to ``keyframes/{idx:03d}.png``.

        Keeps the bundle's ``keyframe_refs`` resolvable by the MCP server and the
        report UI even after the working ``frames/`` directory is cleaned up.
        """
        if not analyzed:
            return
        out_dir = bundle_dir / "keyframes"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, keyframe in enumerate(analyzed):
            source = frames_dir / keyframe.file
            if not source.exists():
                continue
            try:
                shutil.copy2(source, out_dir / f"{idx:03d}.png")
            except OSError as exc:
                logger.warning("Could not persist keyframe %d: %s", idx, exc)

    def _sample_times(self, duration_s: float, *, build_aware: bool = False) -> list[float]:
        """Pick frame timestamps to decode.

        Samples ~2 frames/second (capped) at the *midpoint* of each segment.
        Mid-segment sampling avoids seeking to the keyframe just before a
        transition, so brief on-screen error flashes near a boundary are caught;
        the density bounds the eventual VLM calls (scene selection caps further).
        ``build_aware`` raises the candidate cap so the adaptive keyframe selector
        has more distinct screens to choose from on feature/design walkthroughs.
        """
        if duration_s <= 0:
            return [0.0]
        cap = 28 if build_aware else 16
        count = max(2, min(cap, round(duration_s * 2)))
        step = duration_s / count
        return [round((i + 0.5) * step, 3) for i in range(count)]

    def _sidecar_evidence(
        self, parsed: ParsedSidecars
    ) -> tuple[list[ErrorEvidenceItem], list[Redaction]]:
        """Derive error evidence from sidecars, redacting secrets from each item."""
        evidence = derive_error_evidence(parsed)
        redactions: list[Redaction] = []
        for item in evidence:
            item.text, applied = redact_text(
                item.text, timestamp=item.t, redact_pii=self.settings.REDACT_PII
            )
            redactions.extend(applied)
        return evidence, redactions

    def _keyframes(self, scenes: list[SceneRecord]) -> list[KeyframeRef]:
        """Build keyframe references for any analyzed scenes."""
        return [
            KeyframeRef(
                index=idx,
                t=scene.t0,
                shows=("failure" if scene.is_error_state else "scene"),
                file=f"keyframes/{idx:03d}.png",
            )
            for idx, scene in enumerate(scenes)
        ]

    def _ground(
        self,
        evidence: list[ErrorEvidenceItem],
        workspace_root: Path | None,
        metrics: dict[str, Any],
        *,
        user_intent: str | None = None,
        scenes: list[SceneRecord] | None = None,
    ) -> list[Any]:
        """Locate candidate code locations when a repo is open.

        For bugs, ground the error symbols to their source. For build/feature work
        there is no error text, so fall back to intent + on-screen UI nouns — so a
        feature still grounds to the files to extend instead of returning nothing.
        """
        if workspace_root is None or not workspace_root.exists():
            return []
        start = time.perf_counter()
        queries = _grounding_queries(evidence)
        if not queries:
            queries = _intent_queries(user_intent, scenes or [])
        candidates = locate_in_code(
            workspace_root, queries, max_files=self.settings.GROUNDING_MAX_FILES
        )
        metrics["stages"]["grounding"] = round(time.perf_counter() - start, 3)
        return candidates

    def _store_source_video(self, video_path: Path, bundle_dir: Path) -> None:
        """Copy the recorded video next to the bundle so it can be replayed."""
        if not video_path.exists():
            return
        suffix = video_path.suffix.lower() or ".webm"
        try:
            shutil.copy2(video_path, bundle_dir / f"source{suffix}")
        except OSError as exc:
            logger.warning("Could not store source video: %s", exc)

    def _write_artifacts(
        self,
        bundle_dir: Path,
        parsed: ParsedSidecars,
        transcript: Transcript,
        scenes: list[SceneRecord],
        metrics: dict[str, Any],
    ) -> None:
        """Persist transcript, timeline, sidecars, and metrics alongside the bundle."""
        (bundle_dir / "transcript.json").write_text(
            transcript.model_dump_json(indent=2), encoding="utf-8"
        )
        # Fuse visual + transcript evidence into one ordered, cited timeline, then
        # attach the raw sidecar streams for full-fidelity replay.
        fused = [
            {"t": event.t, "kind": event.kind, "text": event.text, "citation": event.citation}
            for event in build_timeline(scenes, transcript)
        ]
        timeline = {
            "events": fused,
            "console_errors": parsed.console_errors,
            "network": parsed.network,
            "clicks": parsed.clicks,
        }
        (bundle_dir / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        (bundle_dir / "sidecars.json").write_text(
            json.dumps(
                {
                    "console_errors": parsed.console_errors,
                    "network": parsed.network,
                    "clicks": parsed.clicks,
                    "cursor": parsed.cursor,
                    "env": parsed.env,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (bundle_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
