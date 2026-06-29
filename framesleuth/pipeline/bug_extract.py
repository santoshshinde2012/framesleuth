"""Context-bundle extraction (any video) with evidence and anti-fabrication guards."""

from __future__ import annotations

from datetime import UTC, datetime

from framesleuth.schemas import (
    AnalysisQuality,
    Classification,
    ClassificationLabel,
    ContextBundle,
    ErrorEvidenceItem,
    KeyframeRef,
    KeyMoment,
    Priority,
    Reproducibility,
    ReproStep,
    SceneRecord,
    Severity,
    Transcript,
)

_MAX_TITLE_LEN = 200
_MAX_KEY_MOMENTS = 12
_MAX_MOMENT_LABEL_LEN = 200

# Human-readable explanation for each pipeline stage that can degrade. Keeps the
# warning text consistent between the bundle and the downstream fix prompt.
_STAGE_WARNINGS = {
    "preprocess": "Video duration could not be determined; frame sampling was limited.",
    "understand": "Visual frame analysis was unavailable; no on-screen evidence was read.",
    "asr": "No usable audio transcript (silent video or speech model unavailable).",
}


def _derive_repro_steps(scenes: list[SceneRecord]) -> list[ReproStep]:
    """Create reproducible steps from scene actions while preserving citations."""
    steps: list[ReproStep] = []
    for idx, scene in enumerate(scenes, start=1):
        action = (scene.ui_action or scene.caption).strip()
        if not action:
            continue
        steps.append(
            ReproStep(
                n=idx,
                t=scene.t0,
                action=action,
                evidence=[f"frame:{idx - 1}"],
                confidence=0.8 if scene.ui_action else 0.65,
            )
        )
    return steps


def _thin_moments(moments: list[KeyMoment], cap: int) -> list[KeyMoment]:
    """Reduce moments to ``cap``, always keeping errors and spreading the rest in time."""
    errors = [m for m in moments if m.kind == "error"]
    others = [m for m in moments if m.kind != "error"]
    keep = errors[:cap]
    slots = cap - len(keep)
    if slots > 0 and others:
        step = max(1, len(others) // slots)
        keep = keep + others[::step][:slots]
    return sorted(keep, key=lambda m: m.t)


def _derive_key_moments(scenes: list[SceneRecord], transcript: Transcript) -> list[KeyMoment]:
    """Distill the recording into a few salient, timestamped moments.

    This is the analysis backbone for *any* video — it folds the visual scenes and
    the spoken narration into one time-ordered list of "what happens when", each
    anchored to its frame/transcript citation so nothing is fabricated. Consecutive
    duplicate descriptions are collapsed and the list is thinned to a readable size
    while always retaining error moments.
    """
    moments: list[KeyMoment] = []
    for index, scene in enumerate(sorted(scenes, key=lambda s: s.t0)):
        caption = (scene.caption or "").strip()
        action = (scene.ui_action or "").strip()
        if scene.is_error_state:
            kind = "error"
            label = caption or (scene.reason or "").strip() or "Error state observed"
        elif action:
            kind = "action"
            label = caption or action
        else:
            kind = "scene"
            label = caption
        if not label:
            continue
        moments.append(
            KeyMoment(
                t=scene.t0,
                label=label[:_MAX_MOMENT_LABEL_LEN],
                kind=kind,  # type: ignore[arg-type]
                evidence=[f"frame:{index}"],
            )
        )
    for index, segment in enumerate(transcript.segments):
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue
        moments.append(
            KeyMoment(
                t=segment.t0,
                label=text[:_MAX_MOMENT_LABEL_LEN],
                kind="speech",
                evidence=[f"transcript:{index}"],
            )
        )

    moments.sort(key=lambda m: m.t)
    deduped: list[KeyMoment] = []
    last_label: str | None = None
    for moment in moments:
        norm = moment.label.lower()
        if norm == last_label:
            continue
        last_label = norm
        deduped.append(moment)
    if len(deduped) > _MAX_KEY_MOMENTS:
        return _thin_moments(deduped, _MAX_KEY_MOMENTS)
    return deduped


def _derive_error_evidence(scenes: list[SceneRecord]) -> list[ErrorEvidenceItem]:
    """Extract error evidence only from observed OCR or flagged error states."""
    evidence: list[ErrorEvidenceItem] = []
    for scene in scenes:
        if scene.is_error_state and scene.ocr_text.strip():
            evidence.append(
                ErrorEvidenceItem(t=scene.t0, source="ocr", text=scene.ocr_text.strip())
            )
    return evidence


def _renumber(steps: list[ReproStep]) -> list[ReproStep]:
    """Sort steps by timestamp and assign sequential step numbers."""
    ordered = sorted(steps, key=lambda s: s.t)
    return [
        ReproStep(n=i, t=s.t, action=s.action, evidence=s.evidence, confidence=s.confidence)
        for i, s in enumerate(ordered, start=1)
    ]


_ERROR_MARKERS = ("error", "exception", "typeerror", "failed", "undefined", "null", " at ")


def _evidence_rank(item: ErrorEvidenceItem) -> tuple[int, float]:
    """Rank evidence so the most diagnostic item wins (higher score, earlier time).

    Redacted/secret lines carry no diagnostic value and are demoted; genuine
    errors (HTTP failures, exceptions, stack frames) are promoted.
    """
    text = item.text
    meaningful = text.replace("[REDACTED]", "").strip()
    score = 0
    if len(meaningful) < 6:
        score -= 5  # essentially a redacted/empty line
    if item.source == "network":
        score += 3
    lowered = text.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        score += 2
    return (score, -item.t)


def _primary_evidence(evidence: list[ErrorEvidenceItem]) -> ErrorEvidenceItem | None:
    """Select the single most diagnostic error for the title and summary."""
    if not evidence:
        return None
    return max(evidence, key=_evidence_rank)


def _synthesize_title(
    primary: ErrorEvidenceItem | None, scenes: list[SceneRecord], summary: str = ""
) -> str:
    """Derive a concise headline.

    Prefers the strongest error. With no error, describe what was actually
    observed (the first meaningful scene caption, then the summary's opening
    line) so a general recording gets an informative title instead of the generic
    placeholder. Falls back to the placeholder only when there is no evidence at all.
    """
    if primary is not None:
        first = primary.text.splitlines()[0].strip()
        return (first or "Error observed during recorded flow")[: _MAX_TITLE_LEN - 1]
    for scene in scenes:
        caption = (scene.caption or "").strip()
        if caption:
            return caption[: _MAX_TITLE_LEN - 1]
    summary_line = (summary or "").strip().splitlines()[0].strip() if summary else ""
    # Strip a leading markdown heading marker the summary skill may emit.
    summary_line = summary_line.lstrip("#").strip()
    if summary_line:
        return summary_line[: _MAX_TITLE_LEN - 1]
    return "Recorded video (no analyzable content)"


def _assess_quality(
    *,
    degraded_stages: list[str],
    scenes: list[SceneRecord],
    evidence: list[ErrorEvidenceItem],
    cited_steps: list[ReproStep],
    transcript: Transcript,
    keyframes: list[KeyframeRef],
    has_real_steps: bool,
) -> AnalysisQuality:
    """Summarize how trustworthy the bundle is for a downstream agent.

    ``degraded`` means there is essentially nothing to act on (no visual scenes,
    no error evidence, and only the generic fallback repro step); ``partial``
    means some stages degraded but real evidence survived; ``full`` means the
    pipeline ran cleanly. The warnings explain *what* is missing so the consumer
    can gather more rather than guess.
    """
    warnings = [_STAGE_WARNINGS[stage] for stage in degraded_stages if stage in _STAGE_WARNINGS]

    # Any of visual scenes, error evidence, or a spoken transcript is enough to
    # analyze a general video — a narrated clip with no UI is still summarizable.
    has_evidence = bool(scenes or evidence or transcript.segments)
    if not has_evidence and not has_real_steps:
        level: str = "degraded"
        warnings.append(
            "Insufficient evidence was extracted from the recording — treat findings "
            "as low confidence and gather more (re-record, attach console/network logs)."
        )
    elif degraded_stages:
        level = "partial"
    else:
        level = "full"

    return AnalysisQuality(
        level=level,  # type: ignore[arg-type]
        degraded_stages=list(degraded_stages),
        warnings=warnings,
        evidence_counts={
            "keyframes": len(keyframes),
            "error_evidence": len(evidence),
            "repro_steps": len(cited_steps),
            "scenes": len(scenes),
            "transcript_segments": len(transcript.segments),
        },
        # Refined later by ``confidence.assess_actionability`` once the bundle
        # (build context, candidates) is assembled; "ready" is the neutral default.
        actionability="ready",
    )


def extract_bug_context_bundle(
    *,
    job_id: str,
    source_video: str,
    duration_s: float,
    classification: Classification,
    transcript: Transcript,
    scenes: list[SceneRecord],
    keyframes: list[KeyframeRef],
    environment: dict[str, str],
    sidecar_steps: list[ReproStep] | None = None,
    sidecar_evidence: list[ErrorEvidenceItem] | None = None,
    degraded_stages: list[str] | None = None,
    summary: str = "",
) -> ContextBundle:
    """Build the canonical bundle, merging visual and sidecar evidence without fabrication.

    The shape adapts to what the video is. When it depicts a defect — a ``bug``
    classification or any real error evidence — the bug-oriented fields (severity,
    priority, expected/actual behavior, preconditions, a guaranteed repro step) are
    populated. For a *general* video (a demo, walkthrough, real-world clip) those
    fields stay ``None``/empty and the deliverable is the ``summary`` plus the
    derived ``key_moments`` — no fabricated "expected behavior" or severity.
    """
    scene_steps = _derive_repro_steps(scenes)
    all_steps = scene_steps + list(sidecar_steps or [])
    real_steps = _renumber([step for step in all_steps if step.evidence])
    has_real_steps = bool(real_steps)
    # The synthetic fallback step only makes sense for a bug we want reproduced.
    fallback_step = ReproStep(
        n=1,
        t=0.0,
        action="Open the page and reproduce the observed behavior",
        evidence=["sidecar:env"],
        confidence=0.5,
    )

    evidence = _derive_error_evidence(scenes) + list(sidecar_evidence or [])
    evidence.sort(key=lambda item: item.t)
    key_moments = _derive_key_moments(scenes, transcript)

    quality = _assess_quality(
        degraded_stages=list(degraded_stages or []),
        scenes=scenes,
        evidence=evidence,
        cited_steps=real_steps or [fallback_step],
        transcript=transcript,
        keyframes=keyframes,
        has_real_steps=has_real_steps,
    )

    primary = _primary_evidence(evidence)
    title = _synthesize_title(primary, scenes, summary)

    # A bug bundle carries the diagnostic fields; a general bundle suppresses them.
    is_bug = classification.label is ClassificationLabel.BUG or bool(evidence)

    if is_bug:
        if primary is not None:
            actual_behavior: str | None = primary.text.strip()
        elif quality.level == "degraded":
            # Be honest: we could not extract enough to describe behavior. Do not imply
            # the flow succeeded — that would mislead a downstream agent into "no-op".
            actual_behavior = (
                "Analysis incomplete — not enough evidence was extracted to describe "
                "the observed behavior (see analysis_quality.warnings)."
            )
        else:
            actual_behavior = "Recorded flow completed; no explicit error surfaced."
        return ContextBundle(
            schema_version="1.0",
            id=job_id,
            source_video=source_video,
            duration_s=duration_s,
            created_at=datetime.now(UTC),
            classification=classification,
            reproducibility=Reproducibility.SHOWN_ONCE,
            title=title,
            severity=Severity.HIGH if evidence else Severity.MEDIUM,
            priority=Priority.P1 if evidence else Priority.P2,
            suspected_component=environment.get("component", "unknown"),
            environment=environment,
            preconditions="User is authenticated and page is loaded.",
            repro_steps=real_steps or [fallback_step],
            expected_behavior="Action completes successfully without errors.",
            actual_behavior=actual_behavior,
            error_evidence=evidence,
            keyframe_refs=keyframes,
            key_moments=key_moments,
            analysis_quality=quality,
            summary=summary,
            transcript_path="transcript.json" if transcript.segments else None,
            timeline_path="timeline.json",
            redactions=[],
            code_candidates=[],
        )

    # General video: summary + key moments are the substance; bug fields stay null.
    return ContextBundle(
        schema_version="1.0",
        id=job_id,
        source_video=source_video,
        duration_s=duration_s,
        created_at=datetime.now(UTC),
        classification=classification,
        reproducibility=None,
        title=title,
        severity=None,
        priority=None,
        suspected_component=environment.get("component") or None,
        environment=environment,
        preconditions=None,
        repro_steps=real_steps,
        expected_behavior=None,
        actual_behavior=None,
        error_evidence=evidence,
        keyframe_refs=keyframes,
        key_moments=key_moments,
        analysis_quality=quality,
        summary=summary,
        transcript_path="transcript.json" if transcript.segments else None,
        timeline_path="timeline.json",
        redactions=[],
        code_candidates=[],
    )
