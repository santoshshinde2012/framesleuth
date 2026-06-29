"""Tests for classify, extract, redact, and grounding modules."""

from pathlib import Path

from framesleuth.pipeline.bug_extract import extract_bug_context_bundle
from framesleuth.pipeline.classify import classify_video
from framesleuth.pipeline.grounding import locate_in_code
from framesleuth.pipeline.redact import redact_text
from framesleuth.schemas import (
    ClassificationLabel,
    ErrorEvidenceItem,
    KeyframeRef,
    SceneRecord,
    Transcript,
)


def test_classify_bug_label_when_error_signals_present() -> None:
    """Classification should route to bug when multiple error signals are present."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.0, t1=1.0, text="we hit an exception", conf=0.9)],
        words=[],
    )
    scenes = [
        SceneRecord(
            t0=0.5,
            t1=1.0,
            caption="error modal",
            ocr_text="TypeError: undefined",
            ui_action="click",
            is_error_state=True,
            reason="exception",
        )
    ]

    cls = classify_video(transcript, scenes)
    assert cls.label == ClassificationLabel.BUG
    # A confident bug lists "other" as the runner-up, never a phantom "tutorial".
    assert cls.alt_labels == [] or cls.alt_labels[0][0] == ClassificationLabel.OTHER


def test_classify_no_evidence_omits_phantom_alt_label() -> None:
    """A degraded run with no bug/tutorial signal must not assert 'tutorial: 1.0'."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.0, t1=2.0, text="on Friday we meet investors", conf=0.9)],
        words=[],
    )
    cls = classify_video(transcript, [])
    assert cls.label == ClassificationLabel.OTHER
    assert cls.confidence == 0.0
    assert cls.alt_labels == []  # honest: no competing hypothesis, not tutorial=1.0


def test_classify_tutorial_signal_yields_tutorial_alt() -> None:
    """An explicit 'how to' narration surfaces tutorial as the alternative."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.0, t1=2.0, text="how to set up the project", conf=0.9)],
        words=[],
    )
    cls = classify_video(transcript, [])
    assert cls.alt_labels and cls.alt_labels[0][0] == ClassificationLabel.TUTORIAL


def test_extract_bundle_drops_uncited_steps_and_is_schema_valid() -> None:
    """Bundle extraction should always emit cited repro steps."""
    transcript = Transcript(segments=[], words=[])
    scenes = [
        SceneRecord(
            t0=1.0,
            t1=1.0,
            caption="open settings",
            ocr_text="",
            ui_action="open settings",
            is_error_state=False,
            reason=None,
        )
    ]
    keyframes = [KeyframeRef(index=0, t=1.0, shows="state", file="frames/0.png")]

    bundle = extract_bug_context_bundle(
        job_id="job-1",
        source_video="video.mp4",
        duration_s=12.0,
        classification=classify_video(transcript, scenes),
        transcript=transcript,
        scenes=scenes,
        keyframes=keyframes,
        environment={"os": "macOS", "browser": "Chrome", "component": "settings"},
    )

    assert bundle.repro_steps
    assert all(step.evidence for step in bundle.repro_steps)


def test_title_describes_scene_when_no_error() -> None:
    """A non-bug recording titles from what was observed, not a generic placeholder."""
    transcript = Transcript(segments=[], words=[])
    scenes = [
        SceneRecord(
            t0=0.3,
            t1=1.0,
            caption="A hands-on look at Google's new Data Agent Kit",
            ocr_text="",
            ui_action=None,
            is_error_state=False,
            reason=None,
        )
    ]
    bundle = extract_bug_context_bundle(
        job_id="job-title",
        source_video="rec.webm",
        duration_s=3.0,
        classification=classify_video(transcript, scenes),
        transcript=transcript,
        scenes=scenes,
        keyframes=[KeyframeRef(index=0, t=0.3, shows="scene", file="keyframes/000.png")],
        environment={},
    )
    assert bundle.title == "A hands-on look at Google's new Data Agent Kit"
    assert "Observed UI behavior" not in bundle.title


def test_title_falls_back_to_placeholder_without_visual_evidence() -> None:
    """With no scenes and no error, the generic placeholder title is retained."""
    transcript = Transcript(segments=[], words=[])
    bundle = extract_bug_context_bundle(
        job_id="job-noscene",
        source_video="rec.webm",
        duration_s=0.0,
        classification=classify_video(transcript, []),
        transcript=transcript,
        scenes=[],
        keyframes=[],
        environment={},
        degraded_stages=["understand"],
    )
    assert bundle.title == "Recorded video (no analyzable content)"


def _empty_bundle(degraded_stages: list[str]):
    """Build a bundle with no visual/sidecar evidence and given degraded stages."""
    transcript = Transcript(segments=[], words=[])
    return extract_bug_context_bundle(
        job_id="job-q",
        source_video="v.webm",
        duration_s=0.0,
        classification=classify_video(transcript, []),
        transcript=transcript,
        scenes=[],
        keyframes=[],
        environment={},
        degraded_stages=degraded_stages,
    )


def test_quality_degraded_when_no_evidence_and_honest_actual_behavior() -> None:
    """A no-evidence run must self-report as degraded, not as a clean success."""
    bundle = _empty_bundle(["preprocess", "understand"])

    assert bundle.analysis_quality.level == "degraded"
    assert "preprocess" in bundle.analysis_quality.degraded_stages
    assert bundle.analysis_quality.warnings  # explains what is missing
    # A no-error run is a general bundle: bug-shaped behavior fields stay null and
    # the honesty about insufficiency lives in analysis_quality.warnings.
    assert bundle.actual_behavior is None
    assert bundle.expected_behavior is None
    assert bundle.severity is None
    assert any("insufficient" in w.lower() for w in bundle.analysis_quality.warnings)
    assert bundle.analysis_quality.evidence_counts["error_evidence"] == 0


def test_quality_partial_when_degraded_but_evidence_present() -> None:
    """Real sidecar evidence with a degraded visual stage is 'partial', not degraded."""
    transcript = Transcript(segments=[], words=[])
    bundle = extract_bug_context_bundle(
        job_id="job-p",
        source_video="v.webm",
        duration_s=3.0,
        classification=classify_video(transcript, []),
        transcript=transcript,
        scenes=[],
        keyframes=[],
        environment={},
        sidecar_evidence=[ErrorEvidenceItem(t=1.0, source="console", text="TypeError: boom")],
        degraded_stages=["understand"],
    )

    assert bundle.analysis_quality.level == "partial"
    assert bundle.actual_behavior == "TypeError: boom"
    assert bundle.analysis_quality.evidence_counts["error_evidence"] == 1


def test_quality_full_when_clean_run() -> None:
    """No degraded stages and real evidence yields a 'full' quality signal."""
    transcript = Transcript(segments=[], words=[])
    scenes = [
        SceneRecord(
            t0=1.0,
            t1=2.0,
            caption="error modal",
            ocr_text="TypeError: undefined",
            ui_action="click",
            is_error_state=True,
            reason="exception",
        )
    ]
    bundle = extract_bug_context_bundle(
        job_id="job-f",
        source_video="v.mp4",
        duration_s=3.0,
        classification=classify_video(transcript, scenes),
        transcript=transcript,
        scenes=scenes,
        keyframes=[KeyframeRef(index=0, t=1.0, shows="failure", file="keyframes/000.png")],
        environment={"browser": "Chrome"},
        degraded_stages=[],
    )

    assert bundle.analysis_quality.level == "full"
    assert bundle.analysis_quality.degraded_stages == []


def test_redact_text_masks_secrets() -> None:
    """Secret-like tokens should be replaced and tracked as redactions."""
    text = "Bearer abcdefghijklmnopqrstuvwxyz and password=secret123"
    redacted, redactions = redact_text(text, timestamp=3.2)

    assert "[REDACTED]" in redacted
    assert len(redactions) >= 1


def test_grounding_locates_candidates(tmp_path: Path) -> None:
    """Grounding should return deterministic ranked file candidates."""
    source = tmp_path / "module.py"
    source.write_text("def handler():\n    raise ValueError('boom')\n", encoding="utf-8")

    candidates = locate_in_code(tmp_path, ["ValueError", "handler"], max_results=5)
    assert candidates
    assert candidates[0].file == "module.py"
    assert candidates[0].line in {1, 2}
