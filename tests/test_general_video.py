"""End-to-end coverage for the general-video (summary/analysis) path.

These tests pin the behavior added so the agent works on *any* video — not just
bug recordings: bug-shaped fields are suppressed for non-bug videos, the summary
and timestamped key moments become the deliverable, and the action/prompt/render
surfaces adapt accordingly.
"""

from __future__ import annotations

from framesleuth.actions import ACTIONS, auto_action_for, list_actions
from framesleuth.pipeline.bug_extract import _derive_key_moments, extract_bug_context_bundle
from framesleuth.pipeline.classify import classify_video
from framesleuth.pipeline.confidence import assess_actionability, compute_field_confidence
from framesleuth.prompts import FixPrompts, VLMPrompts
from framesleuth.render import render_markdown
from framesleuth.schemas import (
    ClassificationLabel,
    ContextBundle,
    KeyframeRef,
    SceneRecord,
    Transcript,
)


def _general_bundle(summary: str = "A short walkthrough of the new dashboard.") -> ContextBundle:
    """Build a bundle for a non-bug general recording (a demo with narration)."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=0.0, t1=2.0, text="Here is the dashboard", conf=0.9)],
        words=[],
    )
    scenes = [
        SceneRecord(
            t0=0.5,
            t1=1.0,
            caption="A dashboard with revenue charts",
            ocr_text="Revenue",
            ui_action=None,
            is_error_state=False,
        ),
        SceneRecord(
            t0=3.0,
            t1=3.5,
            caption="User opens the settings panel",
            ocr_text="Settings",
            ui_action="click",
            is_error_state=False,
        ),
    ]
    classification = classify_video(transcript, scenes)
    return extract_bug_context_bundle(
        job_id="general-1",
        source_video="demo.mp4",
        duration_s=5.0,
        classification=classification,
        transcript=transcript,
        scenes=scenes,
        keyframes=[KeyframeRef(index=0, t=0.5, shows="scene", file="keyframes/000.png")],
        environment={},
        summary=summary,
    )


def test_general_video_is_not_classified_as_bug() -> None:
    """A plain walkthrough with no errors must not be labelled a bug."""
    bundle = _general_bundle()
    assert bundle.classification.label is not ClassificationLabel.BUG


def test_general_bundle_suppresses_bug_fields() -> None:
    """Bug-shaped fields are null for a general video — no fabricated placeholders."""
    bundle = _general_bundle()
    assert bundle.severity is None
    assert bundle.priority is None
    assert bundle.reproducibility is None
    assert bundle.preconditions is None
    assert bundle.expected_behavior is None
    assert bundle.actual_behavior is None
    # The old hardcoded placeholders must not leak through.
    serialized = bundle.model_dump_json()
    assert "User is authenticated and page is loaded" not in serialized
    assert "Action completes successfully without errors" not in serialized


def test_general_bundle_leads_with_summary_and_key_moments() -> None:
    """The summary and timestamped key moments are the deliverable for general video."""
    bundle = _general_bundle()
    assert bundle.summary == "A short walkthrough of the new dashboard."
    assert bundle.key_moments
    # Title is informative, not the generic placeholder.
    assert "no analyzable content" not in bundle.title
    assert bundle.title


def test_key_moments_fuse_scenes_and_speech_with_citations() -> None:
    """Key moments span visual scenes, actions, errors, and narration, each cited."""
    transcript = Transcript(
        segments=[Transcript.Segment(t0=4.0, t1=5.0, text="and now it crashes", conf=0.9)],
        words=[],
    )
    scenes = [
        SceneRecord(t0=0.0, t1=1.0, caption="Home screen", ocr_text="", ui_action=None),
        SceneRecord(t0=1.0, t1=2.0, caption="Clicks save", ocr_text="", ui_action="click"),
        SceneRecord(
            t0=3.0,
            t1=4.0,
            caption="Error dialog appears",
            ocr_text="TypeError",
            ui_action=None,
            is_error_state=True,
            reason="Exception dialog visible",
        ),
    ]
    moments = _derive_key_moments(scenes, transcript)
    kinds = {m.kind for m in moments}
    assert {"scene", "action", "error", "speech"} <= kinds
    assert all(m.evidence for m in moments)  # every moment is cited
    assert any(m.evidence[0].startswith("transcript:") for m in moments)


def test_key_moments_are_capped() -> None:
    """A long recording is thinned to a readable set, keeping errors."""
    scenes = [
        SceneRecord(
            t0=float(i),
            t1=float(i) + 0.5,
            caption=f"scene {i}",
            ocr_text="",
            ui_action=None,
            is_error_state=(i == 30),
        )
        for i in range(40)
    ]
    moments = _derive_key_moments(scenes, Transcript(segments=[], words=[]))
    assert len(moments) <= 12
    assert any(m.kind == "error" for m in moments)  # error moment is retained


def test_summarize_action_registered_and_auto_picked() -> None:
    """The summarize action exists and is the default for a general ('other') video."""
    assert "summarize" in {a["name"] for a in list_actions()}
    assert auto_action_for("other") == "summarize"
    assert auto_action_for("bug") == "fix"


def test_summarize_is_actionable_with_only_a_summary() -> None:
    """A summary (or key moments) is sufficient for the summarize action."""
    bundle = _general_bundle()
    bundle.action = "summarize"
    assert assess_actionability(bundle) == "ready"


def test_field_confidence_skips_bug_fields_for_general_video() -> None:
    """Confidence is reported for summary/key moments, not for suppressed bug fields."""
    bundle = _general_bundle()
    conf = compute_field_confidence(bundle)
    assert conf.get("summary")
    assert conf.get("key_moments")
    assert "severity" not in conf  # null severity is not scored


def test_fix_prompt_general_video_is_summary_shaped() -> None:
    """For a general video the prompt surfaces the summary and omits empty bug sections."""
    prompt = FixPrompts.fix_from_video(
        title="Dashboard walkthrough",
        severity="",
        component="",
        environment={},
        repro_steps=[],
        expected="",
        actual="",
        errors=[],
        candidates=[],
        summary="A walkthrough of the analytics dashboard.",
        task=ACTIONS["summarize"].task,
    )
    assert "Summary of the recording" in prompt
    assert "A walkthrough of the analytics dashboard." in prompt
    assert "## Expected behavior:" not in prompt
    assert "## Actual behavior:" not in prompt
    assert "## Error evidence" not in prompt
    # Neutral framing, not the bug/repo-only intro.
    assert "expert engineer working in the user's repository" not in prompt


def test_render_markdown_general_video_is_summary_first() -> None:
    """The markdown render leads with summary + key moments and drops bug sections."""
    report = _general_bundle().model_dump(mode="json")
    md = render_markdown(report)
    assert "## Summary" in md
    assert "## Key moments" in md
    assert "Expected vs actual" not in md
    assert "n/a" not in md  # no empty bug placeholders


def test_frame_analysis_prompt_is_general_purpose() -> None:
    """The default frame prompt no longer assumes a software screen recording."""
    prompt = VLMPrompts.frame_analysis(1.0).lower()
    assert "frame of a video" in prompt
    # Error detection is now conditional on the frame actually being software.
    assert "only" in prompt and "software" in prompt
