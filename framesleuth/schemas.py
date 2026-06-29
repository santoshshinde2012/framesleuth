"""Data contracts and schemas for Framesleuth.

All data structures use Pydantic v2 for validation, serialization, and documentation.
Follows the interface segregation principle with focused, composable schemas.
"""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ===== Enums =====


class ClassificationLabel(StrEnum):
    """Video classification labels.

    ``feature`` covers "build/add/change this" intent — a feature demo, a design
    walkthrough, or a spoken request to implement something. It routes to the
    ``implement`` action and drives ``BuildContext`` extraction, so the agent is a
    first-class build assistant, not only a bug fixer.
    """

    BUG = "bug"
    FEATURE = "feature"
    TUTORIAL = "tutorial"
    DEMO = "demo"
    FEEDBACK = "feedback"
    OTHER = "other"


class Reproducibility(StrEnum):
    """Reproducibility of the reported issue."""

    SHOWN_ONCE = "shown_once"
    SHOWN_MULTIPLE = "shown_multiple"
    INTERMITTENT = "intermittent"
    CONSISTENT = "consistent"


class Severity(StrEnum):
    """Severity level of the bug."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(StrEnum):
    """Priority level for fixing."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class JobState(StrEnum):
    """Job processing state (mirrors the orchestrator's stage transitions)."""

    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    UNDERSTANDING = "understanding"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    GROUNDING = "grounding"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ===== Input Contracts =====
#
# Raw browser sidecars arrive as loosely-typed JSON and are normalized by
# ``framesleuth.pipeline.sidecars`` (which tolerates both the flat event stream
# and the structured dict shape), so there is no rigid input model here.


class Transcript(BaseModel):
    """Timestamped transcript from audio."""

    class Segment(BaseModel):
        """Transcript segment."""

        t0: float = Field(..., description="Start time in seconds")
        t1: float = Field(..., description="End time in seconds")
        text: str = Field(..., description="Transcribed text")
        conf: float = Field(..., ge=0, le=1, description="Confidence 0-1")

    segments: list[Segment]
    words: list[dict[str, Any]] | None = Field(None, description="Word-level timing if available")
    language: str | None = Field(
        default=None, description="Detected/forced ISO language code, if known"
    )


class UiElement(BaseModel):
    """A structured UI element observed in a frame (build/feature context).

    Captured so a coding agent can *rebuild* what was shown, not just read a
    caption. Populated by the build-aware frame prompt for non-bug videos.
    """

    kind: str = Field(
        ...,
        description="button | input | link | text | image | icon | list | table | "
        "modal | nav | card | tab | toggle | other",
    )
    label: str = Field(..., description="Visible text/label on the element")
    state: str | None = Field(
        None, description="Observed state, e.g. disabled, active, focused, selected, error"
    )


class SceneRecord(BaseModel):
    """Visual scene record from frame analysis."""

    t0: float = Field(..., description="Scene start time (seconds)")
    t1: float = Field(..., description="Scene end time (seconds)")
    caption: str = Field(..., description="What is visible in the scene")
    ocr_text: str = Field(..., description="All visible text in the scene")
    ui_action: str | None = Field(None, description="Apparent user action (click, type, etc.)")
    is_error_state: bool = Field(False, description="Whether scene shows an error or failure")
    reason: str | None = Field(None, description="Why this frame is marked as error state")
    # Build/feature context — populated by the build-aware prompt for non-bug videos.
    ui_elements: list[UiElement] = Field(
        default_factory=list, description="Structured UI elements observed in the frame"
    )
    layout: str | None = Field(
        None, description="Spatial layout, e.g. 'sidebar left, main content right, modal centered'"
    )
    screen_name: str | None = Field(
        None, description="Inferred screen/page/route name (from title, URL, or heading)"
    )
    design_notes: str | None = Field(
        None, description="Colors, typography, spacing, and visual style observed"
    )
    data_shown: str | None = Field(
        None, description="Structured data visible, e.g. table columns or list item shape"
    )


class PreprocessResult(BaseModel):
    """Result of video preprocessing."""

    video_path: Path
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
    audio_path: Path | None
    frame_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ===== Output Contracts =====


class Classification(BaseModel):
    """Classification result with confidence and alternatives."""

    label: ClassificationLabel
    confidence: float = Field(..., ge=0, le=1)
    alt_labels: list[tuple[ClassificationLabel, float]] = Field(
        default_factory=list, description="Alternative labels and confidences"
    )


class ReproStep(BaseModel):
    """Numbered reproduction step with evidence and confidence."""

    n: int = Field(..., ge=1, description="Step number")
    t: float = Field(..., description="Timestamp in seconds")
    action: str = Field(..., description="What the user did")
    evidence: list[str] = Field(..., description="Citations like 'frame:5' or 'transcript:0:08'")
    confidence: float = Field(..., ge=0, le=1)


class ErrorEvidenceItem(BaseModel):
    """Error or failure indicator with source and timing."""

    t: float = Field(..., description="Timestamp in seconds")
    source: Literal["console", "ocr", "network", "ui"] = Field(
        ..., description="Where the error came from"
    )
    text: str = Field(..., description="Error message or observed behavior")


class KeyframeRef(BaseModel):
    """Reference to a keyframe image."""

    index: int = Field(..., description="Keyframe index")
    t: float = Field(..., description="Timestamp in seconds")
    shows: str = Field(..., description="What this keyframe shows")
    file: str = Field(..., description="Path relative to bundle root")


class KeyMoment(BaseModel):
    """A salient, timestamped moment in the recording.

    The building block of a general-video *analysis*: a short, time-anchored
    description of what happens at a point in the video (a scene change, an
    action, a spoken point, or an error). Derived deterministically from the
    fused video + audio timeline, so it carries an evidence citation and never
    fabricates.
    """

    t: float = Field(..., description="Timestamp in seconds")
    label: str = Field(..., description="Short description of what happens at this moment")
    kind: Literal["scene", "action", "speech", "error"] = Field(
        "scene", description="What kind of moment this is"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Citations like 'frame:3' or 'transcript:0:08'"
    )


class Redaction(BaseModel):
    """Record of a redaction applied to protect sensitive data."""

    t: float = Field(..., description="Timestamp where redaction occurred")
    region: str = Field(..., description="Description of redacted region")
    applied: bool = Field(..., description="Whether redaction was successfully applied")


class CodeCandidate(BaseModel):
    """Candidate code location matched by grounding."""

    file: str
    line: int
    symbol: str | None = None
    match_reason: str = Field(
        ..., description="How this was matched (stacktrace/search/route/label)"
    )
    confidence: float = Field(..., ge=0, le=1)
    is_third_party: bool = Field(False)


class AnalysisQuality(BaseModel):
    """How much of the pipeline succeeded — the trust signal for consumers.

    Downstream agents (Copilot/Claude/tools) read ``level`` to decide whether to
    act confidently, act cautiously, or ask the user for more evidence instead of
    fabricating a fix from a near-empty bundle.
    """

    level: Literal["full", "partial", "degraded"] = Field(
        ..., description="full=all stages ran; partial=some degraded; degraded=little evidence"
    )
    degraded_stages: list[str] = Field(
        default_factory=list, description="Pipeline stages that degraded (e.g. understand, asr)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Human-readable notes on what is missing or uncertain"
    )
    evidence_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of extracted evidence (keyframes, errors, repro_steps, transcript)",
    )
    actionability: Literal["ready", "thin", "insufficient"] = Field(
        "ready",
        description="Whether the evidence suffices for the RESOLVED action (independent of which "
        "stages ran). ready=act on it; thin=act but flag gaps; insufficient=gather more.",
    )


class UiComponent(BaseModel):
    """A distinct UI component aggregated across frames (build/feature context)."""

    kind: str = Field(
        ..., description="button | input | form | list | table | modal | nav | card | …"
    )
    label: str = Field(..., description="Visible label/identifier")
    screen: str | None = Field(None, description="Screen/page where it appears")
    states: list[str] = Field(default_factory=list, description="Observed states across frames")
    evidence: list[str] = Field(default_factory=list, description="Frame citations, e.g. 'frame:3'")


class Screen(BaseModel):
    """A distinct screen/page/route shown in the video."""

    name: str = Field(..., description="Screen/page/route name")
    summary: str = Field("", description="One-line description of the screen")
    t: float = Field(..., description="First-seen timestamp (seconds)")
    components: list[str] = Field(
        default_factory=list, description="Component labels on this screen"
    )
    evidence: list[str] = Field(default_factory=list, description="Frame citations")


class FlowStep(BaseModel):
    """A transition in the user flow: from one screen to the next via an action."""

    n: int = Field(..., ge=1, description="Step number in the flow")
    from_screen: str | None = Field(None, description="Screen before the action")
    action: str | None = Field(None, description="The action that caused the transition")
    to_screen: str | None = Field(None, description="Screen after the action")
    t: float = Field(..., description="Timestamp of the transition (seconds)")


class BuildContext(BaseModel):
    """Structured "what to build" context for feature / build / demo / design videos.

    This is the build counterpart to the bug-oriented fields. It gives a coding
    agent a buildable model of the video — screens, components, the user flow
    between them, design notes, and where in the codebase to implement — instead
    of a flat caption + OCR blob.
    """

    screens: list[Screen] = Field(default_factory=list)
    components: list[UiComponent] = Field(default_factory=list)
    user_flow: list[FlowStep] = Field(
        default_factory=list, description="Screen-to-screen transitions"
    )
    design_notes: list[str] = Field(
        default_factory=list, description="Colors, typography, spacing, visual style observed"
    )
    data_models: list[str] = Field(
        default_factory=list, description="Data shapes shown (table columns, list item fields)"
    )
    is_greenfield: bool = Field(
        False, description="True when this appears net-new — no matching existing code was found"
    )
    target_locations: list[str] = Field(
        default_factory=list,
        description="Where to implement: existing dirs/files to extend, or new-file hints",
    )


class ContextBundle(BaseModel):
    """The canonical output: complete structured context extracted from a video.

    Framesleuth works on *any* video — a bug recording, a feature demo, a design
    walkthrough, a Loom, a phone capture — and distills it into this single
    structured artifact a coding agent can act on: to fix a bug, add or change a
    feature, or build something new. Bug-oriented fields (``severity``,
    ``priority``, ``suspected_component``) double as generic triage signals for
    non-bug work (how urgent/important the task is and where it lives in the code).

    This is the primary artifact delivered to both VS Code and Chrome surfaces.
    Schema versioning enables forward migration and compatibility checks.

    Bug-oriented fields (``severity``, ``priority``, ``suspected_component``,
    ``preconditions``, ``repro_steps``, ``expected_behavior``,
    ``actual_behavior``, ``reproducibility``) are populated only when the video
    actually depicts a defect (a ``bug`` classification or real error evidence).
    For a general video — a demo, a walkthrough, a real-world clip — they are
    ``None``/empty and the deliverable is the ``summary`` plus ``key_moments``.
    """

    schema_version: str = "1.0"
    id: str = Field(..., description="Unique job ID")
    source_video: str = Field(..., description="Original video filename")
    duration_s: float = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    classification: Classification
    reproducibility: Reproducibility | None = Field(
        default=None, description="How reproducible the issue is (bug videos only)"
    )
    title: str = Field(..., max_length=200)
    severity: Severity | None = Field(default=None, description="Bug severity (bug videos only)")
    priority: Priority | None = Field(default=None, description="Fix priority (bug videos only)")
    suspected_component: str | None = Field(
        default=None, max_length=200, description="Suspected component (bug videos only)"
    )

    environment: dict[str, str] = Field(
        default_factory=dict, description="OS, app, version, browser from OCR or sidecar"
    )
    preconditions: str | None = Field(
        default=None, description="Prerequisites for reproduction (bug videos only)"
    )
    repro_steps: list[ReproStep] = Field(
        default_factory=list, description="Reproduction/observed steps (empty for general video)"
    )
    expected_behavior: str | None = Field(
        default=None, description="Expected behavior (bug videos only)"
    )
    actual_behavior: str | None = Field(
        default=None, description="Observed behavior (bug videos only)"
    )

    error_evidence: list[ErrorEvidenceItem] = Field(default_factory=list)
    keyframe_refs: list[KeyframeRef] = Field(default_factory=list)
    key_moments: list[KeyMoment] = Field(
        default_factory=list,
        description="Salient timestamped moments — the analysis backbone for any video",
    )

    build_context: BuildContext | None = Field(
        default=None,
        description="Structured build/feature context (screens, components, user flow, design). "
        "Populated for feature/demo/build videos; null for pure bug reports.",
    )

    analysis_quality: AnalysisQuality = Field(
        default_factory=lambda: AnalysisQuality(level="full", actionability="ready"),
        description="Pipeline completeness/confidence signal for downstream consumers",
    )

    field_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence 0-1 for key fields (title, repro_steps, severity, "
        "suspected_component, build_context) so consumers know which claims to trust.",
    )

    summary: str = Field(
        default="",
        description="Narrative summary/analysis of the recording (video + audio), per the chosen "
        "skill. The primary deliverable for general (non-bug) videos.",
    )
    skill: str | None = Field(
        default=None,
        description="Skill/system-prompt label used for the summary (e.g. 'summary', 'custom')",
    )

    user_intent: str | None = Field(
        default=None, description="The user's natural-language request to act on, if any"
    )

    action: str | None = Field(
        default=None,
        description="Resolved action mode shaping the fix-prompt (e.g. 'fix', 'explain', 'custom')",
    )
    action_prompt: str | None = Field(
        default=None,
        description="Custom action task text to render (set only when action == 'custom')",
    )
    suggested_actions: list[dict[str, str]] = Field(
        default_factory=list,
        description="Machine-readable next-step menu (action/label/rationale/ref) for consumers",
    )

    stage_timings: dict[str, float] = Field(
        default_factory=dict,
        description="Per-stage wall-clock seconds (preprocess, asr, understand, summarize, "
        "grounding…) so consumers can see where analysis time went.",
    )

    transcript_path: str | None = Field(
        None, description="Path to transcript.json relative to bundle"
    )
    timeline_path: str | None = Field(None, description="Path to timeline.json relative to bundle")
    redactions: list[Redaction] = Field(default_factory=list, description="Redactions applied")
    code_candidates: list[CodeCandidate] = Field(
        default_factory=list, description="Ranked code locations"
    )

    @field_validator("repro_steps")
    @classmethod
    def validate_repro_steps(cls, v: list[ReproStep]) -> list[ReproStep]:
        """Validate that all repro steps are numbered sequentially."""
        for i, step in enumerate(v, 1):
            if step.n != i:
                raise ValueError(
                    f"Repro steps must be numbered sequentially, got {step.n} at position {i}"
                )
        return v

    def validate_claims_cited(self) -> list[str]:
        """Validate that the bundle carries a usable, cited deliverable.

        A bug bundle must have reproduction steps; a general video instead carries
        a summary and/or key moments. A bundle missing *all* of these (and a title)
        has nothing actionable to stand on.

        Returns:
            List of missing-deliverable keys (should be empty for a valid bundle).
        """
        missing: list[str] = []
        if not self.title:
            missing.append("title")
        is_bug = self.classification.label is ClassificationLabel.BUG or bool(self.error_evidence)
        if is_bug:
            if not self.repro_steps:
                missing.append("repro_steps")
        elif not (self.summary or self.key_moments):
            missing.append("summary_or_key_moments")
        return missing
