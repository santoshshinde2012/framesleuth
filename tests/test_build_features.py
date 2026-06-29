"""Tests for the build/feature pipeline: classification, build context, flow,
grounding, confidence/actionability, actions, and the build prompt block."""

from __future__ import annotations

from pathlib import Path

from framesleuth.actions import auto_action_for, suggest_actions
from framesleuth.pipeline.build_context import build_build_context, derive_user_flow
from framesleuth.pipeline.classify import (
    classify_video,
    feature_intent_signal,
    looks_like_build_intent,
)
from framesleuth.pipeline.confidence import assess_actionability, compute_field_confidence
from framesleuth.pipeline.grounding import locate_in_code
from framesleuth.prompts import FixPrompts, VLMPrompts
from framesleuth.schemas import (
    AnalysisQuality,
    Classification,
    ClassificationLabel,
    CodeCandidate,
    ContextBundle,
    Priority,
    Reproducibility,
    SceneRecord,
    Severity,
    Transcript,
    UiElement,
)


def _scene(**kw: object) -> SceneRecord:
    base: dict[str, object] = {"t0": 0.0, "t1": 0.0, "caption": "", "ocr_text": ""}
    base.update(kw)
    return SceneRecord(**base)  # type: ignore[arg-type]


def _empty_transcript() -> Transcript:
    return Transcript(segments=[])


# --------------------------------------------------------------------------- #
# D. Feature classification + build-intent helpers
# --------------------------------------------------------------------------- #


def test_feature_intent_signal_detects_build_and_suppresses_fix() -> None:
    assert feature_intent_signal("add a dark mode toggle") >= 0.5
    assert feature_intent_signal("build this settings page") >= 0.5
    assert feature_intent_signal("fix the broken save button") < 0.5
    assert looks_like_build_intent("create a new profile screen")
    assert not looks_like_build_intent(None)


def test_classify_feature_from_intent() -> None:
    result = classify_video(
        _empty_transcript(),
        [_scene(ocr_text="Settings")],
        user_intent="add a dark mode toggle like the demo",
    )
    assert result.label is ClassificationLabel.FEATURE


def test_classify_bug_still_wins_over_feature_when_errors_present() -> None:
    result = classify_video(
        _empty_transcript(),
        [_scene(ocr_text="Checkout", is_error_state=True)],
        error_signals=["500 Internal Server Error"],
        user_intent="fix the broken checkout",
    )
    assert result.label is ClassificationLabel.BUG


# --------------------------------------------------------------------------- #
# C. Actions
# --------------------------------------------------------------------------- #


def test_feature_auto_maps_to_implement() -> None:
    assert auto_action_for("feature") == "implement"


def test_suggest_actions_offers_implement_for_feature() -> None:
    report = {
        "id": "r1",
        "classification": {"label": "feature"},
        "analysis_quality": {"level": "full"},
        "build_context": {"screens": [{"name": "Settings"}]},
    }
    actions = {a["action"] for a in suggest_actions(report)}
    assert "implement" in actions
    assert "design" in actions


# --------------------------------------------------------------------------- #
# A + G. Scene UI extraction + temporal flow
# --------------------------------------------------------------------------- #


def test_build_prompt_extracts_ui_structure() -> None:
    prompt = VLMPrompts.frame_analysis_build(1.0)
    assert "ui_elements" in prompt
    assert "screen_name" in prompt
    assert "design_notes" in prompt


def test_derive_user_flow_links_screens() -> None:
    scenes = [
        _scene(t0=0.0, screen_name="Home", ui_action="click"),
        _scene(t0=1.0, screen_name="Settings", ui_action="click"),
        _scene(t0=2.0, screen_name="Settings"),
        _scene(t0=3.0, screen_name="Profile", ui_action="submit"),
    ]
    flow = derive_user_flow(scenes)
    assert [(s.from_screen, s.to_screen) for s in flow] == [
        ("Home", "Settings"),
        ("Settings", "Profile"),
    ]
    assert flow[0].n == 1 and flow[1].n == 2


# --------------------------------------------------------------------------- #
# B. Build context assembly
# --------------------------------------------------------------------------- #


def test_build_context_aggregates_screens_and_components() -> None:
    scenes = [
        _scene(
            t0=0.0,
            screen_name="Settings",
            design_notes="dark background, sans-serif",
            ui_elements=[UiElement(kind="toggle", label="Dark mode", state="active")],
        ),
        _scene(
            t0=1.0,
            screen_name="Settings",
            ui_elements=[UiElement(kind="button", label="Save")],
        ),
    ]
    classification = Classification(label=ClassificationLabel.FEATURE, confidence=0.7)
    bc = build_build_context(scenes, classification, [])
    assert bc is not None
    assert {s.name for s in bc.screens} == {"Settings"}
    labels = {c.label for c in bc.components}
    assert {"Dark mode", "Save"} <= labels
    assert bc.is_greenfield is True
    assert "dark background, sans-serif" in bc.design_notes


def test_build_context_none_for_pure_bug() -> None:
    scenes = [_scene(ocr_text="TypeError", is_error_state=True)]
    classification = Classification(label=ClassificationLabel.BUG, confidence=0.9)
    assert build_build_context(scenes, classification, []) is None


def test_build_context_uses_candidates_as_targets() -> None:
    scenes = [_scene(screen_name="Settings")]
    classification = Classification(label=ClassificationLabel.FEATURE, confidence=0.7)
    candidates = [
        CodeCandidate(file="app/settings.py", line=10, match_reason="definition", confidence=0.8)
    ]
    bc = build_build_context(scenes, classification, candidates)
    assert bc is not None
    assert bc.is_greenfield is False
    assert "app/settings.py" in bc.target_locations


# --------------------------------------------------------------------------- #
# E. Grounding: definition boost, vendored skip, multi-language
# --------------------------------------------------------------------------- #


def test_grounding_prefers_definitions_and_skips_vendored(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "theme.ts").write_text(
        "export function toggleTheme() { return 1; }\n", encoding="utf-8"
    )
    (tmp_path / "app" / "notes.ts").write_text(
        "// toggleTheme is mentioned here in a comment\n", encoding="utf-8"
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.ts").write_text(
        "export function toggleTheme() {}\n", encoding="utf-8"
    )

    candidates = locate_in_code(tmp_path, ["toggleTheme"], max_results=5)
    files = [c.file.replace("\\", "/") for c in candidates]
    assert any("app/theme.ts" in f for f in files)
    assert not any("node_modules" in f for f in files)
    # The definition outranks the comment mention.
    top = candidates[0]
    assert top.match_reason == "definition"


# --------------------------------------------------------------------------- #
# H. Per-field confidence + actionability
# --------------------------------------------------------------------------- #


def _bundle(**kw: object) -> ContextBundle:
    base: dict[str, object] = {
        "id": "b1",
        "source_video": "v.mp4",
        "duration_s": 1.0,
        "classification": Classification(label=ClassificationLabel.BUG, confidence=0.9),
        "reproducibility": Reproducibility.SHOWN_ONCE,
        "title": "t",
        "severity": Severity.HIGH,
        "priority": Priority.P1,
        "suspected_component": "unknown",
        "environment": {},
        "preconditions": "",
        "repro_steps": [],
        "expected_behavior": "",
        "actual_behavior": "",
    }
    base.update(kw)
    # repro_steps must be non-empty per schema; supply a default if not overridden.
    if not base.get("repro_steps"):
        from framesleuth.schemas import ReproStep

        base["repro_steps"] = [
            ReproStep(n=1, t=0.0, action="open", evidence=["frame:0"], confidence=0.5)
        ]
    return ContextBundle(**base)  # type: ignore[arg-type]


def test_field_confidence_reflects_evidence() -> None:
    from framesleuth.schemas import ErrorEvidenceItem

    bundle = _bundle(error_evidence=[ErrorEvidenceItem(t=0.0, source="console", text="boom")])
    conf = compute_field_confidence(bundle)
    assert conf["title"] > 0.8  # anchored to error text
    assert "repro_steps" in conf


def test_confidence_corroboration_boosts_agreeing_signals() -> None:
    """An error that also grounds to code lifts the title + candidate confidence."""
    from framesleuth.schemas import ErrorEvidenceItem

    error = [ErrorEvidenceItem(t=0.0, source="console", text="boom")]
    base = compute_field_confidence(_bundle(error_evidence=error))
    corroborated = compute_field_confidence(
        _bundle(
            error_evidence=error,
            code_candidates=[
                CodeCandidate(file="a.py", line=1, match_reason="definition", confidence=0.6)
            ],
        )
    )
    assert corroborated["title"] > base["title"]
    assert corroborated["code_candidates"] > 0.6  # boosted above the raw mean


def test_actionability_insufficient_to_implement_without_build_context() -> None:
    bundle = _bundle(
        classification=Classification(label=ClassificationLabel.FEATURE, confidence=0.7),
        action="implement",
        build_context=None,
    )
    assert assess_actionability(bundle) == "insufficient"


def test_actionability_ready_to_fix_with_error_and_candidate() -> None:
    from framesleuth.schemas import ErrorEvidenceItem

    bundle = _bundle(
        action="fix",
        error_evidence=[ErrorEvidenceItem(t=0.0, source="console", text="boom")],
        code_candidates=[
            CodeCandidate(file="a.py", line=1, match_reason="definition", confidence=0.7)
        ],
    )
    assert assess_actionability(bundle) == "ready"


def test_degraded_quality_caps_actionability() -> None:
    bundle = _bundle(
        action="explain",
        analysis_quality=AnalysisQuality(level="degraded"),
    )
    assert assess_actionability(bundle) in {"thin", "insufficient"}


# --------------------------------------------------------------------------- #
# J. Build context renders into the action prompt
# --------------------------------------------------------------------------- #


def test_fix_prompt_renders_build_context() -> None:
    build_context = {
        "screens": [{"name": "Settings", "summary": "config screen", "components": ["Dark mode"]}],
        "components": [{"kind": "toggle", "label": "Dark mode", "states": ["active"]}],
        "user_flow": [{"n": 1, "from_screen": "Home", "action": "click", "to_screen": "Settings"}],
        "design_notes": ["dark background"],
        "target_locations": ["app/settings.py"],
        "is_greenfield": False,
    }
    prompt = FixPrompts.fix_from_video(
        title="Add dark mode",
        severity="medium",
        component="settings",
        environment={},
        repro_steps=[],
        expected="",
        actual="",
        errors=[],
        candidates=[],
        build_context=build_context,
        user_request="add a dark mode toggle",
    )
    assert "Build context (what to build)" in prompt
    assert "Dark mode" in prompt
    assert "Home" in prompt and "Settings" in prompt
    assert "app/settings.py" in prompt
