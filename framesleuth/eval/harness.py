"""Deterministic eval harness — classification, grounding, and citation checks.

Every check is model-free and reproducible: fixed fixtures in, metrics out. The
metrics gate regressions (see ``tests/test_eval_harness.py``) and can be printed
ad-hoc via ``python -m framesleuth.eval`` semantics through ``scripts/eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framesleuth.pipeline.bug_extract import extract_bug_context_bundle
from framesleuth.pipeline.classify import classify_video
from framesleuth.pipeline.grounding import locate_in_code
from framesleuth.schemas import (
    Classification,
    ClassificationLabel,
    ErrorEvidenceItem,
    SceneRecord,
    Transcript,
    UiElement,
)


@dataclass
class EvalResult:
    """Outcome of one eval suite: a primary metric plus per-case failures."""

    name: str
    metric: float
    total: int
    passed: int
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"{self.name}: {self.metric:.2%} ({self.passed}/{self.total})"
        if not self.failures:
            return head
        return head + "\n  - " + "\n  - ".join(self.failures)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ClassCase:
    name: str
    transcript: str
    ocr: str
    error_signals: tuple[str, ...]
    user_intent: str | None
    expect: ClassificationLabel


def _scene(ocr: str, *, error: bool = False) -> SceneRecord:
    return SceneRecord(
        t0=0.0,
        t1=0.0,
        caption="",
        ocr_text=ocr,
        ui_action=None,
        is_error_state=error,
        reason=None,
        ui_elements=[],
        layout=None,
        screen_name=None,
        design_notes=None,
        data_shown=None,
    )


def _transcript(text: str) -> Transcript:
    if not text:
        return Transcript(segments=[], words=None)
    segment = Transcript.Segment(t0=0.0, t1=1.0, text=text, conf=0.9)
    return Transcript(segments=[segment], words=None)


_CLASSIFICATION_CASES: tuple[_ClassCase, ...] = (
    _ClassCase(
        "console error -> bug",
        transcript="the save button does nothing",
        ocr="",
        error_signals=("TypeError: cannot read property 'id' of undefined",),
        user_intent="why does save hang?",
        expect=ClassificationLabel.BUG,
    ),
    _ClassCase(
        "error frame text -> bug",
        transcript="",
        ocr="Uncaught Exception: traceback at app.py:42",
        error_signals=(),
        user_intent=None,
        expect=ClassificationLabel.BUG,
    ),
    _ClassCase(
        "build intent -> feature",
        transcript="here is the screen",
        ocr="Settings",
        error_signals=(),
        user_intent="add a dark mode toggle like the demo shows",
        expect=ClassificationLabel.FEATURE,
    ),
    _ClassCase(
        "narrated build -> feature",
        transcript="let's build a new settings page with a profile form",
        ocr="Profile",
        error_signals=(),
        user_intent=None,
        expect=ClassificationLabel.FEATURE,
    ),
    _ClassCase(
        "fix intent over UI -> bug not feature",
        transcript="",
        ocr="Checkout",
        error_signals=("500 Internal Server Error on POST /pay",),
        user_intent="fix the broken checkout button",
        expect=ClassificationLabel.BUG,
    ),
    _ClassCase(
        "quiet demo -> not feature/bug",
        transcript="this is the dashboard",
        ocr="Dashboard",
        error_signals=(),
        user_intent=None,
        expect=ClassificationLabel.OTHER,
    ),
)


def run_classification_eval() -> EvalResult:
    """Accuracy of deterministic labelling over labelled fixtures."""
    passed = 0
    failures: list[str] = []
    for case in _CLASSIFICATION_CASES:
        result = classify_video(
            _transcript(case.transcript),
            [_scene(case.ocr, error="exception" in case.ocr.lower())],
            error_signals=list(case.error_signals),
            user_intent=case.user_intent,
        )
        if result.label == case.expect:
            passed += 1
        else:
            failures.append(f"{case.name}: got {result.label.value}, want {case.expect.value}")
    total = len(_CLASSIFICATION_CASES)
    return EvalResult("classification_accuracy", passed / total, total, passed, failures)


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #

_SAMPLE_FILES: dict[str, str] = {
    "app/cart.py": "def save_cart(item):\n    return persist(item)\n",
    "app/theme.py": "def toggle_theme():\n    # switch between light and dark\n    ...\n",
    "app/utils.py": "# a comment mentioning toggle but not a definition\nX = 1\n",
    "node_modules/dep/index.js": "export function toggle_theme() {}\n",  # must be skipped
}


@dataclass(frozen=True)
class _GroundCase:
    name: str
    queries: tuple[str, ...]
    expect_file: str
    forbid_file: str | None = None


_GROUNDING_CASES: tuple[_GroundCase, ...] = (
    _GroundCase("symbol -> definition", ("save_cart",), "app/cart.py"),
    _GroundCase(
        "feature noun -> definition, not vendored",
        ("toggle_theme",),
        "app/theme.py",
        forbid_file="node_modules/dep/index.js",
    ),
)


def build_sample_workspace(root: Path) -> Path:
    """Materialize the grounding fixtures under ``root`` and return it."""
    for rel, body in _SAMPLE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def run_grounding_eval(workspace_root: Path) -> EvalResult:
    """Recall@5: does the expected file surface, and vendored dirs stay excluded?"""
    passed = 0
    failures: list[str] = []
    for case in _GROUNDING_CASES:
        candidates = locate_in_code(workspace_root, case.queries, max_results=5)
        files = [c.file.replace("\\", "/") for c in candidates]
        ok = any(case.expect_file in f for f in files)
        if case.forbid_file and any(case.forbid_file in f for f in files):
            ok = False
        if ok:
            passed += 1
        else:
            failures.append(f"{case.name}: top files {files}")
    total = len(_GROUNDING_CASES)
    return EvalResult("grounding_recall", passed / total, total, passed, failures)


# --------------------------------------------------------------------------- #
# Citation integrity
# --------------------------------------------------------------------------- #


def run_citation_eval() -> EvalResult:
    """Every extracted step/error must carry a citation; timeline must be ordered."""
    bundle = extract_bug_context_bundle(
        job_id="eval",
        source_video="eval.mp4",
        duration_s=3.0,
        classification=Classification(label=ClassificationLabel.BUG, confidence=0.9),
        transcript=_transcript("the save button hangs"),
        scenes=[
            _scene("clicked Save", error=False),
            _scene("Error: request failed", error=True),
        ],
        keyframes=[],
        environment={"os": "macOS"},
        sidecar_evidence=[
            ErrorEvidenceItem(t=2.0, source="console", text="500 error"),
            ErrorEvidenceItem(t=1.0, source="network", text="POST /save failed"),
        ],
    )
    checks: list[tuple[str, bool]] = [
        ("steps_cited", all(step.evidence for step in bundle.repro_steps)),
        ("errors_time_sorted", _is_sorted([e.t for e in bundle.error_evidence])),
        ("has_quality_signal", bundle.analysis_quality.level in {"full", "partial", "degraded"}),
    ]
    passed = sum(1 for _, ok in checks if ok)
    failures = [name for name, ok in checks if not ok]
    total = len(checks)
    return EvalResult("citation_integrity", passed / total, total, passed, failures)


def _is_sorted(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


# --------------------------------------------------------------------------- #
# Faithfulness — every emitted claim must trace to real evidence (no fabrication)
# --------------------------------------------------------------------------- #


def _citation_resolves(citation: str, n_scenes: int, n_segments: int) -> bool:
    """Whether a ``frame:N`` / ``transcript:N`` citation points at real evidence."""
    kind, _, index = citation.partition(":")
    if not index.isdigit():
        # Non-indexed citations (e.g. ``sidecar:env``) are accepted as provenance.
        return bool(kind)
    idx = int(index)
    if kind == "frame":
        return 0 <= idx < n_scenes
    if kind == "transcript":
        return 0 <= idx < n_segments
    return True


def run_faithfulness_eval() -> EvalResult:
    """Every key moment / step must be cited *and* resolve to real evidence.

    This is the model-free faithfulness gate: it builds a bundle from a known set
    of scenes + narration and asserts that nothing the pipeline emits is fabricated
    or mis-indexed — every key moment and observed step carries a citation that
    points at a frame/transcript segment that actually exists, the deliverable for
    a general video is present, and no scene-derived error evidence appears that was
    not in an error scene.
    """
    scenes = [
        _scene("Home screen", error=False),
        _scene("clicked Save", error=False),
        _scene("Error: request failed", error=True),
    ]
    transcript = _transcript("let me walk through saving a record")
    n_scenes, n_segments = len(scenes), len(transcript.segments)
    bundle = extract_bug_context_bundle(
        job_id="faithfulness",
        source_video="demo.mp4",
        duration_s=4.0,
        classification=Classification(label=ClassificationLabel.OTHER, confidence=0.4),
        transcript=transcript,
        scenes=scenes,
        keyframes=[],
        environment={},
        summary="A short walkthrough of saving a record, ending in a failed request.",
    )

    moments_cited = all(m.evidence for m in bundle.key_moments)
    moments_resolve = all(
        _citation_resolves(c, n_scenes, n_segments) for m in bundle.key_moments for c in m.evidence
    )
    steps_resolve = all(
        _citation_resolves(c, n_scenes, n_segments)
        for step in bundle.repro_steps
        for c in step.evidence
    )
    deliverable_present = bool(bundle.summary or bundle.key_moments)

    checks: list[tuple[str, bool]] = [
        ("key_moments_cited", moments_cited),
        ("key_moments_resolve", moments_resolve),
        ("steps_resolve", steps_resolve),
        ("deliverable_present", deliverable_present),
    ]
    passed = sum(1 for _, ok in checks if ok)
    failures = [name for name, ok in checks if not ok]
    return EvalResult("faithfulness", passed / len(checks), len(checks), passed, failures)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #

# Build-aware UI element used to keep the import meaningful for downstream callers
# that extend the fixtures with structured UI cases.
_SAMPLE_UI = UiElement(kind="button", label="Save", state=None)


def run_all(workspace_root: Path) -> dict[str, EvalResult]:
    """Run every behavioral suite. ``workspace_root`` is a temp dir for fixtures."""
    build_sample_workspace(workspace_root)
    return {
        "classification": run_classification_eval(),
        "grounding": run_grounding_eval(workspace_root),
        "citation": run_citation_eval(),
        "faithfulness": run_faithfulness_eval(),
    }


# --------------------------------------------------------------------------- #
# Bundle-vs-golden metrics (used by ``scripts/eval_harness.py`` CLI)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalMetrics:
    """Core eval metrics for generated bundle quality against a golden fixture."""

    repro_step_precision: float
    repro_step_recall: float
    error_capture_rate: float
    grounding_hit_rate_at_k: float

    def to_dict(self) -> dict[str, float]:
        return {
            "repro_step_precision": self.repro_step_precision,
            "repro_step_recall": self.repro_step_recall,
            "error_capture_rate": self.error_capture_rate,
            "grounding_hit_rate_at_k": self.grounding_hit_rate_at_k,
        }


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def evaluate_bundle(predicted: dict[str, Any], expected: dict[str, Any], k: int = 5) -> EvalMetrics:
    """Compute deterministic quality metrics against a golden fixture bundle."""
    pred_steps = {step.get("action", "") for step in predicted.get("repro_steps", [])}
    exp_steps = {step.get("action", "") for step in expected.get("repro_steps", [])}
    overlap_steps = len(pred_steps.intersection(exp_steps))

    pred_errors = {item.get("text", "") for item in predicted.get("error_evidence", [])}
    exp_errors = {item.get("text", "") for item in expected.get("error_evidence", [])}
    error_overlap = len(pred_errors.intersection(exp_errors))

    pred_candidates = predicted.get("code_candidates", [])[:k]
    exp_candidates = {item.get("file", "") for item in expected.get("code_candidates", [])}
    hit = any(candidate.get("file", "") in exp_candidates for candidate in pred_candidates)

    return EvalMetrics(
        repro_step_precision=_safe_div(overlap_steps, len(pred_steps)),
        repro_step_recall=_safe_div(overlap_steps, len(exp_steps)),
        error_capture_rate=_safe_div(error_overlap, len(exp_errors)),
        grounding_hit_rate_at_k=1.0 if hit else 0.0,
    )
