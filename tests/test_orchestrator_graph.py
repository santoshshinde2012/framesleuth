"""Tests for the end-to-end analysis orchestrator glue.

The heavy perception dependencies (ffmpeg/av) are stubbed so the stage wiring,
sidecar ingestion, redaction, persistence, and lifecycle-state transitions can be
exercised deterministically without model servers.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from framesleuth.clients.vlm import FrameAnalysisResponse
from framesleuth.config import Settings
from framesleuth.jobs.store import JobStore
from framesleuth.orchestrator import graph as graph_module
from framesleuth.orchestrator.graph import AnalysisOrchestrator
from framesleuth.schemas import JobState


class _FakeVLM:
    """Deterministic VLM stub returning an error-state frame with a secret."""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        self.calls += 1
        return FrameAnalysisResponse(
            caption="Save button shows an infinite spinner",
            ocr_text="TypeError: cannot read property 'id' token=ABCDEF1234567890",
            ui_action="click Save",
            is_error_state=True,
            reason="exception visible",
        )


class _ExplodingVLM:
    """VLM stub that always fails, to exercise graceful degradation."""

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        raise RuntimeError("vlm unreachable")


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        BUNDLE_DIR=tmp_path / "bundles",
        DATABASE_PATH=tmp_path / "jobs.db",
        CHROME_EXTENSION_ORIGIN="chrome-extension://test",
    )
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    return settings


def _stub_preprocess(monkeypatch: pytest.MonkeyPatch, duration_s: float) -> None:
    def _fake(video_path: Path, *, settings: Settings) -> SimpleNamespace:
        return SimpleNamespace(duration_s=duration_s, metadata={})

    monkeypatch.setattr(graph_module, "preprocess_video", _fake)


@pytest.fixture(autouse=True)
def _no_network_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub summary synthesis so orchestrator tests never hit a model server."""

    async def _fake_summarize(*_args: object, **_kwargs: object) -> str:
        return ""

    monkeypatch.setattr(graph_module, "summarize_recording", _fake_summarize)


async def _seed_job(store: JobStore, job_id: str) -> None:
    await store.initialize()
    await store.create_job(job_id, f"hash-{job_id}", "bug.webm")


@pytest.mark.asyncio
async def test_run_uses_vlm_when_frames_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When extracted frames exist, the VLM runs and OCR secrets are redacted."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=12.0)

    # Pre-place the keyframe images in the job's scratch dir, where the
    # orchestrator looks (frame extraction can't decode these stub bytes).
    frames_dir = settings.BUNDLE_DIR / "job-vlm" / ".work" / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(8):
        (frames_dir / f"{i}.png").write_bytes(b"\x89PNG")

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-vlm")
    vlm = _FakeVLM()
    orchestrator = AnalysisOrchestrator(settings, store, vlm)

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run("job-vlm", video_path, "bug.webm")

    assert vlm.calls >= 1
    job = await store.get_job("job-vlm")
    assert job is not None and job.state == JobState.DONE and job.progress_pct == 100
    assert "ABCDEF1234567890" not in bundle_path.read_text(encoding="utf-8")

    # The job's scratch dir is cleaned up, but the persisted artifacts survive.
    assert not (settings.BUNDLE_DIR / "job-vlm" / ".work").exists()

    # Analyzed keyframes are persisted next to the bundle so MCP/report can read them.
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["keyframe_refs"], "expected keyframe references"
    for ref in bundle["keyframe_refs"]:
        assert (bundle_path.parent / ref["file"]).exists(), f"missing {ref['file']}"
    # The fused timeline carries an ordered, cited event stream.
    timeline = json.loads((bundle_path.parent / "timeline.json").read_text(encoding="utf-8"))
    assert "events" in timeline
    assert any(e["citation"].startswith("frame:") for e in timeline["events"])
    # Transcript/timeline are advertised on the bundle.
    assert bundle["transcript_path"] == "transcript.json"
    assert bundle["timeline_path"] == "timeline.json"


def test_coverage_keyframes_spread_and_deltas(tmp_path: Path) -> None:
    """Coverage selection spans the whole clip and keeps high-delta frames."""
    from framesleuth.pipeline.preprocess import ExtractedFrame

    settings = _settings(tmp_path)
    store = JobStore(settings.DATABASE_PATH)
    orch = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    # All frames analyzed when within the cap.
    few = [ExtractedFrame(t=float(i), file=f"frames/{i}.png", change_score=0.0) for i in range(5)]
    assert len(orch._coverage_keyframes(few, max_keyframes=8)) == 5

    # Over the cap: bounded, spans first..last, includes the highest-delta frame.
    many = [
        ExtractedFrame(t=float(i), file=f"frames/{i}.png", change_score=(0.9 if i == 17 else 0.0))
        for i in range(20)
    ]
    kfs = orch._coverage_keyframes(many, max_keyframes=8)
    idxs = [k.index for k in kfs]
    assert len(kfs) == 8
    assert idxs[0] == 0 and idxs[-1] == 19  # spans the clip
    assert 17 in idxs  # the high-delta (error) frame survives
    assert idxs == sorted(idxs)  # time-ordered


def test_resample_times_dedupes_against_analyzed_frames(tmp_path: Path) -> None:
    """The agentic resample must not re-pick timestamps already analyzed.

    This is the guard that keeps the bounded resample loop from spinning without
    new evidence: offsets that land within 0.1s of an existing frame are dropped.
    """
    settings = _settings(tmp_path)
    store = JobStore(settings.DATABASE_PATH)
    orch = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    # Error window at 5.0s; existing frames already cover both offsets (~4.5/5.5).
    covered = orch._resample_times([5.0], duration_s=10.0, existing=[4.5, 5.5])
    assert covered == []  # nothing new -> the loop breaks instead of looping

    # With no nearby coverage, it proposes the just-before/just-after timestamps.
    fresh = orch._resample_times([5.0], duration_s=10.0, existing=[])
    assert fresh == [4.5, 5.5]


@pytest.mark.asyncio
async def test_run_records_user_intent_on_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user_intent passed to run() is persisted on the bundle for the fix prompt."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-intent")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    sidecars = [{"t": 1.0, "source": "console", "text": "TypeError in saveProfile"}]
    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run(
        "job-intent",
        video_path,
        "bug.webm",
        sidecars=sidecars,
        user_intent="Add a confirmation dialog before saving",
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["user_intent"] == "Add a confirmation dialog before saving"


@pytest.mark.asyncio
async def test_run_marks_job_failed_on_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-degradable error transitions the job to FAILED and re-raises."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    def _boom(**_kwargs: object) -> None:
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(graph_module, "extract_bug_context_bundle", _boom)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-fail")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    with pytest.raises(RuntimeError, match="synthesis exploded"):
        await orchestrator.run("job-fail", video_path, "bug.webm")

    job = await store.get_job("job-fail")
    assert job is not None and job.state == JobState.FAILED
    assert job.error_json is not None and job.error_json["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_run_degrades_to_sidecars_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no frames and no VLM, a bundle is still built from sidecars."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-sc")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    sidecars = [
        {
            "t": 0.1,
            "source": "env",
            "ua": "Mozilla/5.0 Chrome/137",
            "url": "https://app.test/profile/settings",
        },
        {"t": 8.2, "source": "click", "selector": "button", "text": "Edit profile"},
        {"t": 27.4, "source": "click", "selector": "button", "text": "Save"},
        {"t": 28.1, "source": "network", "method": "POST", "url": "/api/profile", "status": 500},
        {
            "t": 28.3,
            "source": "console",
            "text": "TypeError: cannot read property 'id' of undefined",
        },
        {"t": 12.0, "source": "console", "text": "password=hunter2 leaked"},
    ]

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run("job-sc", video_path, "bug.webm", sidecars=sidecars)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    # Classified as a bug from the console/network failures, not visual analysis.
    assert bundle["classification"]["label"] == "bug"
    # Error evidence captured from both console and network.
    sources = {e["source"] for e in bundle["error_evidence"]}
    assert "console" in sources and "network" in sources
    # Repro steps derived from clicks.
    actions = [s["action"] for s in bundle["repro_steps"]]
    assert any("Save" in a for a in actions)
    # Environment derived from the env snapshot.
    assert bundle["environment"]["browser"] == "Chrome"
    # Title/summary prefer a real error, not the redacted secret line.
    assert "500" in bundle["title"] or "TypeError" in bundle["title"]
    assert "[REDACTED]" not in bundle["title"]
    # Secret in a console line was redacted before persistence.
    assert "hunter2" not in bundle_path.read_text(encoding="utf-8")
    assert bundle["redactions"]

    # Side artifacts are written next to the bundle.
    bundle_dir = bundle_path.parent
    for name in ("transcript.json", "timeline.json", "sidecars.json", "metrics.json"):
        assert (bundle_dir / name).exists()
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "understand" in metrics["degraded"]
    # Real sidecar evidence survived a degraded visual stage -> partial, not degraded.
    assert bundle["analysis_quality"]["level"] == "partial"
    assert "understand" in bundle["analysis_quality"]["degraded_stages"]

    # A bug auto-picks the 'fix' action and gets a non-empty next-step menu.
    assert bundle["action"] == "fix"
    assert any(s["action"] == "propose_fix" for s in bundle["suggested_actions"])


@pytest.mark.asyncio
async def test_run_records_explicit_action_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit action is stored on the bundle and overrides the auto-pick."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-act")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    sidecars = [{"t": 1.0, "source": "console", "text": "TypeError boom"}]
    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run(
        "job-act", video_path, "bug.webm", sidecars=sidecars, action="explain"
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["action"] == "explain"  # not the bug auto-pick 'fix'
    assert bundle["action_prompt"] is None


class _GeneralVLM:
    """VLM stub for a general (non-bug) recording: a plain scene, no error."""

    async def analyze_frame(
        self,
        image_path: str,
        timestamp: float,
        prompt_override: str | None = None,
        *,
        max_tokens: int | None = None,
        send_jpeg: bool | None = None,
    ) -> FrameAnalysisResponse:
        return FrameAnalysisResponse(
            caption="A presenter demonstrates the analytics dashboard",
            ocr_text="Revenue overview",
            ui_action=None,
            is_error_state=False,
            reason=None,
        )


@pytest.mark.asyncio
async def test_run_general_video_is_summary_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-bug video yields summary + key moments, suppresses bug fields, auto-summarizes."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=10.0)

    # A real summary comes back (overrides the autouse empty-summary stub) so we can
    # assert it leads the bundle.
    async def _fake_summary(*_a: object, **_k: object) -> str:
        return "A walkthrough of the analytics dashboard and its revenue charts."

    monkeypatch.setattr(graph_module, "summarize_recording", _fake_summary)

    frames_dir = settings.BUNDLE_DIR / "job-gen" / ".work" / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(8):
        (frames_dir / f"{i}.png").write_bytes(b"\x89PNG")

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-gen")
    orchestrator = AnalysisOrchestrator(settings, store, _GeneralVLM())

    video_path = tmp_path / "demo.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run("job-gen", video_path, "demo.webm")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    # Not a bug — bug-shaped fields are suppressed.
    assert bundle["classification"]["label"] != "bug"
    assert bundle["severity"] is None
    assert bundle["expected_behavior"] is None
    assert bundle["actual_behavior"] is None
    # The summary + key moments are the deliverable.
    assert bundle["summary"].startswith("A walkthrough of the analytics dashboard")
    assert bundle["key_moments"], "expected key moments for a general video"
    # The action auto-picks 'summarize' and the menu offers it.
    assert bundle["action"] == "summarize"
    assert any(s["action"] == "summarize" for s in bundle["suggested_actions"])
    # Per-stage timings are surfaced on the bundle for observability.
    assert bundle["stage_timings"], "expected stage timings on the bundle"


@pytest.mark.asyncio
async def test_run_cancelled_job_transitions_to_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel requested before/at a checkpoint aborts the run into CANCELLED state."""
    from framesleuth.errors import JobCancelledError

    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-cancel")
    await store.request_cancel("job-cancel")  # cancel before the first checkpoint
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    with pytest.raises(JobCancelledError):
        await orchestrator.run("job-cancel", video_path, "bug.webm")

    job = await store.get_job("job-cancel")
    assert job is not None and job.state == JobState.CANCELLED


@pytest.mark.asyncio
async def test_run_records_custom_action_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom action_prompt is labeled 'custom' and persisted for re-rendering."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-cact")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run(
        "job-cact",
        video_path,
        "bug.webm",
        sidecars=[{"t": 1.0, "source": "console", "text": "boom"}],
        action="fix",
        action_prompt="Only write a one-paragraph explanation.",
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["action"] == "custom"
    assert bundle["action_prompt"] == "Only write a one-paragraph explanation."


@pytest.mark.asyncio
async def test_run_flags_preprocess_degraded_on_unknown_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero/unknown duration must surface as a degraded stage, not silent success."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-dur")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run("job-dur", video_path, "bug.webm")

    metrics = json.loads((bundle_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert "preprocess" in metrics["degraded"]

    # The degraded state must propagate to the bundle so downstream agents see it.
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["analysis_quality"]["level"] == "degraded"
    assert "preprocess" in bundle["analysis_quality"]["degraded_stages"]


@pytest.mark.asyncio
async def test_run_does_not_flag_preprocess_when_duration_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known duration leaves preprocess off the degraded list."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=5.0)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-okdur")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run("job-okdur", video_path, "bug.webm")

    metrics = json.loads((bundle_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert "preprocess" not in metrics["degraded"]


@pytest.mark.asyncio
async def test_run_records_skill_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chosen skill label and generated summary are persisted on the bundle."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=3.0)

    captured: dict[str, object] = {}

    async def _fake_summarize(*_a: object, **kwargs: object) -> str:
        captured["system_prompt"] = kwargs["system_prompt"]
        return "A concise narrative of the recording."

    monkeypatch.setattr(graph_module, "summarize_recording", _fake_summarize)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-skill")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    sidecars = [{"t": 1.0, "source": "console", "text": "TypeError boom"}]
    bundle_path = await orchestrator.run(
        "job-skill", video_path, "bug.webm", sidecars=sidecars, skill="bug_report"
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["skill"] == "bug_report"
    assert bundle["summary"] == "A concise narrative of the recording."
    # The bug_report skill's system prompt was passed to the summarizer.
    assert "bug report" in str(captured["system_prompt"]).lower()


@pytest.mark.asyncio
async def test_run_custom_system_prompt_overrides_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom system_prompt is labeled 'custom' and used verbatim."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=3.0)

    captured: dict[str, object] = {}

    async def _fake_summarize(*_a: object, **kwargs: object) -> str:
        captured["system_prompt"] = kwargs["system_prompt"]
        return "custom output"

    monkeypatch.setattr(graph_module, "summarize_recording", _fake_summarize)

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-custom")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run(
        "job-custom",
        video_path,
        "bug.webm",
        sidecars=[{"t": 1.0, "source": "console", "text": "boom"}],
        skill="summary",
        system_prompt="Answer ONLY in haiku.",
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["skill"] == "custom"
    assert captured["system_prompt"] == "Answer ONLY in haiku."


@pytest.mark.asyncio
async def test_run_grounds_against_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a workspace is provided, error text is grounded to code locations."""
    settings = _settings(tmp_path)
    _stub_preprocess(monkeypatch, duration_s=0.0)

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "profileController.py").write_text(
        "def saveProfile():\n    return user.id\n", encoding="utf-8"
    )

    store = JobStore(settings.DATABASE_PATH)
    await _seed_job(store, "job-gr")
    orchestrator = AnalysisOrchestrator(settings, store, _ExplodingVLM())

    sidecars = [{"t": 1.0, "source": "console", "text": "error in saveProfile"}]
    video_path = tmp_path / "bug.webm"
    video_path.write_bytes(b"fake")
    bundle_path = await orchestrator.run(
        "job-gr", video_path, "bug.webm", sidecars=sidecars, workspace_root=workspace
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["code_candidates"], "expected grounded candidates"
    assert bundle["code_candidates"][0]["file"] == "profileController.py"
