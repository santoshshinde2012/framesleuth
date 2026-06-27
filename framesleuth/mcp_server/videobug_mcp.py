"""videobug MCP server and read-only report helpers (VS Code / Surface A).

The pure helper functions hold the logic (and are unit-tested without the MCP
SDK). ``build_server`` wires them into a `FastMCP` server exposing tools,
resources, and the ``fix-from-video`` prompt; the ``mcp`` SDK is imported lazily
so importing this module never requires it.

All tools are **read-only** over the workspace and bundle directory: the MCP
trust boundary means edits happen only through the editor's reviewed apply-edit
flow, never here.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from framesleuth.prompts import FixPrompts

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.fastmcp import FastMCP

_FIX_PROMPT_FIELDS = {
    "title",
    "severity",
    "suspected_component",
    "environment",
    "repro_steps",
    "expected_behavior",
    "actual_behavior",
    "error_evidence",
    "code_candidates",
    "keyframe_refs",
    "user_intent",
    "analysis_quality",
    "classification",
    "action",
    "action_prompt",
}

# Fields kept in the ``slim`` view of a report — the action-relevant subset that
# fits a small agent context window (drops keyframes, transcript, redactions…).
_SLIM_FIELDS = {
    "id",
    "title",
    "classification",
    "severity",
    "suspected_component",
    "environment",
    "analysis_quality",
    "action",
    "suggested_actions",
    "summary",
    "expected_behavior",
    "actual_behavior",
    "repro_steps",
    "error_evidence",
    "code_candidates",
    "user_intent",
}


def slim_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project a report down to the action-relevant fields (the ``slim`` view)."""
    return {key: report[key] for key in _SLIM_FIELDS if key in report}


def report_dir(bundle_root: Path, report_id: str) -> Path:
    """Resolve a report's directory, rejecting traversal in ``report_id``.

    ``report_id`` is agent/client-supplied, so a value like ``../../etc`` must
    not escape the bundle root. Every filesystem access keyed by a report id
    goes through here so the trust boundary is enforced in one place.
    """
    base = bundle_root.resolve()
    target = (base / report_id).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"invalid report id: {report_id!r}")
    return target


def list_reports(bundle_root: Path) -> list[str]:
    """List available report ids under bundle root."""
    if not bundle_root.exists():
        return []
    return sorted([path.name for path in bundle_root.iterdir() if path.is_dir()])


def get_report(bundle_root: Path, job_id: str) -> dict[str, Any]:
    """Return parsed report bundle for a job."""
    bundle_path = report_dir(bundle_root, job_id) / "bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found for {job_id}")
    report: dict[str, Any] = json.loads(bundle_path.read_text(encoding="utf-8"))
    return report


def get_keyframe(bundle_root: Path, job_id: str, relative_file: str) -> bytes:
    """Return keyframe bytes with path traversal protection."""
    root = report_dir(bundle_root, job_id)
    target = (root / relative_file).resolve()
    if root not in target.parents and target != root:
        raise ValueError("invalid keyframe path")
    return target.read_bytes()


def build_fix_prompt_input(report: dict[str, Any]) -> dict[str, Any]:
    """Build strict evidence-only payload for fix generation prompts."""
    return {key: report[key] for key in _FIX_PROMPT_FIELDS if key in report}


def render_fix_prompt(report: dict[str, Any]) -> str:
    """Render the grounded fix prompt from a bug bundle (evidence-only).

    The task block is resolved from the report's stored ``action`` (or custom
    ``action_prompt``), falling back to an auto-pick from the classification so
    older bundles without an action still render sensibly.
    """
    from framesleuth.actions import resolve_action_task

    data = build_fix_prompt_input(report)
    errors = [str(item.get("text", "")) for item in data.get("error_evidence", [])]
    keyframes = data.get("keyframe_refs", [])
    keyframe_path = keyframes[0].get("file") if keyframes else None
    label = (data.get("classification") or {}).get("label")
    task = resolve_action_task(data.get("action"), data.get("action_prompt"), label)
    return FixPrompts.fix_from_video(
        title=data.get("title", "Unknown bug"),
        severity=data.get("severity", "unknown"),
        component=data.get("suspected_component", "unknown"),
        environment=data.get("environment", {}),
        repro_steps=data.get("repro_steps", []),
        expected=data.get("expected_behavior", "n/a"),
        actual=data.get("actual_behavior", "n/a"),
        errors=errors,
        candidates=data.get("code_candidates", []),
        keyframe_path=keyframe_path,
        user_request=data.get("user_intent"),
        quality=data.get("analysis_quality"),
        task=task,
    )


def build_server(bundle_root: Path | None = None) -> FastMCP:  # noqa: C901
    """Construct the read-only ``videobug`` FastMCP server.

    Args:
        bundle_root: Directory holding ``{id}/bundle.json`` reports. Defaults to
            the configured ``BUNDLE_DIR``.

    Returns:
        A configured ``FastMCP`` instance (transport chosen by the caller).
    """
    from mcp.server.fastmcp import FastMCP, Image  # lazy: SDK only needed to serve

    # FastMCP evaluates tool annotations against this module's globals (PEP 563
    # stringized annotations + eval_str). ``get_keyframe_image`` returns ``Image``,
    # so expose the lazily-imported symbol at module scope before registering it.
    globals()["Image"] = Image

    from framesleuth.config import get_settings
    from framesleuth.jobs.store import JobStore

    settings = get_settings()
    root = bundle_root or settings.BUNDLE_DIR
    mcp = FastMCP("videobug")

    # One store for the server's lifetime; schema is created on first use rather
    # than rebuilt on every analyze_video call.
    store = JobStore(settings.DATABASE_PATH)
    _store_ready = {"init": False}

    async def _ensure_store() -> JobStore:
        if not _store_ready["init"]:
            await store.initialize()
            _store_ready["init"] = True
        return store

    @mcp.tool()
    async def analyze_video(
        path: str,
        repo_root: str | None = None,
        intent: str | None = None,
        skill: str | None = None,
        system_prompt: str | None = None,
        action: str | None = None,
        action_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a screen-recording video and return the new report id.

        Args:
            path: Path to the video file (.mp4/.webm/.mkv/.mov/.avi).
            repo_root: Repo to ground error text against (pass the open workspace).
            intent: The user's request to act on, e.g. "fix the save button that
                hangs" or "add a dark-mode toggle like the demo shows". It is
                recorded on the report and shapes the generated fix prompt so the
                calling agent does what the user actually asked.
            skill: Built-in summary style — one of the names from ``list_skills``
                (e.g. "summary", "bug_report", "tutorial", "action_items").
                Defaults to "summary".
            system_prompt: A fully custom system prompt for the summary; overrides
                ``skill`` when provided.
            action: Built-in action mode shaping the fix-prompt — one of the names
                from ``list_actions`` (e.g. "fix", "explain", "triage", "test",
                "report", "reproduce"). Auto-picked from the classification when
                omitted.
            action_prompt: A fully custom action task; overrides ``action``.

        Returns the report ``id``, the summary/fix-prompt resource URIs, the
        resolved ``action``, and the derived ``suggested_actions`` menu.
        """
        import uuid

        from framesleuth.clients.vlm import VLMClient
        from framesleuth.orchestrator.graph import AnalysisOrchestrator

        await _ensure_store()
        job_id = str(uuid.uuid4())
        video_path = Path(path)
        await store.create_job(job_id, job_id, video_path.name)
        # Build the VLM client from settings (same tuning as the HTTP service) and
        # release its pooled HTTP session when the analysis finishes.
        async with VLMClient.from_settings(settings) as vlm:
            orchestrator = AnalysisOrchestrator(settings, store, vlm)
            await orchestrator.run(
                job_id,
                video_path,
                video_path.name,
                workspace_root=Path(repo_root) if repo_root else None,
                user_intent=intent,
                skill=skill,
                system_prompt=system_prompt,
                action=action,
                action_prompt=action_prompt,
            )
        report = get_report(root, job_id)
        return {
            "id": job_id,
            "action": report.get("action"),
            "suggested_actions": report.get("suggested_actions", []),
            "summary_resource": f"videobug://report/{job_id}/summary",
            "fix_prompt_resource": f"videobug://report/{job_id}/fix-prompt",
        }

    @mcp.tool()
    def list_skills() -> dict[str, Any]:
        """List built-in summary skills (names + descriptions) for ``analyze_video``."""
        from framesleuth.skills import DEFAULT_SKILL
        from framesleuth.skills import list_skills as _list

        return {"default": DEFAULT_SKILL, "skills": _list()}

    @mcp.tool()
    def list_actions() -> dict[str, Any]:
        """List built-in action modes (names + descriptions) for ``analyze_video``."""
        from framesleuth.actions import DEFAULT_ACTION
        from framesleuth.actions import list_actions as _list

        return {"default": DEFAULT_ACTION, "auto": True, "actions": _list()}

    @mcp.tool()
    def get_suggested_actions(report_id: str) -> list[dict[str, str]]:
        """Return the machine-readable next-step menu for a report.

        Each item is ``{action, label, rationale, ref}`` — present them to the
        user or auto-invoke the referenced resource/tool. Recomputed from the
        current bundle so it reflects the latest grounding/quality.
        """
        from framesleuth.actions import suggest_actions

        return suggest_actions(get_report(root, report_id))

    @mcp.tool()
    def render(report_id: str, format: str = "markdown") -> str:
        """Render a report as a shareable artifact.

        ``format`` is one of ``markdown``, ``issue`` (GitHub issue text), or
        ``test-plan``. Returns the rendered text.
        """
        from framesleuth.render import render as _render

        return _render(get_report(root, report_id), format)

    @mcp.tool()
    def list_bug_reports() -> list[str]:
        """List all available bug report ids."""
        return list_reports(root)

    @mcp.tool()
    def get_bug_report(report_id: str, view: str = "full") -> dict[str, Any]:
        """Return the Bug Context Bundle for a report id.

        ``view="full"`` (default) returns everything; ``view="slim"`` returns the
        action-relevant subset (classification, quality, steps, evidence,
        candidates, suggested actions) for agents on a small context window.
        """
        report = get_report(root, report_id)
        return slim_report(report) if view.strip().lower() == "slim" else report

    @mcp.tool()
    def get_repro_steps(report_id: str) -> list[dict[str, Any]]:
        """Return the numbered reproduction steps for a report."""
        return list(get_report(root, report_id).get("repro_steps", []))

    @mcp.tool()
    def get_error_evidence(report_id: str) -> list[dict[str, Any]]:
        """Return the timestamped error evidence for a report."""
        return list(get_report(root, report_id).get("error_evidence", []))

    @mcp.tool()
    def get_timeline(report_id: str) -> dict[str, Any]:
        """Return the merged event timeline for a report."""
        timeline_path = report_dir(root, report_id) / "timeline.json"
        if not timeline_path.exists():
            return {}
        loaded: dict[str, Any] = json.loads(timeline_path.read_text(encoding="utf-8"))
        return loaded

    @mcp.tool()
    def get_keyframe_image(report_id: str, index: int) -> Image:
        """Return a keyframe image for a report by its index."""
        report = get_report(root, report_id)
        refs = report.get("keyframe_refs", [])
        match = next((r for r in refs if r.get("index") == index), None)
        if match is None:
            raise ValueError(f"keyframe {index} not found for {report_id}")
        data = get_keyframe(root, report_id, str(match["file"]))
        return Image(data=data, format="png")

    @mcp.tool()
    def get_video_gif(
        report_id: str,
        fps: float | None = None,
        width: float | None = None,
        start: float = 0.0,
        end: float | None = None,
    ) -> Image:
        """Render an animated GIF preview of the recording for a report.

        Useful for embedding a short looping preview of the bug in an issue, chat,
        or PR description. ``fps``/``width``/``start``/``end`` are optional and
        clamped to safe ranges; the GIF is cached on disk per parameter set.
        """
        from framesleuth.pipeline.gif import encode_gif, normalize_options

        bundle_dir = report_dir(root, report_id)
        source = next(
            (p for p in sorted(bundle_dir.glob("source.*")) if p.suffix.lower() != ".tmp"),
            None,
        )
        if source is None:
            raise FileNotFoundError(f"no source recording stored for {report_id}")
        options = normalize_options(
            fps=fps if fps is not None else settings.GIF_FPS,
            width=width if width is not None else settings.GIF_WIDTH,
            start=start,
            end=end,
            max_duration_s=settings.GIF_MAX_DURATION_S,
        )
        gif_path = bundle_dir / f"preview-{options.cache_key()}.gif"
        if not gif_path.exists() and encode_gif(source, gif_path, options=options) is None:
            raise RuntimeError(f"GIF encoding failed for {report_id}")
        return Image(data=gif_path.read_bytes(), format="gif")

    @mcp.tool()
    def locate_in_code(report_id: str, repo_root: str | None = None) -> list[dict[str, Any]]:
        """Return code candidates already grounded in the bundle, or re-ground now."""
        from framesleuth.pipeline.grounding import locate_in_code as ground

        report = get_report(root, report_id)
        existing = report.get("code_candidates", [])
        if existing:
            return list(existing)
        workspace = Path(repo_root) if repo_root else Path.cwd()
        queries = [str(item.get("text", "")) for item in report.get("error_evidence", [])]
        return [c.model_dump() for c in ground(workspace, queries)]

    @mcp.tool()
    async def render_html_video(
        html: str,
        format: str = "mp4",
        duration_s: float = 5.0,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ) -> str:
        """Render an HTML document (CSS / JS / canvas animation) to mp4/gif/webm.

        Use this to export a self-contained animated HTML page (e.g. one you just
        designed) as a shareable clip. Returns the absolute path to the encoded
        file, written under the bundle directory. Requires the optional
        ``render`` extra (Playwright) plus ``ffmpeg``.
        """
        from framesleuth.pipeline.html_render import RenderOptions, render_html

        options = RenderOptions.normalized(
            fmt=format, duration_s=duration_s, fps=fps, width=width, height=height
        )
        out_dir = root / "renders" / uuid.uuid4().hex
        path = await render_html(html, options, out_dir)
        return str(path)

    @mcp.resource("videobug://report/{report_id}/summary")
    def report_summary(report_id: str) -> str:
        """Concise human-readable summary of a report, with the next-step menu."""
        report = get_report(root, report_id)
        return json.dumps(
            {
                "title": report.get("title"),
                "severity": report.get("severity"),
                "classification": report.get("classification"),
                "analysis_quality": report.get("analysis_quality"),
                "skill": report.get("skill"),
                "action": report.get("action"),
                "summary": report.get("summary"),
                "actual_behavior": report.get("actual_behavior"),
                "repro_steps": report.get("repro_steps", []),
                "error_evidence": report.get("error_evidence", []),
                "suggested_actions": report.get("suggested_actions", []),
            },
            indent=2,
        )

    @mcp.resource("videobug://report/{report_id}/fix-prompt")
    def report_fix_prompt(report_id: str) -> str:
        """Rendered, evidence-only fix prompt for a report (shaped by its action)."""
        return render_fix_prompt(get_report(root, report_id))

    @mcp.resource("videobug://report/{report_id}/markdown")
    def report_markdown(report_id: str) -> str:
        """Shareable markdown report rendered from the bundle."""
        from framesleuth.render import render_markdown

        return render_markdown(get_report(root, report_id))

    @mcp.resource("videobug://report/{report_id}/issue")
    def report_issue(report_id: str) -> str:
        """GitHub-issue text (title + labels + body) rendered from the bundle."""
        from framesleuth.render import render as _render

        return _render(get_report(root, report_id), "issue")

    @mcp.prompt()
    def fix_from_video(report_id: str) -> str:
        """Script the tool sequence and emit a grounded fix prompt for a weak coder."""
        return render_fix_prompt(get_report(root, report_id))

    return mcp


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the videobug MCP server over stdio."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
