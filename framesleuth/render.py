"""Renderers: turn a Context Bundle into a consumable end-user artifact.

The bundle is the canonical machine output; these pure functions project it into
human/agent-facing shapes (markdown report, GitHub issue, test plan) without
adding any new information — everything is read from the bundle, so a renderer
never fabricates. Keep them dependency-free and deterministic.
"""

from __future__ import annotations

from typing import Any

# Supported render formats (kept in one place for validation + listing).
RENDER_FORMATS = ("markdown", "issue", "test-plan")

# Map analysis-quality level → GitHub issue label, surfaced so triagers can see
# at a glance how much to trust the report.
_QUALITY_LABEL = {
    "full": "evidence:full",
    "partial": "evidence:partial",
    "degraded": "evidence:degraded",
}


def _steps_lines(report: dict[str, Any]) -> list[str]:
    """Render numbered repro steps as markdown list items."""
    lines = []
    for step in report.get("repro_steps", []):
        n = step.get("n", "?")
        action = step.get("action", "")
        t = step.get("t")
        suffix = f" _(t={t}s)_" if isinstance(t, int | float) else ""
        lines.append(f"{n}. {action}{suffix}")
    return lines or ["1. (no reproduction steps were extracted)"]


def _error_lines(report: dict[str, Any]) -> list[str]:
    """Render error evidence as markdown bullets with source + timestamp."""
    lines = []
    for item in report.get("error_evidence", []):
        source = item.get("source", "?")
        t = item.get("t")
        text = str(item.get("text", "")).strip().splitlines()[0] if item.get("text") else ""
        stamp = f"t={t}s " if isinstance(t, int | float) else ""
        lines.append(f"- `{source}` {stamp}— {text}")
    return lines


def _key_moment_lines(report: dict[str, Any]) -> list[str]:
    """Render the timestamped key moments as markdown bullets."""
    lines = []
    for moment in report.get("key_moments", []):
        t = moment.get("t")
        stamp = f"_(t={t}s)_ " if isinstance(t, int | float) else ""
        kind = moment.get("kind", "scene")
        label = str(moment.get("label", "")).strip()
        if label:
            lines.append(f"- {stamp}`{kind}` — {label}")
    return lines


def _is_bug_report(report: dict[str, Any]) -> bool:
    """Whether the report carries bug-shaped fields worth rendering as a bug."""
    cls = str((report.get("classification") or {}).get("label", "")).lower()
    return cls == "bug" or bool(report.get("error_evidence")) or bool(report.get("actual_behavior"))


def _quality_note(report: dict[str, Any]) -> str:
    """One-line trust note derived from analysis_quality."""
    quality = report.get("analysis_quality") or {}
    level = quality.get("level", "full")
    warnings = quality.get("warnings", [])
    note = f"**Analysis quality:** `{level}`"
    if warnings:
        note += " — " + "; ".join(str(w) for w in warnings)
    return note


def render_markdown(report: dict[str, Any]) -> str:
    """Render a shareable markdown report of the bundle.

    Adapts to the video: a bug report leads with steps/expected-vs-actual/errors;
    a general video leads with the summary and key moments, omitting the bug-shaped
    sections that would otherwise render as empty ``n/a`` noise.
    """
    title = report.get("title", "Untitled recording")
    cls = (report.get("classification") or {}).get("label", "other")
    env = report.get("environment") or {}
    env_str = ", ".join(f"{k}={v}" for k, v in env.items()) or "(unknown)"
    is_bug = _is_bug_report(report)

    meta = f"**Type:** {cls}"
    if report.get("severity"):
        meta += f"  |  **Severity:** {report['severity']}"
    if report.get("suspected_component"):
        meta += f"  |  **Component:** {report['suspected_component']}"

    parts: list[str] = [f"# {title}", "", meta, "", _quality_note(report)]
    if env:
        parts += ["", f"**Environment:** {env_str}"]
    if report.get("summary"):
        parts += ["", "## Summary", "", str(report["summary"])]

    moment_lines = _key_moment_lines(report)
    if moment_lines:
        parts += ["", "## Key moments", "", *moment_lines]

    if is_bug:
        parts += ["", "## Steps to reproduce", "", *_steps_lines(report)]
        parts += [
            "",
            "## Expected vs actual",
            "",
            f"- **Expected:** {report.get('expected_behavior', 'n/a')}",
            f"- **Actual:** {report.get('actual_behavior', 'n/a')}",
        ]
    elif report.get("repro_steps"):
        parts += ["", "## Observed steps", "", *_steps_lines(report)]

    error_lines = _error_lines(report)
    if error_lines:
        parts += ["", "## Error evidence", "", *error_lines]
    candidates = report.get("code_candidates", [])
    if candidates:
        parts += ["", "## Candidate code locations", ""]
        parts += [
            f"- `{c.get('file', '?')}:{c.get('line', '?')}` — {c.get('match_reason', '?')}"
            for c in candidates
        ]
    return "\n".join(parts) + "\n"


def render_github_issue(report: dict[str, Any]) -> dict[str, Any]:
    """Render a GitHub issue payload: ``{title, labels, body}``.

    The body reuses the markdown renderer (minus its H1, since the title is
    separate); labels encode the type and the evidence-quality level.
    """
    title = str(report.get("title", "Recording report"))
    cls = str((report.get("classification") or {}).get("label", "other")).lower()
    level = str((report.get("analysis_quality") or {}).get("level", "full")).lower()

    labels = [cls]
    if cls == "bug":
        labels.append(f"severity:{report.get('severity', 'medium')}")
    if level in _QUALITY_LABEL:
        labels.append(_QUALITY_LABEL[level])

    body_md = render_markdown(report)
    # Drop the leading "# title" line; the issue title carries it.
    body_lines = body_md.splitlines()
    if body_lines and body_lines[0].startswith("# "):
        body_lines = body_lines[1:]
    body = "\n".join(body_lines).lstrip("\n")
    body += f"\n\n_Generated by Framesleuth from `{report.get('source_video', 'a recording')}`._\n"
    return {"title": title, "labels": labels, "body": body}


def render_test_plan(report: dict[str, Any]) -> str:
    """Render a framework-agnostic regression test plan from the evidence."""
    title = report.get("title", "the observed behavior")
    parts: list[str] = [
        f"# Regression test plan: {title}",
        "",
        _quality_note(report),
        "",
        "## What to assert",
        "",
        f"- The flow should: {report.get('expected_behavior') or 'complete without error'}",
        f"- Today it instead: {report.get('actual_behavior') or 'n/a'}",
        "",
        "## Arrange — preconditions",
        "",
        f"- {report.get('preconditions') or 'Application is running and the page is loaded.'}",
    ]
    env = report.get("environment") or {}
    if env:
        parts.append("- Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    parts += ["", "## Act — reproduction steps", "", *_steps_lines(report)]

    error_lines = _error_lines(report)
    if error_lines:
        parts += [
            "",
            "## Assert — failure signal to detect",
            "",
            "The test should fail today by detecting:",
            *error_lines,
        ]
    candidates = report.get("code_candidates", [])
    if candidates:
        first = candidates[0]
        parts += [
            "",
            "## Suggested location",
            "",
            f"Start near `{first.get('file', '?')}:{first.get('line', '?')}` "
            f"({first.get('match_reason', 'grounded from error text')}); "
            "follow the repository's existing test framework and conventions.",
        ]
    return "\n".join(parts) + "\n"


def render(report: dict[str, Any], fmt: str) -> str:
    """Render ``report`` in ``fmt`` (one of :data:`RENDER_FORMATS`) as text.

    The ``issue`` format is serialized as ``Title / Labels / body`` text so it is
    usable from a plain string tool result; callers needing the structured
    payload can use :func:`render_github_issue` directly.
    """
    key = fmt.strip().lower()
    if key == "markdown":
        return render_markdown(report)
    if key == "test-plan":
        return render_test_plan(report)
    if key == "issue":
        issue = render_github_issue(report)
        labels = ", ".join(issue["labels"])
        return f"Title: {issue['title']}\nLabels: {labels}\n\n{issue['body']}"
    raise ValueError(f"unknown render format {fmt!r}; expected one of {RENDER_FORMATS}")
