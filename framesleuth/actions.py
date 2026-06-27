"""Actions: named "what should the agent do next" modes for a report.

An *action* is the response counterpart to a :mod:`~framesleuth.skills` summary
skill. Where a skill shapes the prose *summary* of a recording, an action shapes
the **task block** of the grounded fix prompt — i.e. what a downstream coding
agent (Copilot/Claude) is told to actually do with the evidence: fix it, explain
it, triage it, write a test, draft a report, or reproduce it.

Callers either pick a built-in action by name, supply a fully custom
``action_prompt`` to override, or (the default) let the action be *auto-picked*
from the recording's classification label. The perception pipeline is untouched;
only the rendered instruction changes.

This module also derives :func:`suggest_actions` — a machine-readable menu of
sensible next steps for a report, so an agent can present or auto-invoke them
rather than guessing what is possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from framesleuth.logging_config import get_logger

logger = get_logger("actions")


@dataclass(frozen=True)
class Action:
    """A named response mode that shapes the fix-prompt's task block."""

    name: str
    description: str
    task: str


# The trailing instruction shared by every action keeps citations honest
# regardless of the task. Rendered after the per-action ``task`` block.
ACTION_FOOTER = "Be concise and precise. Always cite the evidence you used."


ACTIONS: dict[str, Action] = {
    "fix": Action(
        name="fix",
        description="Diagnose the root cause and propose/make a minimal, targeted fix.",
        task=(
            "## Your task:\n"
            "1. Restate the user's request in one line and how the evidence supports it.\n"
            "2. If it is a bug: confirm the root cause from the evidence + candidate "
            "locations, then propose a minimal, targeted fix. If it is a feature/change "
            "request: propose the concrete implementation, grounded in the candidate "
            "locations and current behavior shown.\n"
            "3. Make the edits (or a precise patch plan) referencing real files/lines.\n"
            "4. Note potential regressions and any evidence you still need."
        ),
    ),
    "explain": Action(
        name="explain",
        description="Explain what the recording shows and what happened — no code changes.",
        task=(
            "## Your task:\n"
            "Explain what the recording shows and what happened, grounded ONLY in the "
            "evidence. Do NOT propose or make code changes. Cover: what the user did, "
            "what the system did in response, and whether anything went wrong. If the "
            "evidence is thin, say plainly what is missing."
        ),
    ),
    "triage": Action(
        name="triage",
        description="Assess severity/priority and route to the right component — no fix.",
        task=(
            "## Your task:\n"
            "Triage this report from the evidence. Do NOT fix anything. Output: "
            "severity, priority, the most likely affected component/area (use the "
            "candidate code locations), a routing recommendation (who/where), and your "
            "confidence. Flag any conclusion that depends on missing evidence."
        ),
    ),
    "test": Action(
        name="test",
        description="Write a failing regression test that reproduces the behavior.",
        task=(
            "## Your task:\n"
            "Write a failing regression test that reproduces the observed behavior, "
            "grounded in the repro steps and error evidence. Use the repository's "
            "existing test framework and conventions (infer them from the candidate "
            "locations). Output the test file path and the test code, and explain what "
            "it asserts and why it fails today. Do NOT fix the underlying bug."
        ),
    ),
    "report": Action(
        name="report",
        description="Produce a shareable, ready-to-paste issue/PR description.",
        task=(
            "## Your task:\n"
            "Produce a shareable report from the evidence: a clear title, environment, "
            "numbered steps to reproduce, expected vs actual behavior, severity, and the "
            "key error evidence with timestamps. Write it as a ready-to-paste issue "
            "body. Do NOT make code changes."
        ),
    ),
    "reproduce": Action(
        name="reproduce",
        description="Produce minimal exact steps / a script to reproduce locally.",
        task=(
            "## Your task:\n"
            "Produce the minimal, exact steps (and a script or commands where possible) "
            "to reproduce the observed behavior locally, grounded in the repro steps, "
            "environment, and error evidence. Call out preconditions and any data "
            "needed. Do NOT fix anything."
        ),
    ),
}

DEFAULT_ACTION = "fix"

# When no action is requested, pick one from the recording's classification.
_AUTO_BY_LABEL: dict[str, str] = {
    "bug": "fix",
    "tutorial": "explain",
    "demo": "explain",
    "feedback": "report",
    "other": "explain",
}


def auto_action_for(classification_label: str | None) -> str:
    """Return the action name auto-picked for a classification label."""
    if not classification_label:
        return DEFAULT_ACTION
    return _AUTO_BY_LABEL.get(classification_label.lower(), DEFAULT_ACTION)


def resolve_action(
    action: str | None,
    action_prompt: str | None,
    classification_label: str | None = None,
) -> tuple[str, str | None, str]:
    """Resolve a request into ``(label, custom_prompt, task)``.

    Precedence:
      1. an explicit ``action_prompt`` wins (label ``"custom"``; ``task`` is it);
      2. a known ``action`` name;
      3. otherwise auto-pick from ``classification_label`` (the default).

    An unknown action name falls back to the auto-picked action rather than
    failing the run. ``custom_prompt`` is the verbatim custom task to persist
    (non-``None`` only for the custom case), so a report can be re-rendered later.
    """
    if action_prompt and action_prompt.strip():
        text = action_prompt.strip()
        return ("custom", text, text)
    if action:
        key = action.strip().lower()
        if key in ACTIONS:
            return (key, None, ACTIONS[key].task)
        logger.warning("Unknown action %r; auto-picking from classification", action)
    auto = auto_action_for(classification_label)
    return (auto, None, ACTIONS[auto].task)


def resolve_action_task(
    action: str | None,
    action_prompt: str | None,
    classification_label: str | None = None,
) -> str:
    """Return just the task block for an action (used at render time)."""
    return resolve_action(action, action_prompt, classification_label)[2]


def list_actions() -> list[dict[str, str]]:
    """Return the catalog of built-in actions as ``{name, description}`` dicts."""
    return [{"name": a.name, "description": a.description} for a in ACTIONS.values()]


def _has(report: dict[str, Any], key: str) -> bool:
    """Whether a report has a non-empty list/value at ``key``."""
    return bool(report.get(key))


def suggest_actions(report: dict[str, Any]) -> list[dict[str, str]]:
    """Derive a machine-readable menu of sensible next steps for a report.

    Each item is ``{action, label, rationale, ref}`` where ``ref`` names the MCP
    resource/tool (or ``analyze_video`` action) that performs it, so an agent can
    present the menu or auto-invoke an entry. The set adapts to the report's
    classification label and analysis quality.
    """
    report_id = str(report.get("id", ""))
    label = str((report.get("classification") or {}).get("label", "other")).lower()
    quality = str((report.get("analysis_quality") or {}).get("level", "full")).lower()
    has_errors = _has(report, "error_evidence")
    has_candidates = _has(report, "code_candidates")

    suggestions: list[dict[str, str]] = []

    def add(action: str, label_text: str, rationale: str, ref: str) -> None:
        suggestions.append(
            {"action": action, "label": label_text, "rationale": rationale, "ref": ref}
        )

    if quality == "degraded":
        add(
            "recapture",
            "Re-record with console & network logs",
            "Analysis quality is degraded — there is not enough evidence to act confidently.",
            "capture with sidecars attached",
        )

    if label == "bug" or has_errors:
        add(
            "propose_fix",
            "Propose a grounded fix",
            "A failure was observed; the fix-prompt confirms the root cause from cited evidence.",
            f"resource videobug://report/{report_id}/fix-prompt",
        )
        add(
            "write_test",
            "Write a failing regression test",
            "Lock the bug in with a test before fixing it.",
            "tool render(report_id, 'test-plan') or analyze_video(action='test')",
        )
        add(
            "locate_in_code",
            "Locate the suspect code",
            "Ground the error text to candidate files/lines in the repo.",
            "tool locate_in_code(report_id, repo_root)"
            if not has_candidates
            else "report.code_candidates",
        )
        add(
            "open_issue",
            "Open a shareable issue",
            "Hand off a ready-to-paste bug report.",
            "tool render(report_id, 'issue')",
        )
    elif label in {"tutorial", "demo"}:
        add(
            "explain",
            "Explain / document what was shown",
            "The recording demonstrates a flow rather than a failure.",
            f"resource videobug://report/{report_id}/summary",
        )
        add(
            "write_docs",
            "Draft user-facing notes",
            "Turn the demonstrated steps into a tutorial or changelog entry.",
            "tool render(report_id, 'markdown')",
        )
    else:  # feedback / other
        add(
            "summarize",
            "Summarize the recording",
            "Capture what was said/shown for follow-up.",
            f"resource videobug://report/{report_id}/summary",
        )
        add(
            "open_issue",
            "File it as a tracked item",
            "Route feedback/requests to a tracker.",
            "tool render(report_id, 'issue')",
        )

    return suggestions
