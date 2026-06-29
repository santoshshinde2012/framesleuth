"""Per-field confidence and task-aware actionability for a Context Bundle.

``analysis_quality.level`` says how much of the *pipeline* ran. These signals say
something finer and more useful to a downstream agent: how much to trust each
individual claim (``field_confidence``), and whether the evidence is sufficient
for the *resolved action* (``actionability``) — a bundle can have "full" pipeline
quality yet be "insufficient" to implement a feature if no UI was captured.
"""

from __future__ import annotations

from typing import Literal

from framesleuth.schemas import ContextBundle

Actionability = Literal["ready", "thin", "insufficient"]


def compute_field_confidence(bundle: ContextBundle) -> dict[str, float]:
    """Confidence 0-1 for the key bundle fields, from the evidence behind each."""
    has_errors = bool(bundle.error_evidence)
    conf: dict[str, float] = {}

    # Title: strong when anchored to extracted error text, weaker when paraphrasing
    # a scene caption.
    conf["title"] = 0.85 if has_errors else 0.55

    # Repro/observed steps: the mean of the per-step confidences the extractor set.
    if bundle.repro_steps:
        conf["repro_steps"] = round(
            sum(s.confidence for s in bundle.repro_steps) / len(bundle.repro_steps), 2
        )
    else:
        conf["repro_steps"] = 0.0

    # Severity/priority are bug-shaped guesses — scored only when actually set
    # (a general video leaves them null, so there is nothing to rate).
    if bundle.severity is not None:
        conf["severity"] = 0.8 if has_errors else 0.35

    if bundle.suspected_component:
        conf["suspected_component"] = (
            0.7 if bundle.suspected_component not in ("", "unknown") else 0.3
        )

    # The summary is the deliverable for a general video — rate it by whether one
    # was produced at all (the model either summarized the evidence or degraded).
    if bundle.summary:
        conf["summary"] = 0.7

    if bundle.key_moments:
        conf["key_moments"] = round(min(0.9, 0.4 + 0.05 * len(bundle.key_moments)), 2)

    if bundle.code_candidates:
        conf["code_candidates"] = round(
            sum(c.confidence for c in bundle.code_candidates) / len(bundle.code_candidates), 2
        )

    bc = bundle.build_context
    if bc is not None:
        signal = len(bc.screens) + len(bc.components) + len(bc.user_flow)
        # 0 signal -> 0.3, saturating toward 0.9 as structured evidence accumulates.
        conf["build_context"] = round(min(0.9, 0.3 + 0.1 * signal), 2)

    _apply_corroboration(bundle, conf)
    return conf


def _apply_corroboration(bundle: ContextBundle, conf: dict[str, float]) -> None:
    """Boost confidence where independent evidence sources agree.

    A single signal is a guess; two independent signals pointing the same way are
    corroboration. When an observed failure *also* grounds to a code location, the
    title and that candidate are more trustworthy; when a summary is backed by
    distinct key moments, it is more trustworthy. Boosts are small and capped at 1.0
    so they sharpen — never invent — confidence.
    """
    has_errors = bool(bundle.error_evidence)

    def _bump(key: str, delta: float) -> None:
        if key in conf:
            conf[key] = round(min(1.0, conf[key] + delta), 2)

    if has_errors and bundle.code_candidates:
        _bump("title", 0.05)
        _bump("code_candidates", 0.1)
    if bundle.summary and len(bundle.key_moments) >= 2:
        _bump("summary", 0.1)


def assess_actionability(bundle: ContextBundle) -> Actionability:
    """Whether the evidence suffices for the bundle's resolved action.

    Independent of which stages ran: it asks "given what we extracted, can the
    downstream agent actually do the requested action?" A degraded pipeline can
    never be more than ``thin``.
    """
    action = (bundle.action or "").lower()
    degraded = bundle.analysis_quality.level == "degraded"
    has_errors = bool(bundle.error_evidence)
    has_candidates = bool(bundle.code_candidates)
    bc = bundle.build_context
    has_build = bc is not None and bool(bc.screens or bc.components)

    if action in {"implement", "design"}:
        level: Actionability = (
            "ready" if has_build else ("thin" if (bc and bc.components) else "insufficient")
        )
    elif action == "fix":
        if has_errors and has_candidates:
            level = "ready"
        elif has_errors or has_candidates:
            level = "thin"
        else:
            level = "insufficient"
    elif action == "test":
        level = "ready" if (has_errors and bundle.repro_steps) else "thin"
    elif action == "summarize":
        # A summary or distilled key moments IS the deliverable here.
        level = "ready" if (bundle.summary or bundle.key_moments) else "insufficient"
    else:  # explain / report / triage / reproduce / custom
        any_evidence = (
            has_errors
            or has_build
            or len(bundle.repro_steps) > 0
            or bool(bundle.summary)
            or bool(bundle.key_moments)
        )
        level = "ready" if any_evidence else "insufficient"

    if degraded and level == "ready":
        level = "thin"
    return level
