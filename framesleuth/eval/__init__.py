"""Deterministic evaluation harness for Framesleuth's analysis quality.

Measures the parts of the pipeline that decide whether the output is *good* —
classification accuracy, grounding recall, and citation integrity — without any
model calls, so it runs in CI and gives an empirical signal on prompt/heuristic
changes. See :mod:`framesleuth.eval.harness`.
"""

from framesleuth.eval.harness import (
    EvalMetrics,
    EvalResult,
    evaluate_bundle,
    run_all,
    run_citation_eval,
    run_classification_eval,
    run_faithfulness_eval,
    run_grounding_eval,
)

__all__ = [
    "EvalMetrics",
    "EvalResult",
    "evaluate_bundle",
    "run_all",
    "run_citation_eval",
    "run_classification_eval",
    "run_faithfulness_eval",
    "run_grounding_eval",
]
