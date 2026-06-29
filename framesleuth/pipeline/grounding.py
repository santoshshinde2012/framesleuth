"""Repository grounding: rank candidate code locations from evidence.

Works for both bugs (ground error symbols to their source) and build/feature work
(ground intent + on-screen UI nouns to the files to extend). Ranking prefers
*definition* lines — function/class/component declarations — over incidental
matches in comments or strings, so the candidate is a place to act, not just a
place a word appears.

Retrieval is lexical and deterministic (no embeddings, no external index) but
applies two corpus-aware signals on top of the per-line match: an IDF weight so a
distinctive symbol outranks a common word, and a whole-word boost so ``save``
prefers ``def save(`` over ``unsaved``. The scan respects ``.gitignore`` and a
hard file-count cap so a large monorepo cannot make it unbounded.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections.abc import Iterable
from pathlib import Path

from framesleuth.logging_config import get_logger
from framesleuth.schemas import CodeCandidate

logger = get_logger("pipeline.grounding")

# Source extensions worth grounding against — Python plus the common web/app and
# systems stacks a bug or feature video is likely about. Keeps the scan bounded
# vs. reading every file.
_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".svelte",
    ".go",
    ".rb",
    ".java",
    ".kt",
    ".php",
    ".cs",
    ".rs",
    ".swift",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".scala",
    ".dart",
    ".ex",
    ".exs",
    ".lua",
    ".r",
    ".sql",
}
# Vendored / generated dirs never worth grounding into (a baseline that applies
# even when a repo has no .gitignore).
_SKIP_DIRS = {
    "node_modules",
    ".venv",
    "venv",
    ".git",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".mypy_cache",
    "site-packages",
    "target",
    "vendor",
}
# Lines that *declare* a symbol — strong grounding anchors across languages.
_DEFINITION = re.compile(
    r"\b(def|class|function|func|const|let|var|export|interface|type|component|struct|fn)\b"
    r"|=>|=\s*\(",
)
# Split camelCase / PascalCase boundaries (``saveCart`` -> ``save`` + ``Cart``).
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifier(token: str) -> list[str]:
    """Break an identifier into its sub-words (snake/kebab/camel boundaries)."""
    parts: list[str] = []
    for chunk in re.split(r"[_\-\s.]+", token):
        parts.extend(p for p in _CAMEL_BOUNDARY.split(chunk) if p)
    return parts


def _expand_queries(queries: list[str]) -> list[str]:
    """Add identifier sub-words so ``saveCart`` also grounds via ``save``/``cart``.

    Only sub-words of length >= 4 are added (shorter ones are noisy), and the
    original queries keep their place; the corpus-aware IDF weighting then keeps a
    common sub-word from drowning out the distinctive full symbol.
    """
    expanded = list(queries)
    seen = {q.lower() for q in queries}
    for query in queries:
        for sub in _split_identifier(query):
            if len(sub) >= 4 and sub.lower() not in seen:
                seen.add(sub.lower())
                expanded.append(sub)
    return expanded


class _GitignoreMatcher:
    """A pragmatic ``.gitignore`` matcher (common patterns, not the full spec).

    Loads the repo-root ``.gitignore`` and supports the cases that matter for
    grounding: blank/comment lines, negation (``!``), anchored paths (``/dist``),
    directory rules (``build/``), and globs (``*.log``, ``**/gen``). It is a best
    effort layered on top of the always-skipped vendored directories — better than
    ignoring ``.gitignore`` entirely, without pulling in a dependency.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._rules: list[tuple[str, bool, bool]] = []  # (pattern, negated, has_slash)
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            return
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            # A trailing slash marks a directory rule; a slash elsewhere anchors the
            # pattern to a path. We normalize both into a segment/path match below.
            stripped = line.rstrip("/").lstrip("/")
            if stripped:
                self._rules.append((stripped, negated, "/" in stripped))

    def ignored(self, path: Path) -> bool:
        """Whether ``path`` (under the repo root) is ignored by the loaded rules.

        A directory rule (``build/``) excludes everything beneath it, so the match is
        tested against every path segment, not just the basename.
        """
        if not self._rules:
            return False
        try:
            rel = path.relative_to(self._root).as_posix()
        except ValueError:
            return False
        parts = rel.split("/")
        result = False
        for pattern, negated, has_slash in self._rules:
            if has_slash:
                matched = fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, f"{pattern}/*")
            else:
                matched = any(fnmatch.fnmatch(part, pattern) for part in parts)
            if matched:
                result = not negated
        return result


def _score_match(line: str, query: str, word_re: re.Pattern[str]) -> tuple[float, str]:
    """Score a query/line match and report why (definition vs. plain search)."""
    if query.lower() not in line.lower():
        return (0.0, "")
    base = min(1.0, 0.5 + (len(query) / max(20, len(line))))
    # A whole-word hit is a stronger signal than an incidental substring.
    if word_re.search(line):
        base = min(1.0, base + 0.15)
    stripped = line.lstrip()
    if stripped.startswith(("#", "//", "*", "/*")):
        return (max(0.1, base - 0.25), "comment_search")  # incidental mention
    if _DEFINITION.search(line):
        return (min(1.0, base + 0.2), "definition")  # a place to act
    return (base, "search")


def _iter_code_files(
    matcher: _GitignoreMatcher, workspace_root: Path, max_files: int
) -> list[Path]:
    """Collect source files, skipping vendored/generated/ignored dirs, bounded by count."""
    files: list[Path] = []
    for path in workspace_root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix not in _CODE_EXTENSIONS:
            continue
        if matcher.ignored(path):
            continue
        files.append(path)
        if len(files) >= max_files:
            logger.info("Grounding scan hit the %d-file cap; remaining files skipped", max_files)
            break
    return files


def _read_documents(
    matcher: _GitignoreMatcher, workspace_root: Path, max_files: int
) -> list[tuple[Path, list[str]]]:
    """Read each in-scope source file once into ``(path, lines)`` pairs."""
    documents: list[tuple[Path, list[str]]] = []
    for file_path in _iter_code_files(matcher, workspace_root, max_files):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        documents.append((file_path, lines))
    return documents


def _idf_weights(documents: list[tuple[Path, list[str]]], queries: list[str]) -> dict[str, float]:
    """Per-query IDF weight in ``[0.5, 1.0]`` — rarer across the corpus weighs more."""
    doc_freq = dict.fromkeys(queries, 0)
    for _, lines in documents:
        blob = "\n".join(lines).lower()
        for query in queries:
            if query.lower() in blob:
                doc_freq[query] += 1
    total_docs = max(1, len(documents))
    idf = {q: math.log(1 + total_docs / (1 + doc_freq[q])) for q in queries}
    max_idf = max(idf.values()) if idf else 1.0
    return {q: 0.5 + 0.5 * (idf[q] / max_idf) for q in queries}


def _collect_candidates(
    documents: list[tuple[Path, list[str]]],
    queries: list[str],
    workspace_root: Path,
    word_res: dict[str, re.Pattern[str]],
    idf_weight: dict[str, float],
) -> list[CodeCandidate]:
    """Score every (line, query) match into a flat candidate list."""
    candidates: list[CodeCandidate] = []
    for file_path, lines in documents:
        is_third_party = "site-packages" in str(file_path)
        rel = str(file_path.relative_to(workspace_root))
        for line_no, line in enumerate(lines, start=1):
            for query in queries:
                score, reason = _score_match(line, query, word_res[query])
                if score <= 0:
                    continue
                candidates.append(
                    CodeCandidate(
                        file=rel,
                        line=line_no,
                        symbol=None,
                        match_reason=reason,
                        confidence=round(min(1.0, score * idf_weight[query]), 4),
                        is_third_party=is_third_party,
                    )
                )
    return candidates


def locate_in_code(
    workspace_root: Path,
    queries: Iterable[str],
    max_results: int = 10,
    *,
    max_files: int = 5000,
) -> list[CodeCandidate]:
    """Locate code candidates via deterministic, corpus-aware text matching.

    Files are scanned once. A query's IDF weight (rarer across the corpus → higher)
    and a whole-word boost refine the per-line score so distinctive symbols and
    definition sites rise to the top. The scan respects ``.gitignore`` and stops at
    ``max_files`` so a large repository stays bounded.
    """
    normalized_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if not normalized_queries:
        return []
    normalized_queries = _expand_queries(normalized_queries)

    matcher = _GitignoreMatcher(workspace_root)
    word_res = {q: re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE) for q in normalized_queries}
    documents = _read_documents(matcher, workspace_root, max_files)
    idf_weight = _idf_weights(documents, normalized_queries)
    candidates = _collect_candidates(
        documents, normalized_queries, workspace_root, word_res, idf_weight
    )

    # Deterministic ordering: confidence desc, file asc, line asc; one hit per line.
    ordered = sorted(candidates, key=lambda c: (-c.confidence, c.file, c.line))
    unique: list[CodeCandidate] = []
    seen: set[tuple[str, int]] = set()
    for candidate in ordered:
        key = (candidate.file, candidate.line)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= max_results:
            break

    return unique
