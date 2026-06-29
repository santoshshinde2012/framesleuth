"""Tests for the corpus-aware grounding upgrades (gitignore, ranking, bound)."""

from __future__ import annotations

from pathlib import Path

from framesleuth.pipeline.grounding import locate_in_code


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_gitignore_excludes_ignored_files(tmp_path: Path) -> None:
    """A file matched by .gitignore is not returned as a candidate."""
    _write(tmp_path, ".gitignore", "generated/\n*.gen.ts\n")
    _write(tmp_path, "src/handler.py", "def handler():\n    pass\n")
    _write(tmp_path, "generated/handler.py", "def handler():\n    pass\n")
    _write(tmp_path, "src/thing.gen.ts", "export function handler() {}\n")

    files = [c.file.replace("\\", "/") for c in locate_in_code(tmp_path, ["handler"])]
    assert "src/handler.py" in files
    assert not any("generated/" in f for f in files)
    assert not any(f.endswith(".gen.ts") for f in files)


def test_definition_outranks_comment_mention(tmp_path: Path) -> None:
    """A definition site ranks above an incidental comment mention."""
    _write(tmp_path, "a.py", "# toggle_theme is referenced here\nX = 1\n")
    _write(tmp_path, "b.py", "def toggle_theme():\n    return None\n")
    candidates = locate_in_code(tmp_path, ["toggle_theme"])
    assert candidates[0].file.replace("\\", "/") == "b.py"
    assert candidates[0].match_reason == "definition"


def test_distinctive_query_outranks_common_one(tmp_path: Path) -> None:
    """IDF weighting lifts a rare symbol above a word that appears everywhere."""
    for i in range(8):
        _write(tmp_path, f"mod{i}.py", "value = get()\n")  # 'get' is everywhere
    _write(tmp_path, "special.py", "def reconcile_ledger():\n    get()\n")
    candidates = locate_in_code(tmp_path, ["get", "reconcile_ledger"])
    # The distinctive symbol's definition should be the top candidate.
    assert candidates[0].file.replace("\\", "/") == "special.py"


def test_newly_supported_language_extension(tmp_path: Path) -> None:
    """C/C++ (and other added) extensions are now scanned."""
    _write(tmp_path, "main.cpp", "void render_frame() {}\n")
    files = [c.file for c in locate_in_code(tmp_path, ["render_frame"])]
    assert "main.cpp" in files


def test_camelcase_query_grounds_via_subwords(tmp_path: Path) -> None:
    """A camelCase query also matches via its sub-words (saveCart -> save/cart)."""
    _write(tmp_path, "cart.py", "def save_cart(item):\n    return item\n")
    files = [c.file for c in locate_in_code(tmp_path, ["saveCart"])]
    assert "cart.py" in files


def test_snake_case_query_grounds_via_subwords(tmp_path: Path) -> None:
    """A snake_case query's sub-words broaden recall to related identifiers."""
    _write(tmp_path, "profile.py", "def render_profile():\n    pass\n")
    files = [c.file for c in locate_in_code(tmp_path, ["user_profile"])]
    assert "profile.py" in files  # matched via the 'profile' sub-word


def test_scan_is_bounded_by_max_files(tmp_path: Path) -> None:
    """The scan stops at max_files so a huge repo stays bounded."""
    for i in range(20):
        _write(tmp_path, f"f{i:02d}.py", "def needle():\n    pass\n")
    # With a cap of 3 files, at most 3 distinct files can surface.
    candidates = locate_in_code(tmp_path, ["needle"], max_results=50, max_files=3)
    assert len({c.file for c in candidates}) <= 3
