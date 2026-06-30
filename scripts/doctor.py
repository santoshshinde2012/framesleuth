#!/usr/bin/env python3
"""Diagnose a Framesleuth local setup.

Pure standard library — run it with ANY Python, even when your virtualenv is
broken (that's often the problem):

    python3 scripts/doctor.py

It checks the things that commonly break a first run — a stale/active
virtualenv, whether the console scripts + ffmpeg are on PATH, whether the package
imports in *this* interpreter, the optional HTML→video render prerequisites, and
whether the backend and model servers are reachable — and prints a one-line fix
for anything that's wrong. Exit code is 0 unless ``--strict`` is given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "FAIL"
_BAD = {WARN, FAIL}


class Report:
    def __init__(self) -> None:
        self.failed = False

    def add(self, status: str, msg: str, fix: str = "") -> None:
        mark = {OK: " ok ", WARN: "warn", FAIL: "FAIL"}[status]
        print(f"[{mark}] {msg}")
        if status in _BAD and fix:
            print(f"        → {fix}")
        if status == FAIL:
            self.failed = True


def _reach(url: str, timeout: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data: bytes = r.read()
        return data.decode("utf-8", "replace")
    except Exception:
        return None


def check_python(r: Report) -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        r.add(OK, f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    else:
        r.add(
            FAIL,
            f"Python {v.major}.{v.minor} is too old (need 3.11+)",
            "create the venv with a 3.11+ interpreter: uv venv --python 3.11",
        )


def check_venv(r: Report) -> None:
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        r.add(
            WARN,
            "no virtualenv is active",
            "in the framesleuth dir: uv venv && source .venv/bin/activate",
        )
        return
    p = Path(venv)
    if (p / "bin" / "python3").exists() or (p / "Scripts" / "python.exe").exists():
        r.add(OK, f"virtualenv active and present: {venv}")
    else:
        r.add(
            FAIL,
            f"active virtualenv points to a MISSING path: {venv}",
            "deactivate 2>/dev/null; unset VIRTUAL_ENV; then recreate it in the "
            "framesleuth dir: uv venv && source .venv/bin/activate && "
            'uv pip install -e ".[dev]"',
        )


def check_tool(r: Report, name: str, fix: str, required: bool = True) -> None:
    if shutil.which(name):
        r.add(OK, f"{name} on PATH")
    else:
        r.add(FAIL if required else WARN, f"{name} not found on PATH", fix)


def check_package(r: Report) -> None:
    try:
        import framesleuth

        r.add(OK, f"framesleuth importable (v{getattr(framesleuth, '__version__', '?')})")
    except Exception as exc:
        r.add(
            FAIL,
            f"framesleuth not importable in this Python: {exc}",
            'install it: uv pip install -e ".[dev]" (in the active venv)',
        )


def check_render(r: Report) -> None:
    try:
        import importlib.metadata as md

        md.version("playwright")
        pw = True
    except Exception:
        pw = False
    ff = shutil.which("ffmpeg") is not None
    if pw and ff:
        r.add(OK, "HTML→video prerequisites present (playwright + ffmpeg)")
    else:
        missing = []
        if not pw:
            missing.append('uv pip install -e ".[render]"')
        if not ff:
            missing.append("install ffmpeg on PATH")
        r.add(
            WARN,
            "HTML→video is optional and not fully set up",
            " && ".join(missing) + " && playwright install chromium (then restart)",
        )


def check_servers(r: Report) -> None:
    health = _reach("http://127.0.0.1:8010/v1/healthz")
    if health is not None:
        r.add(OK, "backend reachable at http://127.0.0.1:8010")
    else:
        r.add(
            WARN,
            "backend not reachable at http://127.0.0.1:8010",
            "start it from the framesleuth dir (venv active): framesleuth-api",
        )

    ollama = _reach("http://127.0.0.1:11434/api/tags")
    llama = _reach("http://127.0.0.1:8080/v1/models")
    if ollama is not None:
        has_vlm = "qwen2.5vl" in ollama or "qwen2.5-vl" in ollama
        r.add(
            OK if has_vlm else WARN,
            "Ollama reachable at :11434" + ("; qwen2.5vl present" if has_vlm else ""),
            "" if has_vlm else "pull a vision model: ollama pull qwen2.5vl",
        )
        if ollama is not None:
            r.add(
                WARN,
                "using the Ollama vision path? point the VLM at it",
                "in .env set VLM_URL=http://127.0.0.1:11434 and VLM_MODEL=qwen2.5vl",
            )
    elif llama is not None:
        r.add(OK, "llama.cpp reachable at :8080 (VLM_URL default)")
    else:
        r.add(
            WARN,
            "no model server reachable (:11434 Ollama or :8080 llama.cpp)",
            "start one, e.g.: ollama serve & ; ollama pull qwen2.5vl. Analyses still "
            "run in degraded (sidecar-only) mode without it.",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a Framesleuth local setup")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on any FAIL")
    args = parser.parse_args()

    print("Framesleuth setup doctor")
    print("=" * 52)
    r = Report()
    check_python(r)
    check_venv(r)
    check_tool(
        r,
        "framesleuth-api",
        "install the package in the active venv: " 'uv pip install -e ".[dev]"',
    )
    check_tool(r, "framesleuth-mcp", "same as above — comes from the package install")
    check_package(r)
    check_render(r)
    check_tool(
        r, "ffmpeg", "brew install ffmpeg (macOS) / apt install ffmpeg (Linux)", required=False
    )
    check_servers(r)
    print("=" * 52)
    print("Tip: run this with a plain `python3` — it works even when the venv is broken.")

    if args.strict and r.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
