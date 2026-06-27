#!/usr/bin/env python3
"""Prepare local model prerequisites for Framesleuth.

What this script does:
  * Pre-fetches the faster-whisper ASR model — the only weight Framesleuth itself
    downloads. It is cached under ``~/.cache/huggingface`` and reused on every run.
  * Checks that your VLM (vision) and coder model servers are reachable and that
    the configured models look available, printing the exact ``ollama pull`` (or
    equivalent) command when something is missing.

What it does NOT do: download the vision/coder LLMs. Those are **served by your
engine** (Ollama / llama.cpp / vLLM), not placed on disk by Framesleuth. For the
verified Ollama path that means, for example::

    ollama serve &
    ollama pull qwen2.5vl          # VLM (frame understanding)
    ollama pull qwen2.5-coder:7b   # coder

See the README "Start & stop the stack" section for the full setup.
"""

from __future__ import annotations

import argparse
import os
import urllib.request


def _http_get(url: str, timeout: float = 4.0) -> str | None:
    """Return the response body, or None if the endpoint isn't reachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _load_config() -> dict[str, str]:
    """Configured model server URLs/names — from Settings, falling back to env."""
    try:
        from framesleuth.config import get_settings

        s = get_settings()
        return {
            "vlm_url": s.VLM_URL,
            "vlm_model": s.VLM_MODEL,
            "coder_url": s.CODER_URL,
            "coder_model": s.CODER_MODEL,
        }
    except Exception:
        return {
            "vlm_url": os.environ.get("VLM_URL", "http://127.0.0.1:8080"),
            "vlm_model": os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct-GGUF"),
            "coder_url": os.environ.get("CODER_URL", "http://127.0.0.1:11434"),
            "coder_model": os.environ.get("CODER_MODEL", "qwen2.5-coder:7b"),
        }


def prefetch_asr(model: str) -> bool:
    """Download (and cache) the faster-whisper model so the first run isn't slow."""
    print(f"→ Pre-fetching faster-whisper '{model}' (cached under ~/.cache/huggingface)…")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print('  [skip] faster-whisper not installed — run: uv pip install -e ".[dev]"')
        return False
    try:
        WhisperModel(model, compute_type="int8")  # triggers the one-time download
        print(f"  [ok] faster-whisper '{model}' is ready")
        return True
    except Exception as exc:
        print(f"  [fail] could not fetch '{model}': {exc}")
        return False


def check_server(label: str, url: str, model: str) -> bool:
    """Probe a model server (Ollama or OpenAI-compatible) and report readiness."""
    ollama = _http_get(f"{url}/api/tags")
    openai = _http_get(f"{url}/v1/models")
    body = ollama if ollama is not None else openai
    if body is None:
        print(f"  [down] {label}: not reachable at {url}")
        print("         start your engine (e.g. `ollama serve`), then pull the model")
        return False

    # Best-effort presence check (engines name models differently).
    short = model.split("/")[-1].split(":")[0]
    if model in body or short in body:
        print(f"  [ok]   {label}: server up at {url}; '{model}' available")
        return True

    print(f"  [warn] {label}: server up at {url}, but '{model}' not found")
    if ollama is not None:
        print(f"         run: ollama pull {model}")
    else:
        print(f"         load '{model}' into your server (see README)")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Framesleuth model prerequisites")
    parser.add_argument(
        "--asr-model", default="small", help="faster-whisper size to pre-fetch (default: small)"
    )
    parser.add_argument("--skip-asr", action="store_true", help="don't pre-fetch the ASR model")
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero if any prerequisite is missing"
    )
    args = parser.parse_args()

    cfg = _load_config()
    print("Framesleuth model prerequisites")
    print("-" * 48)

    asr_ok = True if args.skip_asr else prefetch_asr(args.asr_model)

    print("\nModel servers (served by your engine — not downloaded here):")
    vlm_ok = check_server("VLM  ", cfg["vlm_url"], cfg["vlm_model"])
    coder_ok = check_server("Coder", cfg["coder_url"], cfg["coder_model"])

    print("-" * 48)
    if not (vlm_ok and coder_ok):
        print("Tip: with Ollama —")
        print("  ollama serve &")
        print(f"  ollama pull {cfg['coder_model']}")
        print("  ollama pull qwen2.5vl   # vision model (frame understanding)")
    print(
        "Note: a VLM that's still down only means analyses run in degraded "
        "(sidecar-only) mode — they don't fail. See the README."
    )

    all_ok = asr_ok and vlm_ok and coder_ok
    if args.strict and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
