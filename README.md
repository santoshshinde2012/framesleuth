<p align="center">
  <img src="docs/logo/framesleuth-logo-256.png" alt="Framesleuth" width="128" height="128" />
</p>

# Framesleuth

**Local video → structured context for coding agents, exposed over MCP.**

Give Framesleuth *any* video — a bug recording, a feature demo, a design walkthrough, a
Loom, a phone capture — and it understands it frame-by-frame (plus optional browser
sidecars) and produces a structured **Context Bundle**. It is **MCP-ready**, so any MCP
client — a VS Code agent, another coding agent, or a custom system — can drive the
analysis and consume the result to **fix a bug, add or change a feature, or build a whole
new feature/app** grounded in what the video actually shows.

Capture happens outside this repo: any video works, or a browser capture extension can
record a session and post the video + sidecars to this agent's local API. This repo is the
analysis agent only.

Everything runs locally. Nothing leaves your machine.

## Quick start

> **Want to go from a video to a grounded change inside VS Code?** Follow
> [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect
> the bundled MCP server and turn a recording into a fix, a feature, or a new build.

### Fastest: one command with Docker

Everything — the model server, the models, and the API — comes up with a single
command. No Python, no virtualenv, no manual model setup.

```bash
git clone https://github.com/santoshshinde2012/framesleuth.git
cd framesleuth
docker compose up            # or: ./scripts/dev_up.sh
```

The **first** run automatically pulls the vision + coder models (`qwen2.5vl` and
`qwen2.5-coder:7b`, ~11 GB total) into a Docker volume, then starts the backend on
`http://127.0.0.1:8010`. Subsequent runs are instant. It's ready when the health
check reports `healthy`:

```bash
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool   # "status": "healthy"
```

That's the whole setup — **run your first analysis** (below), or connect the MCP
server in your editor ([VS Code & Claude](docs/use-with-vscode-and-claude.md)).

```bash
docker compose logs -f                  # follow progress / model download
docker compose down --remove-orphans    # stop  (add -v to also delete model volumes)
```

The stack runs its **own** Ollama on the internal Docker network only (its port is
not published), so it never clashes with a native Ollama you may already run on
`:11434` — the only host port is the API on `:8010`.

> **Already run Ollama natively (with the models)?** The Docker stack's Ollama is
> separate and would re-download them. Skip Docker and use the **direct path**
> below instead — it reuses your existing Ollama and is faster (especially on
> macOS, where Docker can't use the GPU).
>
> **macOS / no GPU:** Docker runs the models on **CPU**, so the vision model is
> slow. **NVIDIA GPU on Linux:** uncomment the `deploy:` block on the `ollama`
> service in `docker-compose.yml` for acceleration.

### Run your first analysis (curl)

Once the API reports healthy (either setup path), go from a video to a Context
Bundle in three calls — analysis is async (submit → poll → read):

```bash
# 1. Submit any screen recording (mp4/webm). Returns 202 { job_id, ... }
JOB=$(curl -s -F "video=@bug.mp4" http://127.0.0.1:8010/v1/analyze \
  | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll until state is "done" (queued → running → done)
curl -s "http://127.0.0.1:8010/v1/jobs/$JOB" | python -m json.tool

# 3. Read the Context Bundle
curl -s "http://127.0.0.1:8010/v1/report/$JOB" | python -m json.tool
```

Optional form fields on step 1: `-F intent="why does save hang?"`, `-F skill=bug_report`,
`-F action=fix` (`GET /v1/skills` and `/v1/actions` list the choices). Prefer a UI?
Import the [Postman collection](postman/README.md) — it chains these calls for you.

### Run it directly (no Docker — fastest on macOS, best for development)

**Prerequisites:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/), 8 GB+ RAM, and
a local model server. ffmpeg is **not** required (PyAV bundles its own; `ffprobe`,
if present, is used opportunistically to detect an audio stream).

```bash
git clone https://github.com/santoshshinde2012/framesleuth.git
cd framesleuth

# 1. Models — native Ollama (uses the Mac GPU) is the quick path
ollama serve &                                  # skip if already running
ollama pull qwen2.5vl && ollama pull qwen2.5-coder:7b

# 2. Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/download_models.py               # optional: pre-warm ASR + check servers

# 3. Configure + start the API (binds 127.0.0.1:8010)
cp .env.example .env                            # already defaults to the Ollama path above
framesleuth-api                                 # or: uvicorn framesleuth.service.api:app --port 8010

# 4. Verify
curl -s http://127.0.0.1:11434/v1/models | grep -q qwen2.5vl && echo "VLM ready"
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool   # status: healthy, vlm: ready
```

When `/v1/healthz` shows `vlm: ready`, recordings analyze with a real
classification (`analysis_quality.level` = `full`/`partial`). With no vision model
reachable, Framesleuth **degrades gracefully** — it still produces a valid Context
Bundle from the browser sidecars (console errors, failed requests, clicks)
and records what was thin in `analysis_quality`. Record **with narration** so the
audio transcript (`asr`) stage contributes too.

> **Something not working?** Run the setup doctor — it works with a plain
> `python3` even when your virtualenv is broken, and prints a one-line fix for
> each problem (stale/missing venv, `framesleuth-api` not on PATH, ffmpeg/render
> prerequisites, backend or model server not reachable, wrong `VLM_URL`):
>
> ```bash
> python3 scripts/doctor.py
> ```
>
> Common gotcha: `command not found: framesleuth-api` or a `uv pip install` error
> about a missing interpreter means your active venv was deleted/moved. Fix it
> from the **framesleuth** directory:
> `deactivate; unset VIRTUAL_ENV; uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"`.

**Stop**

```bash
# Stop the backend: Ctrl+C in its terminal, or
pkill -f framesleuth-api

# Stop Ollama (optional — leaving it running keeps the model warm)
pkill -f "ollama serve"              # macOS app users: quit Ollama from the menu bar
```

## Architecture

```
Any video (mp4/webm) + optional sidecars
    ↓
Local Analysis Service (pipeline)
    ├─ Preprocess (PyAV: duration/fps/dims)
    ├─ Transcript (faster-whisper)
    ├─ Keyframes (visual-delta change scoring)
    ├─ Understanding (local vision model — Qwen2.5-VL by default)
    ├─ Fusion + Classification
    ├─ Extraction → Context Bundle
    ├─ Summarize (skill/system-prompt-driven)
    └─ Grounding (workspace search)
    ↓
Context Bundle
    ↓
MCP server + local HTTP API
    └─ consumed by any MCP client (VS Code agent, other agents, capture extension)
```

## Features

- **Frame-by-frame understanding** using a local vision model (Qwen2.5-VL by default; engine-agnostic)
- **Adaptive keyframe selection** — coverage-binned, visual-salience-ranked (AKS-style), with a build-aware budget for feature/design videos and **perceptual-hash dedup** that drops near-identical frames so the VLM budget is spent on distinct content
- **Bug *and* build** — a `feature` class plus a structured **build context** (screens, UI components, a screen-to-screen user flow, design notes, and where to implement) so an agent can implement, not just diagnose
- **Error detection and extraction** from console, OCR, and UI state
- **Corpus-aware grounding** — error symbols *or* feature/UI nouns → ranked `file:line` (definitions preferred, distinctive symbols weighted via IDF + whole-word match), respecting `.gitignore` and bounded for large repos
- **Trust signals** — per-field confidence (with **cross-modal corroboration** — agreeing signals reinforce each other) and a task-aware `actionability` (ready/thin/insufficient) alongside the pipeline quality level
- **Redaction-first design** — secrets (passwords, tokens, keys) **and PII** (emails, Luhn-valid card numbers, SSNs/phones, cloud keys) redacted before models see it
- **Observability** — per-stage timings on every bundle (`stage_timings`) and live on `GET /v1/jobs/{id}`, so you can see where analysis time went
- **Job lifecycle & delivery** — cooperative **cancellation** (`DELETE /v1/jobs/{id}`), **SSE progress** (`GET /v1/jobs/{id}/events`), a completion **webhook** (`WEBHOOK_URL`), real queue depth in `/healthz`, and **TTL retention** cleanup (`BUNDLE_TTL_DAYS`)
- **Interaction overlay** — a click/cursor sidecar with coordinates draws a marker on the matching keyframe, so the model sees *where* the user acted
- **Cleaner transcripts** — faster-whisper voice-activity filtering (`ASR_VAD_FILTER`) drops silence before decoding; detected/forced language is recorded
- **OCR backstop** *(optional `ocr` extra)* — a sparse VLM OCR on an error frame gets a second, independent Tesseract reading; a no-op without the extra
- **No data leaves your machine** — fully local, no telemetry or cloud APIs
- **Engine-agnostic** — swap Ollama, llama.cpp, or vLLM via config only
- **Works on *any* video** — not just bug recordings. A general video (a demo, a
  walkthrough, a talk, a phone/real-world clip) yields a faithful **summary + a
  timeline of key moments** (`summary`, `key_moments[]`) instead of being forced
  into a bug shape; the bug-only fields (severity, expected/actual, repro steps)
  stay `null` rather than carrying fabricated placeholders
- **Structured output** — canonical Context Bundle with evidence citations
- **Configurable response** — pick a summary **skill** *and* an **action mode**
  (`fix`/`implement`/`design`/`summarize`/`explain`/`triage`/`test`/`report`/`reproduce`,
  auto-picked from the classification), plus a machine-readable `suggested_actions`
  menu and on-demand artifact renderers (markdown / GitHub issue / test plan)
- **Eval harness** — model-free classification / grounding / citation / **faithfulness**
  suites (`python scripts/eval_harness.py --behavioral`) gate quality in CI; the
  faithfulness suite proves every emitted key moment and step cites real, resolvable
  evidence (no fabrication)
- **Resilient** — handles no-audio videos, weak local models, low-confidence cases
- **HTML → video (frame-by-frame, whole animation)** — turn a self-contained HTML
  animation (CSS/JS/canvas) into MP4, GIF, or WebM via the `render_html_video` MCP
  tool or `POST /v1/render-html`. Captures the animation **frame-by-frame** under a
  paused virtual clock and encodes a color-correct H.264 MP4 (`yuv420p`+`bt709`,
  near-lossless) — **full color, no dropped frames, no quality loss** (up to 4K,
  5–60 fps). **Omit the duration and the *whole* animation is captured** — its
  length is auto-detected (CSS animations/transitions, Web Animations API, or a
  `window.__renderDurationMs` hint for canvas loops); bounded by
  `RENDER_MAX_DURATION_S` / `RENDER_MAX_FRAMES` (raise for very long clips).
  **Included by default in the Docker image** (headless Chromium + ffmpeg). For the
  direct (non-Docker) path, add the `render` extra (see below); returns `503` with
  an actionable message when unavailable.

### Enable & troubleshoot HTML → video

> Using **Docker** (`docker compose up`)? HTML→video already works — the image bakes
> in Playwright + Chromium + ffmpeg. (Build with `--build-arg INSTALL_RENDER=false`
> for a slimmer image without it.) The steps below are for the **direct** path.

**Why is Playwright not in the core install?** It's an **optional `[render]`
extra**, not a core dependency, because it pulls a ~150 MB headless-Chromium
browser the core video→bundle pipeline never needs — the standard way to ship a
heavy, feature-specific dependency. (`av`, `opencv`, `faster-whisper` are core
because the pipeline requires them.) Install the extra and you're done — **the
Chromium build downloads automatically on your first render**, so there's no
separate `playwright install chromium` step:

```bash
# In the same environment the server runs in:
uv pip install -e ".[render]"        # or ".[all]" = dev + render
# ffmpeg must be on PATH (brew install ffmpeg / apt-get install ffmpeg)

# Restart framesleuth-api, then verify (Chromium fetches itself on first render):
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool
# → "render": {"playwright": true, "chromium": <true after first render>, "ffmpeg": true}
```

Set `FRAMESLEUTH_AUTO_INSTALL_BROWSER=0` to disable the auto-download and run
`playwright install chromium` yourself (e.g. in a locked-down environment).

> **Other optional extra — `ocr`.** For the dedicated OCR backstop on error frames,
> `uv pip install -e ".[ocr]"` and put the `tesseract` binary on PATH
> (`brew install tesseract` / `apt-get install tesseract-ocr`). It's a no-op when
> absent — the VLM still does OCR; the backstop only adds a second reading. Use
> `".[all]"` for dev + render + ocr.

If `render.ready` is `false`, the `render.hint` field tells you exactly what's
missing. The most common cause of *"Playwright is not installed"* despite
following the steps is that `framesleuth-api` is running from a **different
environment** than the one you installed into (the `render.python` field shows
which interpreter the server uses) — or the server simply wasn't restarted.

## Project structure

```
framesleuth/
├── framesleuth/              # Main package
│   ├── config.py            # Typed config (pydantic-settings)
│   ├── schemas.py           # Data contracts (Context Bundle, enums)
│   ├── errors.py            # Exception taxonomy
│   ├── logging_config.py    # Structured JSON logging, job-id correlation
│   ├── prompts.py           # VLM / classify / summary / fix prompt templates
│   ├── skills.py            # Built-in summary skills (summary, bug_report, ...)
│   ├── actions.py           # Action modes (fix/explain/triage/...) + suggested-actions menu
│   ├── render.py            # Artifact renderers (markdown / GitHub issue / test plan)
│   ├── clients/             # VLM, coder HTTP clients (OpenAI-compatible)
│   ├── pipeline/            # preprocess, asr, scenes, understand, fusion, classify, bug_extract, redact, summarize, sidecars, grounding, html_render
│   ├── orchestrator/        # graph.py — linear async stage pipeline
│   ├── jobs/                # store.py — SQLite job state + bundle index
│   ├── service/             # FastAPI HTTP endpoints
│   └── mcp_server/          # framesleuth MCP server (VS Code + any MCP client)
├── tests/                   # pytest tests + fixtures
├── scripts/                 # doctor.py (setup check), download_models.py, dev_up.sh
├── postman/                 # HTTP API collection + environment
├── docs/                    # capabilities, use-with-vscode-and-claude, web-integration
└── pyproject.toml           # Dependencies and tool config
```

## Development

### Run tests
```bash
pytest tests/ -v --cov=framesleuth
```

### Code quality
```bash
ruff check framesleuth tests
black --check framesleuth tests
mypy --strict framesleuth
```

### Set up pre-commit hooks
```bash
pre-commit install
```

A short, focused set:

- [Capabilities](docs/capabilities.md) — the single reference: every input, output, skill, action, renderer, HTTP endpoint, and MCP tool
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect the `framesleuth` MCP server to Copilot, Claude Code, and Claude Desktop
- [Web App Integration (end-to-end)](docs/web-integration.md) — embed Framesleuth behind your own backend with an agent loop
- [Postman Collection](postman/README.md) — exercise the HTTP API end-to-end (import or run headless with Newman)
- [Runbook & Troubleshooting](runbook.md) — setup, health checks, and common issues

## License

Apache-2.0

---

## Capture client

Bug capture lives outside this repo. Any screen recording works — drive the agent directly
with your own video file. A browser capture extension can also record a session, collect
browser sidecars (console errors, failed requests, clicks), and post the video + sidecars
to this agent's local API. CORS is allowlisted (`WEB_ORIGINS`, default: the hosted demo site
+ local dev) plus `chrome-extension://` origins, and the agent answers Chrome's Private
Network Access preflight — so both a capture extension and the "Try it" widget on
framesleuth.com work against a locally running backend with no extra setup. The agent stays
bound to loopback; CORS only controls which browser origins may read its responses.

**Status:** Backend + pipeline + MCP server completed.
**Questions?** Open an issue or check [runbook.md](runbook.md) for common questions.
