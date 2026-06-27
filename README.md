<p align="center">
  <img src="docs/logo/framesleuth-logo-256.png" alt="Framesleuth" width="128" height="128" />
</p>

# Framesleuth

**Local bug-reproduction video analysis, exposed over MCP.**

Framesleuth takes a bug-recording video (plus optional browser sidecars), understands it
frame-by-frame, and produces a structured **Bug Context Bundle**. It is **MCP-ready**, so
any MCP client — a VS Code agent, another coding agent, or a custom system — can drive the
analysis and consume the result to fix the bug directly.

Capture happens outside this repo: any screen recording works, or a browser capture
extension can record the bug and post the video + sidecars to this agent's local API.
This repo is the analysis agent only.

Everything runs locally. Nothing leaves your machine.

## Quick start

> **Want to fix a bug from a video inside VS Code?** Follow
> [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect
> the bundled MCP server and go from a recording to a grounded fix.

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

Once the API reports healthy (either setup path), go from a recording to a Bug
Context Bundle in three calls — analysis is async (submit → poll → read):

```bash
# 1. Submit any screen recording (mp4/webm). Returns 202 { job_id, ... }
JOB=$(curl -s -F "video=@bug.mp4" http://127.0.0.1:8010/v1/analyze \
  | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll until state is "done" (queued → running → done)
curl -s "http://127.0.0.1:8010/v1/jobs/$JOB" | python -m json.tool

# 3. Read the Bug Context Bundle
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
reachable, Framesleuth **degrades gracefully** — it still produces a valid Bug
Context Bundle from the browser sidecars (console errors, failed requests, clicks)
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
Bug video (mp4/webm) + sidecars
    ↓
Local Analysis Service (pipeline)
    ├─ Preprocess (PyAV: duration/fps/dims)
    ├─ Transcript (faster-whisper)
    ├─ Keyframes (visual-delta change scoring)
    ├─ Understanding (local vision model — Qwen2.5-VL by default)
    ├─ Fusion + Classification
    ├─ Extraction → Bug Context Bundle
    ├─ Summarize (skill/system-prompt-driven)
    └─ Grounding (workspace search)
    ↓
Bug Context Bundle
    ↓
MCP server + local HTTP API
    └─ consumed by any MCP client (VS Code agent, other agents, capture extension)
```

## Features

- **Frame-by-frame understanding** using a local vision model (Qwen2.5-VL by default; engine-agnostic)
- **Automatic keyframe selection** via frame-to-frame visual-delta change scoring
- **Error detection and extraction** from console, OCR, and UI state
- **Redaction-first design** — sensitive data (passwords, tokens) redacted before models see it
- **No data leaves your machine** — fully local, no telemetry or cloud APIs
- **Engine-agnostic** — swap Ollama, llama.cpp, or vLLM via config only
- **Structured output** — canonical Bug Context Bundle with evidence citations
- **Configurable response** — pick a summary **skill** *and* an **action mode**
  (`fix`/`explain`/`triage`/`test`/`report`/`reproduce`, auto-picked from the
  classification), plus a machine-readable `suggested_actions` menu and on-demand
  artifact renderers (markdown / GitHub issue / test plan)
- **Resilient** — handles no-audio videos, weak local models, low-confidence cases
- **HTML → video** — render a self-contained HTML animation (CSS/JS/canvas) to
  MP4, GIF, or WebM via the `render_html_video` MCP tool or `POST /v1/render-html`.
  **Included by default in the Docker image** (real headless Chromium + ffmpeg, the
  highest-fidelity path). For the direct (non-Docker) path, add the `render` extra
  (see below); returns `503` with an actionable message when unavailable.

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
│   ├── schemas.py           # Data contracts (Bug Context Bundle, enums)
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
│   └── mcp_server/          # videobug MCP server (VS Code + any MCP client)
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
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect the `videobug` MCP server to Copilot, Claude Code, and Claude Desktop
- [Web App Integration (end-to-end)](docs/web-integration.md) — embed Framesleuth behind your own backend with an agent loop
- [Postman Collection](postman/README.md) — exercise the HTTP API end-to-end (import or run headless with Newman)
- [Runbook & Troubleshooting](runbook.md) — setup, health checks, and common issues

## License

Apache-2.0

---

## Capture client

Bug capture lives outside this repo. Any screen recording works — drive the agent directly
with your own video file. A browser capture extension can also record the bug, collect
browser sidecars (console errors, failed requests, clicks), and post the video + sidecars
to this agent's local API. The agent's CORS is already scoped to `chrome-extension://`
origins and the loopback bind, so an extension works against a locally running backend with
no extra setup.

**Status:** Backend + pipeline + MCP server completed.  
**Questions?** Open an issue or check [runbook.md](runbook.md) for common questions.
