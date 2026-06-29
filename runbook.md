# Framesleuth Runbook

**Setup, health checks, troubleshooting, and common operational tasks.**

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Initial setup](#initial-setup)
3. [Model downloading](#model-downloading)
4. [Starting services](#starting-services)
5. [Health checks](#health-checks)
6. [Common issues](#common-issues)
7. [Testing the system](#testing-the-system)
8. [Logs and debugging](#logs-and-debugging)

---

## Prerequisites

### macOS

```bash
brew install python@3.11 ffmpeg
# Install uv for Python management
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
# Download Python 3.11 from python.org
# Install ffmpeg via scoop or download from ffmpeg.org
scoop install ffmpeg
# Install uv
irm https://astral.sh/uv/install.ps1 | iex
```

### NVIDIA GPU support (Linux)

If you plan to use vLLM for higher throughput:
```bash
sudo apt install -y nvidia-cuda-toolkit nvidia-utils
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Initial setup

### 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/framesleuth.git
cd framesleuth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Optional extras** (each is a no-op when absent — install only what you need):

| Extra | Install | Enables |
|---|---|---|
| `render` | `uv pip install -e ".[render]"` | HTML→video (Playwright + Chromium; `ffmpeg` on PATH) |
| `ocr` | `uv pip install -e ".[ocr]"` | Dedicated OCR backstop on error frames (needs the `tesseract` binary) |
| `all` | `uv pip install -e ".[all]"` | dev + render + ocr |

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your preferences
nano .env
```

Key settings:
- `ENGINE_PROFILE`: `local-default` (Ollama+llama.cpp) or `local-onestack` (all llama.cpp)
- Model URLs and model names
- Storage paths
- Concurrency limits

### 3. Validate configuration

```bash
python -c "from framesleuth.config import get_settings; s = get_settings(); s.validate_paths(); print('✓ Config OK')"
```

---

## Model downloading

### One-time setup

```bash
python scripts/download_models.py
```

This script **pre-fetches the faster-whisper ASR model** (the only weight
Framesleuth downloads itself) and then **checks that your VLM and coder servers
are reachable**, printing the exact `ollama pull ...` command for anything that's
missing. It does **not** download the vision/coder LLMs — those are served by your
engine. Useful flags: `--skip-asr`, `--asr-model <size>`, `--strict` (exit non-zero
if a prerequisite is missing, e.g. for CI).

Fetch the LLMs via the engine you use:
- VLM (llama.cpp): `llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF ...` (pulls on first run)
- VLM (Ollama path): `ollama pull qwen2.5vl`
- Coder (Ollama): `ollama pull qwen2.5-coder:7b`
- Whisper: pre-fetched by this script, or downloaded automatically by `faster-whisper`
  on the first transcription

### What gets downloaded

Default (Ollama) — pulled automatically by `docker compose up`, or via `ollama pull`:

| Model | Size | Purpose | Where |
|---|---|---|---|
| `qwen2.5vl` | ~6GB | Frame understanding (vision) | Ollama store |
| `qwen2.5-coder:7b` | ~5GB | Code fixing (coder) | Ollama store |
| faster-whisper (`small`) | ~0.5GB | Speech-to-text | `~/.cache/huggingface/` |

**Total: ~11–12GB.** Alternative (llama.cpp): `Qwen3-VL-8B-Instruct-GGUF` (~8GB) +
its `mmproj` (~2GB) under `~/.cache/huggingface/`, pulled on first `llama-server` run.

---

## Starting services

### Option A: Docker Compose — one command (recommended)

```bash
docker compose up          # or: ./scripts/dev_up.sh  (run it, don't `source` it)
```

This brings up three services with **no manual steps**:

1. `ollama` — the local model server (healthchecked).
2. `ollama-init` — a one-shot container that pulls `qwen2.5vl` + `qwen2.5-coder:7b`
   (~11 GB on first run) into a named volume, then exits.
3. `backend` — the Framesleuth API on `http://127.0.0.1:8010`, started only once
   the models are ready (`depends_on: service_completed_successfully`).

```bash
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool   # status: healthy
docker compose logs -f                  # watch the model download / startup
docker compose down --remove-orphans    # stop  (add -v to also delete model volumes)
```

The compose Ollama runs on the internal network only (its port is **not**
published), so it won't collide with a native Ollama on `:11434`. The only host
port is the API on `:8010`. If you already run Ollama natively with the models,
prefer **Option B** below — it reuses them instead of downloading a second copy.

It's **CPU-only** under Docker (incl. Docker on macOS, which can't use the Mac
GPU), so the vision model is slow there — for speed on macOS use **Option B**
(native Ollama). For an **NVIDIA GPU** on Linux, uncomment the `deploy:` block on
the `ollama` service in `docker-compose.yml`.

### Option B: Run it directly (fastest on macOS, best for development)

```bash
# Models — native Ollama uses the Mac GPU
ollama serve &                                    # skip if already running
ollama pull qwen2.5vl && ollama pull qwen2.5-coder:7b

# Backend — point the VLM at Ollama (in .env or inline) and start the API
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
VLM_URL=http://127.0.0.1:11434 VLM_MODEL=qwen2.5vl framesleuth-api

# MCP server (launched by your editor; or run standalone over stdio)
framesleuth-mcp
```

> The vision model is engine-agnostic — it only needs an OpenAI-compatible
> `/v1/chat/completions` endpoint that accepts images, which Ollama provides. Set
> `VLM_URL`/`VLM_MODEL` to point at it. Without a reachable VLM the `understand`
> stage degrades and the bundle relies on browser sidecars.

---

## Health checks

### Backend health

```bash
curl http://127.0.0.1:8010/v1/healthz
```

Expected response (all services up):
```json
{
  "status": "healthy",
  "services": {
    "vlm": { "name": "vlm", "status": "ready", "latency_ms": null, "error": null },
    "coder": { "name": "coder", "status": "ready", "latency_ms": null, "error": null },
    "storage": { "name": "storage", "status": "ready", "latency_ms": null, "error": null }
  },
  "queue_depth": 0,
  "timestamp": "2026-06-22T08:25:19.936358+00:00",
  "render": {
    "playwright": true, "playwright_version": "1.49.0", "chromium": true,
    "ffmpeg": true, "python": "/path/to/.venv/bin/python", "ready": true, "hint": null
  }
}
```

The `render` block reports readiness of the optional HTML→video capability in the
running process. When `ready` is `false`, `hint` says what to install and `python`
shows which interpreter the server is using (see the HTML→video troubleshooting
entry below). `queue_depth` reflects the number of jobs currently queued or running
(bounded by `MAX_CONCURRENT_JOBS`).

Without the model servers running you'll instead see `"status": "unhealthy"`
with `vlm`/`coder` reporting `"status": "unavailable"` and `storage: ready`.
That is expected — analysis still runs in degraded (sidecar-only) mode, and the
resulting bundle's `analysis_quality` records what was skipped.

### Model servers

```bash
# Default: vision + coder both on Ollama
curl http://127.0.0.1:11434/api/tags        # should list qwen2.5vl and qwen2.5-coder

# Alternative: llama.cpp VLM (only if you use VLM_URL=http://127.0.0.1:8080)
curl http://127.0.0.1:8080/v1/models
```

### MCP server

```bash
# Check logs
tail -f .framesleuth-mcp.log
```

---

## Common issues

> **First, run the doctor.** `python3 scripts/doctor.py` checks your whole local
> setup (venv, PATH, package import, ffmpeg/render prerequisites, backend + model
> servers) and prints a one-line fix for each problem. It uses only the standard
> library, so it runs with a plain `python3` even when the virtualenv is broken.

### Issue: `command not found: framesleuth-api` / `uv pip install` can't find the interpreter

**Cause:** Your shell has a virtualenv "active" (the `(framesleuth)` prefix) whose
directory was deleted or moved — so the console scripts and `uv` point at a Python
that no longer exists. (`uv` says: *Python interpreter not found at …/.venv/bin/python3*.)

**Fix:** drop the dead venv and recreate it **in the framesleuth directory**:
```bash
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd /path/to/framesleuth          # the agent repo — not the website folder
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"       # add ,render for HTML→video: ".[dev,render]"
framesleuth-api                  # now on PATH
```

### Issue: `docker compose up` → "Bind for 0.0.0.0:11434 failed: port is already allocated" or "orphan containers"

**Cause:** something else already owns a port (commonly a **native Ollama** on
`:11434`, or **leftover containers from an older version** of this compose file).

**Fix:** the current compose no longer publishes Ollama's port, so update + clean up:
```bash
git pull                                   # get the current docker-compose.yml
docker compose down --remove-orphans       # remove old/renamed containers
docker compose up -d --remove-orphans
```
If you already run Ollama natively with the models, skip Docker entirely and use
**Option B** (run it directly) — it reuses your existing Ollama. The only host
port the stack needs is the API on `:8010`; if `:8010` is taken, change the left
side of `"8010:8000"` in `docker-compose.yml`.

### Issue: "VLM server not responding (503)"

**Cause:** llama.cpp not running or model not loaded.

**Fix:**
```bash
# Check if running
lsof -i :8080

# Restart and wait for model load (~1-2 min)
llama-server ... --n-gpu-layers 99 -c 32768 --port 8080
# Wait for "llama_server: server is listening"
```

### Issue: "Coder unavailable (503)"

**Cause:** Ollama not running or model not pulled.

**Fix:**
```bash
# Check if running
pgrep ollama || OLLAMA_KEEP_ALIVE=-1 ollama serve

# Pull model
ollama pull qwen2.5-coder:7b
```

### Issue: "Upload too large (413)"

**Cause:** Video exceeds `MAX_UPLOAD_MB` limit.

**Fix:**
- Increase limit in `.env` (`MAX_UPLOAD_MB=1024`)
- Or record shorter videos

### Issue: "Out of memory (OOM)"

**Cause:** Models competing for VRAM.

**Fix:**
- Reduce `MAX_CONCURRENT_JOBS` in `.env` (default 2)
- Use smaller quant (8B instead of 13B)
- Enable GPU offloading with `--n-gpu-layers 99`

### Issue: "FFmpeg: No such file"

**Cause:** ffmpeg not installed or not on PATH.

**Fix:**
```bash
# Install
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux

# Verify
ffmpeg -version
```

### Issue: HTML→video "Playwright is not installed" (or `503` from `/v1/render-html`)

**Why optional?** Playwright is an optional `[render]` extra — it pulls a ~150 MB
Chromium browser the core pipeline doesn't need, so it isn't a core dependency.
The error means the extra isn't in **the process the server runs** — usually the
server is a different environment than the one you installed into, or it wasn't
restarted. (Docker bundles it already; this is the direct path.)

**Fix:**
```bash
# Install the extra into the SAME environment the server runs in:
uv pip install -e ".[render]"        # or ".[all]" = dev + render
# ffmpeg must also be on PATH (see above)

# Restart framesleuth-api. Chromium downloads automatically on the first render —
# no separate `playwright install chromium` needed. Verify in the running process:
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool
# → "render": {"playwright": true, "ffmpeg": true, ...}  (chromium flips true after first render)
```

If `render.playwright` is still `false`, the server is the wrong environment —
check `render.python` (the interpreter it actually uses) and reinstall there. To
provision the browser yourself instead of auto-download, set
`FRAMESLEUTH_AUTO_INSTALL_BROWSER=0` and run `playwright install chromium`.

---

## Testing the system

### Run unit tests

```bash
pytest tests/ -v --cov=framesleuth
```

### Run integration tests (requires services)

```bash
pytest tests/ -v -m integration
```

### Create a fixture report (end-to-end)

Analysis is **asynchronous**: `POST /v1/analyze` returns `202` with a `job_id`
immediately and runs the pipeline in the background. Poll `/v1/jobs/{id}` until
`state` is `done`, then read the bundle.

```bash
# 1. Queue a sample (returns 202 {job_id, status: "queued"}):
JOB=$(curl -s -X POST http://127.0.0.1:8010/v1/analyze \
  -F "video=@samples/flash_bug.mp4" \
  -F "intent=Find why the Save button hangs and fix it." \
  | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$JOB"

# 2. Poll until the job reports state: done (or failed):
curl -s http://127.0.0.1:8010/v1/jobs/$JOB | python -m json.tool

# 3. Read the Context Bundle:
curl -s http://127.0.0.1:8010/v1/report/$JOB | python -m json.tool
```

> Idempotency: re-posting the *same bytes* returns the existing job
> (`idempotent: true`) without re-running. Use a different file, or clear the
> scratch store (`rm -rf bug-reports/*`) to force a fresh run.

**Follow progress without polling (SSE), or cancel a run:**

```bash
# Stream live progress until the job is done/failed/cancelled:
curl -N http://127.0.0.1:8010/v1/jobs/$JOB/events

# Cooperatively cancel a still-running job (it stops at the next stage boundary):
curl -X DELETE http://127.0.0.1:8010/v1/jobs/$JOB        # -> {state, cancel_requested: true}
```

Set `WEBHOOK_URL` in `.env` to have the backend POST a compact `{id, state, title,
action}` payload to your endpoint when a job finishes — no polling needed.

Prefer a GUI? Import the Postman collection in [`postman/`](postman/README.md).

---

## Logs and debugging

### Backend logs

**Real-time:**
```bash
tail -f bug-reports/jobs.db-journal
# or in the backend terminal
```

**Per-job:**
```bash
# After a job completes, check:
cat bug-reports/{job-id}/job.log
cat bug-reports/{job-id}/metrics.json
```

### Model server logs

**llama.cpp:**
```bash
# Logs to stdout; check terminal where llama-server started
```

**Ollama:**
```bash
# macOS: ~/Library/Application Support/Ollama/logs
# Linux: ~/.ollama/logs
```

### Debug mode

```bash
LOG_LEVEL=DEBUG framesleuth-api
```

---

## Configuration tuning

### For low-end hardware (4GB RAM)

```env
MAX_CONCURRENT_JOBS=1
FRAME_LOWRES_HEIGHT=360
MAX_FRAMES_PER_MIN=15
VLM_TIMEOUT_S=120  # Give more time
```

### For production (shared GPU)

```env
ENGINE_PROFILE=server
MAX_CONCURRENT_JOBS=4
FRAME_LOWRES_HEIGHT=480
MAX_FRAMES_PER_MIN=60
CLASSIFY_CONFIDENCE_THRESHOLD=0.8  # Higher bar
BUNDLE_TTL_DAYS=14                 # Purge old bundles on startup so disk stays bounded
WEBHOOK_URL=https://your.app/hooks/framesleuth   # Notify on completion instead of polling
```

### Quality / cost knobs (all optional, on by default)

```env
KEYFRAME_DEDUP=true        # Drop near-identical frames before the VLM (saves budget)
ASR_VAD_FILTER=true        # Voice-activity filter — fewer hallucinated transcript lines
ASR_LANGUAGE=              # Force a language (ISO code), e.g. en; blank = auto-detect
OVERLAY_INTERACTIONS=true  # Draw a marker where a click/cursor sidecar landed (if it has coords)
OCR_BACKSTOP=true          # Second OCR read on error frames (needs the `ocr` extra + tesseract)
REDACT_PII=true            # Scrub emails / cards / SSNs / phones / cloud keys, on top of secrets
GROUNDING_MAX_FILES=5000   # Cap the workspace scan so large monorepos stay bounded
```

---

## Next steps

- [Capabilities](docs/capabilities.md) — full reference for inputs, outputs, endpoints, and MCP tools
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md) — connect an MCP client
- [Postman Collection](postman/README.md) — exercise the HTTP API end-to-end
