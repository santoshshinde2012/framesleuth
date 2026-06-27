#!/usr/bin/env bash
#
# Start the whole Framesleuth stack with one command (Docker Compose).
# RUN it, do not SOURCE it:  ./scripts/dev_up.sh
#
# (Sourcing would run `set -e` in your interactive shell, so any failed command
#  — e.g. Docker not installed — would close your terminal.)

# Refuse to be sourced: return without touching the parent shell's options.
if (return 0 2>/dev/null); then
  echo "Don't source this script — run it:  ./scripts/dev_up.sh" >&2
  return 1 2>/dev/null || exit 1
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found — this script starts everything with Docker Compose."
  echo
  echo "No Docker? Run the agent directly (README → 'Run it directly'):"
  echo "  ollama serve & ; ollama pull qwen2.5vl && ollama pull qwen2.5-coder:7b"
  echo "  uv venv && source .venv/bin/activate && uv pip install -e \".[dev]\""
  echo "  VLM_URL=http://127.0.0.1:11434 VLM_MODEL=qwen2.5vl framesleuth-api"
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" && -z "${FRAMESLEUTH_FORCE_DOCKER:-}" ]]; then
  echo "Heads up: Docker on macOS runs models on CPU (Docker can't use the Mac GPU),"
  echo "so the vision model will be slow. For speed on macOS, use the native Ollama"
  echo "app + run the agent directly (no Docker) — see the README."
  echo
  read -r -p "Start the Docker stack anyway? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted — see the README 'Run it directly' path."; exit 0; }
fi

echo "Starting Framesleuth — first run pulls the models (~11 GB), so it can take a while."
# --remove-orphans cleans up containers from older versions of this compose file.
docker compose up -d --remove-orphans

echo
echo "Bringing models + backend up. Waiting for the API to report healthy…"
for _ in $(seq 1 180); do
  if curl -fs http://127.0.0.1:8010/v1/healthz >/dev/null 2>&1; then
    echo "✓ Backend is up: http://127.0.0.1:8010/v1/healthz"
    echo
    echo "Test it:"
    echo "  curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool"
    echo "  # then POST a recording to /v1/analyze (see README / Postman)"
    echo
    echo "Follow logs:  docker compose logs -f"
    echo "Stop:         docker compose down       (add -v to also delete model volumes)"
    exit 0
  fi
  sleep 5
done

echo "Still starting (models may still be downloading). Check progress with:"
echo "  docker compose logs -f"
echo "  curl -s http://127.0.0.1:8010/v1/healthz"
