FROM python:3.11-slim

WORKDIR /app

# System dependencies.
#   ffmpeg            — encodes the HTML→video output (mp4/gif) and helps media decode
#   libgl1/libglib2.0 — required by opencv-python at import time (cv2 needs libGL)
#   build-essential   — only needed to compile any wheels that lack a binary build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project (a .dockerignore keeps local state — .venv, .git, bug-reports — out).
COPY . .

# Install the package. Render support (HTML→video via Playwright + headless
# Chromium) is ON by default so the feature works out of the box; build with
# --build-arg INSTALL_RENDER=false for a slimmer image without it.
ARG INSTALL_RENDER=true
RUN if [ "$INSTALL_RENDER" = "true" ]; then \
        pip install --no-cache-dir -e ".[render]" && \
        playwright install --with-deps chromium; \
    else \
        pip install --no-cache-dir -e .; \
    fi

# Runtime output directory (also bind-mounted by docker-compose).
RUN mkdir -p bug-reports

# Liveness: the API returns 200 even in degraded/sidecar-only mode.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/healthz')"

# The API listens on 8000 inside the container (compose maps host 8010 → 8000).
EXPOSE 8000
CMD ["uvicorn", "framesleuth.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
