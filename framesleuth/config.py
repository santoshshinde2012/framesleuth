"""Configuration management for Framesleuth.

Uses pydantic-settings for environment-based configuration with type safety.
All configuration is immutable after initialization.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class EngineProfile(StrEnum):
    """Model serving engine profile selection.

    Follows the principle of configuration-driven flexibility without code changes.
    """

    LOCAL_DEFAULT = "local-default"  # llama.cpp (VLM) + Ollama (coder)
    LOCAL_ONESTACK = "local-onestack"  # llama.cpp for both
    SERVER = "server"  # vLLM for both (Linux+NVIDIA)


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment and .env file.

    All settings are typed, validated, and immutable. Follows the
    dependency inversion principle by centralizing configuration.
    """

    # === Server configuration ===
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8010  # 8000 left free for a capture backend, if you run one
    MCP_HOST: str = "127.0.0.1"
    MCP_PORT: int = 3001

    # === Model serving strategy ===
    ENGINE_PROFILE: EngineProfile = EngineProfile.LOCAL_DEFAULT
    VLM_URL: str = "http://127.0.0.1:8080"
    VLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    CODER_URL: str = "http://127.0.0.1:11434"
    CODER_MODEL: str = "qwen2.5-coder:7b"

    # === Model parameters ===
    VLM_TIMEOUT_S: float = 60.0
    VLM_MAX_RETRIES: int = 3
    CODER_TIMEOUT_S: float = 120.0
    CODER_MAX_RETRIES: int = 2

    # Per-frame generation budget. The frame-analysis JSON is small, so a tight
    # cap keeps decode latency low; the focused error retry is allowed more room
    # for long stack traces (see ``VLM_ERROR_MAX_TOKENS``).
    VLM_MAX_TOKENS: int = 768
    VLM_ERROR_MAX_TOKENS: int = 1024
    # How many keyframes to send to the VLM at once. Local single-GPU servers
    # serialize internally, so keep this modest (raise only when the engine is
    # configured for parallelism, e.g. Ollama OLLAMA_NUM_PARALLEL / llama.cpp --parallel).
    VLM_MAX_CONCURRENCY: int = 3
    # Request OpenAI-style JSON output (``response_format``). Supported by
    # llama.cpp, Ollama, and vLLM; disable for an engine that rejects the field.
    VLM_JSON_MODE: bool = True
    # Transcode frames to JPEG before sending to the VLM to cut upload bytes and
    # vision prefill tokens. Stored keyframes stay lossless PNG regardless.
    VLM_SEND_JPEG: bool = True
    VLM_JPEG_QUALITY: int = 85

    # === Summary synthesis ===
    # Text model that turns the fused video+audio timeline into the summary.
    # Leave URL/MODEL blank to reuse the VLM endpoint (a vision-language model
    # summarizes text fine), so no extra model server is required.
    SUMMARY_URL: str = ""
    SUMMARY_MODEL: str = ""
    SUMMARY_TIMEOUT_S: float = 120.0
    SUMMARY_MAX_TOKENS: int = 1024

    # === Classification and routing ===
    CLASSIFY_CONFIDENCE_THRESHOLD: float = 0.7
    # When the deterministic heuristic lands in the ambiguous band
    # ``[floor, threshold)`` — some signal, but not confident — the orchestrator
    # may (a) resample extra frames around the suspected failure and (b) ask the
    # model to break the tie. Below the floor there is no signal worth the call.
    CLASSIFY_AMBIGUOUS_FLOOR: float = 0.3
    # Break ambiguous-band ties with a model classification (uses the summary
    # endpoint). Deterministic heuristics still decide the confident cases.
    CLASSIFY_USE_MODEL: bool = True
    # Bounded agentic resample: on an ambiguous classification with visual
    # evidence, re-sample extra frames around error timestamps and re-classify,
    # at most this many times. 0 disables it (pure single-pass).
    MAX_RESAMPLE_RETRIES: int = 2

    # === Transcription ===
    # Whisper hallucinates short filler phrases ("Thank you.", "Bye.") on silent
    # or near-silent audio with a high no-speech probability. Drop any segment
    # below this confidence so the timeline is not polluted with phantom speech.
    ASR_MIN_CONFIDENCE: float = 0.5
    # Run faster-whisper's built-in (Silero) voice-activity filter so silence is
    # never fed to the decoder — the most effective guard against hallucinated
    # speech, on top of the confidence floor above.
    ASR_VAD_FILTER: bool = True
    # Force a transcription language (ISO code, e.g. "en"); blank = auto-detect.
    ASR_LANGUAGE: str = ""

    # === Upload and processing limits ===
    MAX_UPLOAD_MB: int = 512
    MAX_DURATION_S: int = 600  # 10 minutes
    MAX_FRAMES_PER_MIN: int = 30

    # === Frame processing ===
    # Frames are extracted/captioned at the low-res height to bound decode + VLM
    # cost. A suspected error frame with sparse OCR is re-decoded at the high-res
    # height (uncompressed) for a focused re-read, so tiny stack-trace text that a
    # 480p downscale would smear stays legible — "resolution where text lives".
    FRAME_LOWRES_HEIGHT: int = 480
    FRAME_HIGHRES_HEIGHT: int = 1080
    # Visual-delta score above which a frame is treated as a scene cut (fallback
    # keyframe selector). Higher = fewer, sharper cuts; lower = more sensitive.
    SCENE_CUT_THRESHOLD: float = 0.35
    # Collapse near-identical keyframes (held spinners, static title cards, repeated
    # screens) with a perceptual hash before the VLM, so the frame budget is spent
    # on distinct content. Conservative by default: only frames within
    # KEYFRAME_PHASH_HAMMING_MAX bits (out of 64) are treated as duplicates, so a
    # subtle but real on-screen change (e.g. an error appearing) is never dropped.
    KEYFRAME_DEDUP: bool = True
    KEYFRAME_PHASH_HAMMING_MAX: int = 4

    # Draw a marker on a keyframe when a click/cursor sidecar event with numeric
    # coordinates lands near it, so the VLM (and a human reviewer) can see exactly
    # where the user interacted. A no-op when events carry no coordinates.
    OVERLAY_INTERACTIONS: bool = True
    # Re-read suspected error frames with a dedicated OCR engine (pytesseract, the
    # optional ``[ocr]`` extra) when the VLM's OCR comes back sparse — a deterministic
    # backstop for tiny stack-trace text. A no-op when the extra is not installed.
    OCR_BACKSTOP: bool = True

    # === Grounding ===
    # Bound the workspace scan so a large monorepo cannot make grounding unbounded;
    # files beyond this count are skipped (and the truncation is logged).
    GROUNDING_MAX_FILES: int = 5000

    # === GIF preview ===
    # On-demand animated GIF rendered from the stored recording so a client can
    # embed a looping preview (issue, chat, extension) without a video player.
    # The window is capped at GIF_MAX_DURATION_S to keep the file small.
    GIF_FPS: int = 10
    GIF_WIDTH: int = 640
    GIF_MAX_DURATION_S: float = 30.0

    # === Storage ===
    BUNDLE_DIR: Path = Path("./bug-reports")
    DATABASE_PATH: Path = Path("./bug-reports/jobs.db")
    # Delete persisted bundles (and their job rows) older than this many days on
    # startup, so disk use stays bounded. 0 disables cleanup (keep forever).
    BUNDLE_TTL_DAYS: int = 0

    # === Job queue ===
    MAX_CONCURRENT_JOBS: int = 2
    JOB_TIMEOUT_S: int = 1800  # 30 minutes

    # === Notifications ===
    # POST a compact completion payload (job id, state, title, action) to this URL
    # when a job finishes, so an external system can react without polling. Blank
    # disables it. Loopback/private targets are allowed (it is your own webhook).
    WEBHOOK_URL: str = ""
    WEBHOOK_TIMEOUT_S: float = 10.0

    # === Logging ===
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = True

    # === Security ===
    CHROME_EXTENSION_ORIGIN: str = "chrome-extension://localhost"
    # Browser origins (besides the extension) allowed to call this API directly —
    # e.g. the hosted site connecting to your locally-running agent. The deployed
    # marketing site + local dev are allowed by default so the "Try it" widget
    # works with zero config; override with a comma-separated WEB_ORIGINS.
    WEB_ORIGINS: str = (
        "https://framesleuth.com,https://www.framesleuth.com,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )
    REDACT_BEFORE_PROMPTS: bool = True
    # Also redact PII (emails, Luhn-valid card numbers, US SSNs/phones, cloud keys)
    # from OCR/console text, on top of the always-on secret/token patterns.
    REDACT_PII: bool = True

    @property
    def web_origins_list(self) -> list[str]:
        """Parsed, de-blanked list of allowed browser origins."""
        return [o.strip() for o in self.WEB_ORIGINS.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "forbid",  # Reject unknown env vars for typo detection
    }

    def validate_paths(self) -> None:
        """Ensure required directories exist."""
        self.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get the global settings instance (singleton pattern).

    Returns:
        Settings: Validated application configuration.
    """
    return Settings()
