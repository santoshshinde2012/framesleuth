# Framesleuth capabilities

The single reference for **everything the agent can do** — its inputs, its outputs,
and every surface it exposes. For setup and deeper walkthroughs, follow the links at
the end.

> **What it is, in one line:** Framesleuth takes a screen-recording (plus optional
> browser sidecars) and produces a structured, evidence-cited **Bug Context Bundle**
> — then exposes it over an HTTP API and an MCP server so any agent can read it and
> act. Everything runs locally; nothing leaves the machine.

---

## 1. What the pipeline does

A single linear pass (with one bounded resample step). Every stage **degrades
gracefully** — if a model or input is missing, the run still produces a valid bundle
and says what was missing via `analysis_quality`.

| Stage | Capability |
|---|---|
| **Preprocess** | Probe the container (recovers duration even from header-less browser WebM); sample frames at a bounded budget. |
| **Transcribe (ASR)** | Local `faster-whisper` narration → timestamped transcript; drops silence hallucinations. |
| **Understand (VLM)** | Per-keyframe `caption` / `ocr_text` / `ui_action` / `is_error_state` / `reason`, run **concurrently**. Sparse error frames are **re-OCR'd at full resolution, uncompressed**. |
| **Classify** | Deterministic, auditable bug/tutorial/demo/feedback/other scoring; an **ambiguous** result triggers a bounded **resample** + an optional **model tie-breaker**. |
| **Extract** | Assemble the canonical Bug Context Bundle with evidence citations and an `analysis_quality` trust signal. |
| **Redact** | Strip secret-like strings (tokens, keys, passwords) from text **before** persistence. |
| **Summarize** | A skill-shaped narrative of the recording (video + audio). |
| **Ground** | Read-only workspace search → ranked candidate `file:line` locations. |
| **Resolve action** | Pick what a downstream agent should *do* (see Actions), auto-selected from the classification. |

---

## 2. Inputs

Supplied to `POST /v1/analyze` (form fields) or the `analyze_video` MCP tool (args).

| Input | Required | Purpose |
|---|---|---|
| `video` | ✅ | The recording (`.mp4` / `.webm` / `.mkv` / `.mov` / `.avi`). |
| `sidecars` | — | JSON array of browser events (console errors, failed network requests, clicks, env). Strong evidence even with no vision model. |
| `intent` | — | The user's request in their own words; recorded and woven into the fix-prompt. |
| `skill` | — | Summary **style** (see Skills). Defaults to `summary`. |
| `system_prompt` | — | Fully custom summary system prompt; overrides `skill`. |
| `action` | — | What the agent should **do** (see Actions). Auto-picked from the classification when omitted. |
| `action_prompt` | — | Fully custom action task; overrides `action`. |
| `capture_options` | — | (HTTP only) Arbitrary capture metadata, stored beside the bundle for provenance. |
| `repo_root` | — | (MCP only) Workspace to ground error text against. |

**Limits:** `MAX_UPLOAD_MB` (default 512), `MAX_DURATION_S` (default 600 s). Identical
bytes return the cached report (content-hash idempotency).

---

## 3. Output — the Bug Context Bundle

`GET /v1/report/{id}` (or `get_bug_report`) returns this stable, versioned JSON.

| Field | What it is |
|---|---|
| `id`, `schema_version`, `source_video`, `duration_s`, `created_at` | Provenance. |
| `classification` | `label` (bug/tutorial/demo/feedback/other) + `confidence` + `alt_labels`. |
| `title`, `severity`, `priority`, `suspected_component`, `reproducibility` | Triage headline. |
| `environment` | OS / app / browser / version from sidecars or OCR. |
| `preconditions`, `expected_behavior`, `actual_behavior` | The behavioral story. |
| `repro_steps[]` | Numbered, **evidence-cited** steps. |
| `error_evidence[]` | Timestamped console / OCR / network / UI errors. |
| `keyframe_refs[]` | The frames the model read (resolvable images). |
| `code_candidates[]` | Ranked `file:line` locations from grounding. |
| `summary`, `skill` | Narrative summary + the style used. |
| `action`, `action_prompt` | Resolved response mode (and custom task, if any). |
| `suggested_actions[]` | Machine-readable next-step menu (`action` / `label` / `rationale` / `ref`). |
| `user_intent` | The request you passed. |
| `analysis_quality` | **Trust signal** — `level` (`full`/`partial`/`degraded`) + `degraded_stages` + `warnings` + `evidence_counts`. Read this first. |
| `redactions[]` | What was scrubbed. |
| `transcript_path`, `timeline_path` | Sibling artifacts (`transcript.json`, `timeline.json`). |

Sibling files written next to `bundle.json`: `source.<ext>`, `keyframes/*.png`,
`transcript.json`, `timeline.json`, `sidecars.json`, `metrics.json`.

---

## 4. Skills — how the summary reads

Pick with `skill`, or override with `system_prompt`. List live via `GET /v1/skills`.

| Skill | Output |
|---|---|
| `summary` *(default)* | Narrative + ordered steps + any issues. |
| `bug_report` | QA bug report (title, repro, expected/actual, severity). |
| `tutorial` | Step-by-step how-to of what was demonstrated. |
| `action_items` | Decisions + follow-ups (owners when shown). |
| `release_notes` | Short, user-facing change notes. |

---

## 5. Actions — what the agent should do

Pick with `action`, or override with `action_prompt`. List live via `GET /v1/actions`.
With no `action`, one is **auto-picked** from the classification (bug→`fix`,
tutorial/demo→`explain`, feedback→`report`).

| Action | The downstream agent is told to… |
|---|---|
| `fix` | Diagnose the root cause and propose/make a minimal fix. |
| `explain` | Explain what happened — no code changes. |
| `triage` | Assess severity/priority and route it — no fix. |
| `test` | Write a failing regression test that reproduces it. |
| `report` | Produce a ready-to-paste issue/PR description. |
| `reproduce` | Produce minimal exact steps / a script to reproduce. |

---

## 6. Renderers — shareable artifacts

Turn any finished report into a consumable artifact (no new info — projected from the
bundle). Via the `render(report_id, format)` MCP tool or the resources below.

| Format | Output |
|---|---|
| `markdown` | A shareable markdown report. |
| `issue` | GitHub-issue text (title + labels + body). |
| `test-plan` | A framework-agnostic regression test plan. |

---

## 7. HTTP API

Loopback only (`127.0.0.1:8010` by default). Front it with your own backend; never
expose it to the browser/internet directly.

| Method · Path | Returns |
|---|---|
| `GET /v1/healthz` | VLM / coder / storage readiness + overall `status`. Also includes a `render` block reporting optional HTML→video readiness in the running process: `{playwright, playwright_version, chromium, ffmpeg, python, ready, hint}`. |
| `GET /v1/skills` | Built-in summary skills + default. |
| `GET /v1/actions` | Built-in action modes + default + `auto` flag. |
| `POST /v1/analyze` | Queues the pipeline (background, bounded by `MAX_CONCURRENT_JOBS`) → `202 {job_id, status: "queued", idempotent}`. Poll `/v1/jobs/{id}` for completion. |
| `GET /v1/jobs/{id}` | Lifecycle state + progress + error. |
| `GET /v1/report/{id}` | The full Bug Context Bundle. |
| `GET /v1/video/{id}` | The stored source recording (correct media type). |
| `GET /v1/gif/{id}` | An animated GIF preview of the recording (`image/gif`). Optional `fps`/`width`/`start`/`end` query params (clamped); rendered on demand and cached on disk per parameter set. |
| `POST /v1/render-html` | Render an HTML document (CSS/JS/canvas animation) to a clip. JSON body `{html, format: mp4\|gif\|webm, duration_s, fps, width, height}`; returns the encoded file. Optional capability — needs the `render` extra (Playwright) + `ffmpeg`; returns `503` with an actionable message when unavailable. Check `GET /v1/healthz` → `render.ready` first. |

---

## 8. MCP server (`videobug`)

Stdio (`framesleuth-mcp`). All tools are **read-only** over the workspace/bundle dir
— edits happen only through the calling agent's reviewed apply flow.

**Tools (14):** `analyze_video`, `list_skills`, `list_actions`, `list_bug_reports`,
`get_bug_report(view=full|slim)`, `get_suggested_actions`, `get_repro_steps`,
`get_error_evidence`, `get_timeline`, `get_keyframe_image`,
`get_video_gif(fps,width,start,end)`, `locate_in_code`, `render(format)`,
`render_html_video(html,format,duration_s,fps,width,height)`.

**Resources (4):** `videobug://report/{id}/summary` · `…/fix-prompt` · `…/markdown` ·
`…/issue`.

**Prompt (1):** `fix_from_video(report_id)` — the grounded, action-aware action prompt.

See [Use with VS Code & Claude](use-with-vscode-and-claude.md) for client setup.

---

## 9. Cross-cutting capabilities

- **Graceful degradation** — every stage can fail independently; the bundle stays
  well-formed and `analysis_quality` reports `full` / `partial` / `degraded` with
  warnings, so a thin recording is never mistaken for "nothing wrong".
- **Evidence by structure** — every claim carries a citation (frame / sidecar /
  transcript); uncited claims are dropped.
- **Redaction-first** — secret-like strings are scrubbed from text before anything is
  persisted or returned (text-only today; pixel-level redaction is not yet built).
- **Engine-agnostic** — swap Ollama / llama.cpp / vLLM via config (`VLM_URL`,
  `VLM_MODEL`, …); no code change.
- **Idempotent & self-cleaning** — identical uploads return the cached report; per-job
  scratch files are removed after each run.
- **Local & private** — all models run on `localhost`; no telemetry, no cloud calls.

Tuning knobs (concurrency, JSON mode, token caps, resample, classification
tie-breaker, frame resolution) are documented in [`.env.example`](../.env.example).

---

## Where to go next

- [Use with VS Code & Claude (MCP)](use-with-vscode-and-claude.md) — connect an agent + the end-to-end test recipe.
- [Web App Integration](web-integration.md) — embed Framesleuth behind your own backend with an agent loop.
- [Postman Collection](../postman/README.md) — exercise the HTTP API end-to-end.
- [Runbook & Troubleshooting](../runbook.md) — setup, health checks, and common issues.
