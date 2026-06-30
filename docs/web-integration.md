# Integrating Framesleuth into a web application (end-to-end)

This is the **one** guide for embedding Framesleuth in **your own web/client app** so a
user can upload a bug video, type what they want done, and have an **agent** act on it —
diagnose the bug, draft a fix, or answer a question — grounded in what the video shows.

It covers the whole path: the reference architecture, the plain HTTP integration (with
the **async analyze → poll → report** flow), and the agentic loop that hands the result
to Claude.

> **Local-first & private.** Framesleuth binds loopback (`127.0.0.1`) — it is **not**
> meant to be exposed on a network or the public internet. Its CORS allowlist is an
> explicit set of origins (`WEB_ORIGINS`, default: the hosted demo site + local dev)
> plus `chrome-extension://`, and it answers Chrome's Private Network Access preflight,
> so a page on an allowed origin running **in your browser, on your machine** can call
> the loopback API directly. This is what lets the "Try it" widget on framesleuth.com
> drive your locally-running agent with zero setup — it never makes the agent reachable
> off your machine.
>
> For your **own production web app**, prefer the proxy pattern below (browser → your
> server → Framesleuth) and narrow `WEB_ORIGINS` to just your origins, so you keep the
> analysis on a machine you control and add your own auth.

---

## 1. Reference architecture

```
┌─────────────┐   video + "fix the save bug"   ┌────────────────────┐
│  Browser    │ ─────────  (your UI)  ────────▶ │  Your backend      │
│  (your app) │                                 │  (Node/Python/…)   │
└─────────────┘                                 └─────────┬──────────┘
                                                          │ 1. POST /v1/analyze  → 202 {job_id}
                                                          │ 2. poll GET /v1/jobs/{id} until done
                                                          │ 3. GET /v1/report/{id}
                                                          ▼
                                              ┌────────────────────────┐
                                              │  Framesleuth :8010      │
                                              │  → Context Bundle   │
                                              └─────────┬──────────────┘
                                       4. bundle + evidence            │
                                                          ▼
                                              ┌────────────────────────┐
                                              │  Agent (Claude API)     │
                                              │  loops over tools,      │
                                              │  reads your repo, drafts│
                                              │  a fix / answer         │
                                              └─────────┬──────────────┘
                                       5. proposed change / answer      │
                                                          ▼
                                                    back to the browser
```

The **agentic** part is steps 4–5: instead of just showing the bundle, you hand it to an
LLM agent that decides what to do with it and takes the next action.

---

## 2. The plain HTTP integration (no agent yet)

Analysis is **asynchronous**: `POST /v1/analyze` returns `202` with a `job_id` immediately
and runs the pipeline in the background. **Poll `GET /v1/jobs/{id}`** until `state` is
`done`, then read `GET /v1/report/{id}`. Your backend forwards the upload and does the poll.

### Node / Express

```js
import express from "express";
import multer from "multer";

const upload = multer();
const FRAMESLEUTH = "http://127.0.0.1:8010";
const app = express();

// Submit the recording, then poll until the bundle is ready.
async function analyzeAndWait(file, fields) {
  const form = new FormData();
  form.append("video", new Blob([file.buffer]), file.originalname);
  if (fields.intent)   form.append("intent", fields.intent);     // the user's request
  if (fields.sidecars) form.append("sidecars", fields.sidecars); // optional JSON array
  if (fields.skill)    form.append("skill", fields.skill);       // optional summary style
  if (fields.action)   form.append("action", fields.action);     // optional: fix|explain|test|…

  // 1. Queue → 202 {job_id, status: "queued", idempotent}
  const { job_id } = await (
    await fetch(`${FRAMESLEUTH}/v1/analyze`, { method: "POST", body: form })
  ).json();

  // 2. Poll job status until it finishes
  for (;;) {
    const job = await (await fetch(`${FRAMESLEUTH}/v1/jobs/${job_id}`)).json();
    if (job.state === "done") break;
    if (job.state === "failed") throw new Error(JSON.stringify(job.error));
    await new Promise((r) => setTimeout(r, 1000));
  }

  // 3. Read the Context Bundle
  const report = await (await fetch(`${FRAMESLEUTH}/v1/report/${job_id}`)).json();
  return { job_id, report };
}

app.post("/api/report-bug", upload.single("video"), async (req, res) => {
  const { job_id, report } = await analyzeAndWait(req.file, req.body);
  res.json({ job_id, report }); // your UI renders steps, errors, intent, etc.
});
```

Browser side (`fetch` + `FormData`):

```js
const fd = new FormData();
fd.append("video", fileInput.files[0]);
fd.append("intent", "The Save button hangs. Find why and fix it.");
const { report } = await (await fetch("/api/report-bug", { method: "POST", body: fd })).json();
```

That's the whole non-agentic path.

> **Don't want to poll?** Stream `GET /v1/jobs/{id}/events` (Server-Sent Events) to
> receive JSON progress snapshots until the job is terminal, or set `WEBHOOK_URL` so
> the backend POSTs a compact `{id, state, title, action}` payload to your server on
> completion. To abort a run, `DELETE /v1/jobs/{id}` (cooperative — it stops at the
> next stage boundary and the job becomes `cancelled`).

### Inputs — `POST /v1/analyze` (multipart form)

| Field | Required | Purpose |
|---|---|---|
| `video` | ✅ | The recording (`.mp4`/`.webm`/`.mkv`/`.mov`/`.avi`). |
| `sidecars` | — | JSON array of browser events (console errors, failed requests, clicks, env). Strong bug evidence even with no vision model. |
| `intent` | — | The user's request, in their words. Recorded and woven into the fix-prompt. |
| `skill` | — | Summary **style**: `summary` (default) · `bug_report` · `tutorial` · `action_items` · `release_notes`. `GET /v1/skills` lists them. |
| `system_prompt` | — | Fully custom summary prompt; overrides `skill`. |
| `action` | — | What the agent should **do**: `fix` · `summarize` · `explain` · `triage` · `test` · `report` · `reproduce`. Auto-picked from the classification when omitted. `GET /v1/actions` lists them. |
| `action_prompt` | — | Fully custom action task; overrides `action`. |
| `capture_options` | — | Arbitrary capture metadata, stored beside the bundle for provenance. |

Returns `202 { job_id, status: "queued", idempotent }`. Re-posting identical bytes returns
the cached job (`idempotent: "true"`) without re-running. Limits: `MAX_UPLOAD_MB`,
`MAX_DURATION_S`.

### Output — the Context Bundle (`GET /v1/report/{id}`)

The bundle the consumer reads. Key fields:

| Field | What it is |
|---|---|
| `classification` | `label` (bug/tutorial/demo/feedback/other) + `confidence`. |
| `title`, `severity`, `suspected_component` | Triage headline. Bug-only — `null` for a general video. |
| `summary`, `key_moments[]` | Narrative summary + salient timestamped moments — the deliverable for any (non-bug) video. |
| `repro_steps[]` | Numbered, evidence-cited steps (empty when none were shown). |
| `error_evidence[]` | Timestamped console / OCR / network errors. |
| `code_candidates[]` | Ranked `file:line` locations from grounding. |
| `skill` | The summary style/skill used to produce `summary`. |
| `action`, `suggested_actions[]` | Resolved response mode + a machine-readable next-step menu. |
| `keyframe_refs[]`, `user_intent`, `environment` | Frames read, your request, environment. |
| `stage_timings` | Per-stage wall-clock seconds — surface "where the time went" in your UI. |
| `analysis_quality` | **Trust signal** — `full`/`partial`/`degraded` + `warnings`. Gate your UI/agent on it so a thin recording isn't shown as "no issues found". |

See **[capabilities.md](capabilities.md)** for the complete input/output reference and
every supported skill, action, renderer, endpoint, and MCP tool.

### Also available — HTML → video (optional)

Separate from the analyze loop, the backend can render a self-contained HTML
animation (CSS/JS/canvas) to a clip **frame-by-frame** — full color, no dropped
frames, no quality loss (up to 4K, 5–60 fps): `POST /v1/render-html` with
`{html, format: mp4|gif|webm, duration_s?, fps, width, height}` returns the encoded
file. **Omit `duration_s` to capture the whole animation** (length auto-detected; a
`window.__renderDurationMs` hint covers pure canvas loops). It's an **optional
capability** (needs the `render` extra + `ffmpeg`) and
returns `503` when unavailable — gate your UI on `GET /v1/healthz` → `render.ready`.
Proxy it the same way as the rest (browser → your backend → Framesleuth, loopback
only); binary responses pass straight through.

---

## 3. The agentic approach

Hand the bundle to an LLM agent and let it take the next action. Two patterns.

### Pattern A — Custom tool + Claude agent loop (recommended for web apps)

Expose Framesleuth to Claude as a **tool**. Claude decides when to analyze the video, reads
the result, and (with repo-access tools you add) drafts a fix. Your backend runs the loop
and keeps control of every side effect. This uses the Anthropic SDK — Framesleuth is
Python, so a Python backend is shown; the shapes are identical in the TypeScript SDK.

```python
# pip install anthropic httpx
import json
import time
import httpx
import anthropic

FRAMESLEUTH = "http://127.0.0.1:8010"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

# Submit a recording and block until the bundle is ready (async analyze → poll).
def analyze_and_wait(video_path: str, *, intent: str | None, action: str | None) -> dict:
    with open(video_path, "rb") as f:
        data = {}
        if intent:
            data["intent"] = intent
        if action:
            data["action"] = action
        r = httpx.post(
            f"{FRAMESLEUTH}/v1/analyze",
            files={"video": (video_path, f, "video/webm")},
            data=data,
            timeout=60,
        )
    job_id = r.json()["job_id"]
    # Poll until done (the analysis runs in the background).
    while True:
        job = httpx.get(f"{FRAMESLEUTH}/v1/jobs/{job_id}", timeout=30).json()
        if job["state"] == "done":
            break
        if job["state"] == "failed":
            raise RuntimeError(job.get("error"))
        time.sleep(1)
    return httpx.get(f"{FRAMESLEUTH}/v1/report/{job_id}", timeout=30).json()

# 1) The tool Claude can call. Your backend executes it against Framesleuth.
TOOLS = [
    {
        "name": "analyze_bug_video",
        "description": (
            "Analyze a screen-recording of a bug and return a structured Bug Context "
            "Bundle (repro steps, error evidence, environment, candidate code "
            "locations). Call this once before diagnosing or fixing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Server path to the uploaded video"},
                "intent": {"type": "string", "description": "What the user asked you to do"},
                "action": {
                    "type": "string",
                    "description": (
                        "Optional response mode: fix | summarize | explain | triage | test | "
                        "report | reproduce. Omit to auto-pick from the classification."
                    ),
                },
            },
            "required": ["video_path", "intent"],
        },
    }
]

def run_tool(name: str, args: dict) -> str:
    if name == "analyze_bug_video":
        bundle = analyze_and_wait(
            args["video_path"], intent=args.get("intent"), action=args.get("action")
        )
        # Return only what the model needs, keep it compact.
        return json.dumps({
            "title": bundle["title"],
            "classification": bundle["classification"],
            "repro_steps": bundle["repro_steps"],
            "error_evidence": bundle["error_evidence"],
            "environment": bundle["environment"],
            "code_candidates": bundle["code_candidates"],
            "user_intent": bundle["user_intent"],
            # Resolved response mode + a ready-made menu of next steps (propose a fix,
            # write a test, open an issue, re-record). Let the agent pick.
            "action": bundle.get("action"),
            "suggested_actions": bundle.get("suggested_actions", []),
            # Trust signal — full | partial | degraded, with warnings. Lets the model act
            # confidently or gather more evidence instead of guessing.
            "analysis_quality": bundle["analysis_quality"],
        })
    raise ValueError(f"unknown tool {name}")

# 2) The agent loop: user gives a request + a video path; Claude drives.
def agent_turn(user_request: str, video_path: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"{user_request}\n\nThe screen recording is at: {video_path}. "
            "Use analyze_bug_video first, then carry out my request grounded in the "
            "evidence. Cite the evidence you used."
        ),
    }]
    while True:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},     # let Claude decide how hard to think
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "refusal":
            return "The request was declined by the model's safety system."
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = run_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})
```

Wire `agent_turn(...)` to an endpoint your browser calls. The user gets a grounded
answer/fix; Claude only ever *proposes* — your app decides whether to apply edits (open a
PR, show a diff for review). Add more tools (`read_file`, `search_repo`,
`open_pull_request`) to let the agent act on your codebase, gating the side-effecting ones
behind your own confirmation UI.

> **Why a tool, not just pasting the bundle?** The agent calls it only when needed, can
> re-analyze with a refined intent, and the same loop scales to many tools (read repo, run
> tests, open a PR) — the standard agentic shape. `GET /v1/report/{id}` returns the full
> Context Bundle; build your own prompt from it. Framesleuth's own grounded fix-prompt
> template is rendered over MCP, via the `framesleuth://report/{id}/fix-prompt` resource.

### Pattern B — MCP connector (agent already speaks MCP)

If your agent is MCP-native, point it at the `framesleuth` MCP server instead of defining a
tool. The server exposes `analyze_video` (with `action`/`skill`), `get_report`,
`get_suggested_actions`, `get_repro_steps`, `locate_in_code`, `render`, the
`fix-prompt`/`markdown`/`issue` resources, and more. Setup and the full tool list are in
**[use-with-vscode-and-claude.md](use-with-vscode-and-claude.md)**. For a backend agent,
launch it over stdio (`framesleuth-mcp`) and drive it with an MCP client; the Anthropic SDK
can convert MCP tools for the tool runner (`anthropic.lib.tools.mcp`), or Claude's MCP
connector can call a remote MCP endpoint directly.

---

## 4. Putting it together — the end-to-end flow

1. **Browser**: user picks a video, types a request, submits to *your* endpoint.
2. **Your backend**: saves the upload, calls `agent_turn(request, path)`.
3. **Agent**: calls `analyze_bug_video` → Framesleuth queues, your backend polls until
   `done` and returns the bundle → Claude reasons over the cited evidence and (with repo
   tools) drafts the change.
4. **Your backend**: returns the agent's answer/diff; optionally applies it behind a
   review/confirm step.
5. **Browser**: renders the result — the repro steps, the proposed fix, the citations.

Everything heavy (frame decode, VLM, ASR) runs locally inside Framesleuth; your app only
orchestrates and renders.

---

## 5. Deployment & security checklist

- **Keep Framesleuth on loopback.** It binds `127.0.0.1` and is not meant to be exposed on
  a network or the internet. The CORS allowlist (`WEB_ORIGINS`) only controls which
  browser origins may *read* responses from the local API — it does **not** make the agent
  network-reachable. For a production web app, narrow `WEB_ORIGINS` to your own origins (or
  set it empty) and proxy through your backend instead.
- **Put auth on *your* layer** (the browser→backend hop). Framesleuth assumes a trusted
  local caller.
- **Redaction is built in** — Framesleuth strips secrets (passwords, tokens, JWTs) **and
  PII** (emails, Luhn-valid card numbers, SSNs/phones, cloud keys) from evidence text
  before it's stored or returned, so the bundle you forward to an LLM is already scrubbed
  (`REDACT_PII`, on by default). Note redaction is text-only today — a secret visible *in a
  frame* is not yet pixel-masked, so still treat bundles and keyframes as sensitive.
- **Webhook target is yours.** If you set `WEBHOOK_URL`, the backend POSTs there on job
  completion — point it at an endpoint you control; the payload carries no secrets beyond
  the job title/action.
- **Gate side effects in the agent.** The analysis is read-only; any tool that writes
  (apply patch, open PR, run a command) should require your explicit confirmation, not the
  model's.
- **Bound the work.** `MAX_UPLOAD_MB`/`MAX_DURATION_S` cap input size; `MAX_CONCURRENT_JOBS`
  bounds how many analyses run at once (extras queue). Poll `/v1/jobs/{id}` rather than
  holding a request open.
- **Models.** Visual understanding needs a local VLM server (e.g. Ollama serving
  `qwen2.5vl`); without it Framesleuth degrades to sidecar/console evidence. See
  **[runbook.md](../runbook.md)**.

See also: **[capabilities.md](capabilities.md)** (every input, output, skill, action,
renderer, endpoint, and MCP tool), **[use-with-vscode-and-claude.md](use-with-vscode-and-claude.md)**
(MCP client setup), **[postman/README.md](../postman/README.md)** (exercise the HTTP API),
and **[runbook.md](../runbook.md)** (setup & troubleshooting).
