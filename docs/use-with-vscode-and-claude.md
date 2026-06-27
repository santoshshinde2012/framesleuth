# Use Framesleuth from VS Code Copilot & Claude (MCP)

Framesleuth ships an MCP server (`framesleuth-mcp`, server name **`videobug`**)
that lets an AI agent analyze a bug video and act on the result — all over the
Model Context Protocol. This page is the **client setup reference** for the three
common agents:

- **VS Code + GitHub Copilot** (agent mode)
- **Claude Code** (CLI)
- **Claude Desktop**

> This page focuses on **connecting each client**, then driving it (§4) and a
> copy/paste end-to-end test (§5). For setup, health checks, and troubleshooting
> of the analysis backend, see [runbook.md](../runbook.md).

---

## What the `videobug` server exposes

Everything is **read-only** over the workspace and the bundle directory — the
server never edits files. Edits happen through the editor/agent's own reviewed
apply flow.

| Kind | Name | Purpose |
|---|---|---|
| tool | `analyze_video(path, repo_root?, intent?, skill?, system_prompt?, action?, action_prompt?)` | Run the full pipeline on a local video; returns the report `id`, the resolved `action`, the `suggested_actions` menu, and `summary_resource`/`fix_prompt_resource` URIs. Pass `repo_root` to ground errors to code, `intent` to steer the work, `skill` for a summary style, and `action` (or `action_prompt`) to shape what the agent should do (see §A). |
| tool | `list_skills()` | List built-in summary skills (names + descriptions) |
| tool | `list_actions()` | List built-in **action modes** (`fix`, `explain`, `triage`, `test`, `report`, `reproduce`) |
| tool | `list_bug_reports()` | List available report ids |
| tool | `get_bug_report(report_id, view?)` | The Bug Context Bundle. `view="slim"` returns the action-relevant subset for small context windows; `view="full"` (default) returns everything. |
| tool | `get_suggested_actions(report_id)` | Machine-readable next-step menu (`action`/`label`/`rationale`/`ref`) |
| tool | `get_repro_steps(report_id)` | Numbered, cited reproduction steps |
| tool | `get_error_evidence(report_id)` | Timestamped console/OCR/network errors |
| tool | `get_timeline(report_id)` | Merged scene + transcript + sidecar timeline |
| tool | `get_keyframe_image(report_id, index)` | A decoded failure frame (PNG) |
| tool | `get_video_gif(report_id, fps?, width?, start?, end?)` | An animated GIF preview of the recording (cached on disk) |
| tool | `locate_in_code(report_id, repo_root?)` | Ranked candidate files/lines in the repo |
| tool | `render(report_id, format)` | Render the report as `markdown`, `issue` (GitHub issue text), or `test-plan` |
| resource | `videobug://report/{id}/summary` | Concise human summary + `suggested_actions` |
| resource | `videobug://report/{id}/fix-prompt` | Intent- and **action**-aware, evidence-only action prompt |
| resource | `videobug://report/{id}/markdown` | Shareable markdown report |
| resource | `videobug://report/{id}/issue` | GitHub-issue text (title + labels + body) |
| prompt | `fix_from_video(report_id)` | Same action prompt, as an MCP prompt |

The `fix-prompt` leads with an **Analysis confidence** block derived from
`analysis_quality` (`full` / `partial` / `degraded`): on a degraded run it tells
the agent *not* to fabricate and to gather more evidence, so a thin recording is
never mistaken for "nothing is wrong."

### A. Action modes — what the agent should *do*

Skills shape the **summary** prose; **actions** shape the **fix-prompt's task** —
i.e. what a coding agent is told to do with the evidence. Pick one with the
`action` param, or pass a fully custom `action_prompt`.

| Action | The agent is told to… |
|---|---|
| `fix` | Diagnose the root cause and propose/make a minimal, targeted fix |
| `explain` | Explain what happened — no code changes |
| `triage` | Assess severity/priority and route to a component — no fix |
| `test` | Write a failing regression test that reproduces it |
| `report` | Produce a ready-to-paste issue/PR description |
| `reproduce` | Produce minimal exact steps / a script to reproduce locally |

**Default = auto-pick from classification:** with no `action`, a `bug` →
`fix`, `tutorial`/`demo` → `explain`, `feedback` → `report`. The resolved action
is stored on the report and drives the `fix-prompt`, so re-reading it later is
consistent. `get_suggested_actions` returns a menu of follow-ups (propose a fix,
write a test, open an issue, re-record with logs on a degraded run) that an agent
can present or auto-invoke.

---

## 0. Prerequisite (all clients)

Install the package so the `framesleuth-mcp` entrypoint resolves:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
which framesleuth-mcp          # -> <repo>/.venv/bin/framesleuth-mcp
```

Two environment variables point the server at the report store (the **same**
`bug-reports/` the HTTP backend writes to, so reports from either path show up):

- `BUNDLE_DIR` → e.g. `<repo>/bug-reports`
- `DATABASE_PATH` → e.g. `<repo>/bug-reports/jobs.db`

> **Absolute paths:** config files do **not** expand `~` or `${workspaceFolder}`
> (except VS Code's `mcp.json`, which expands `${workspaceFolder}`). Everywhere
> else use the real absolute path, e.g.
> `/Users/you/workspace/framesleuth/.venv/bin/framesleuth-mcp`.

### Smoke-test the server before wiring a client

Confirm the server starts and registers its tools/resources/prompt without
needing a client — this isolates "the server is broken" from "the client config
is wrong":

```bash
python - <<'PY'
import asyncio
from framesleuth.mcp_server.videobug_mcp import build_server

async def main():
    s = build_server()
    print("tools:    ", sorted(t.name for t in await s.list_tools()))
    print("resources:", sorted(r.uriTemplate for r in await s.list_resource_templates()))
    print("prompts:  ", sorted(p.name for p in await s.list_prompts()))

asyncio.run(main())
PY
```

Expected: 14 tools (`analyze_video`, `list_skills`, `list_actions`,
`list_bug_reports`, `get_bug_report`, `get_suggested_actions`, `get_repro_steps`,
`get_error_evidence`, `get_timeline`, `get_keyframe_image`, `get_video_gif`,
`locate_in_code`, `render`, `render_html_video`), 4 resources, and the
`fix_from_video` prompt.
Running `framesleuth-mcp`
directly starts the stdio server (it waits for
a client and produces no output — Ctrl-C to exit; that silence is success).

---

## 1. VS Code + GitHub Copilot

Top-level key is **`servers`** (VS Code-specific — differs from Claude). This repo
already ships [`.vscode/mcp.json`](../.vscode/mcp.json) using `${workspaceFolder}`:

```jsonc
{
  "servers": {
    "videobug": {
      "type": "stdio",
      "command": "${workspaceFolder}/.venv/bin/framesleuth-mcp",
      "args": [],
      "env": {
        "BUNDLE_DIR": "${workspaceFolder}/bug-reports",
        "DATABASE_PATH": "${workspaceFolder}/bug-reports/jobs.db"
      }
    }
  }
}
```

1. `code .` from the repo root.
2. Open `.vscode/mcp.json` → click the **Start** code lens above `videobug`
   (or **MCP: List Servers** → start `videobug`).
3. Open **Copilot Chat → Agent mode** (tools are invisible in Ask mode).

> Windows: `${workspaceFolder}\\.venv\\Scripts\\framesleuth-mcp.exe`.

---

## 2. Claude Code (CLI)

### Quick add

`-e` env flags and `-s` scope go **before** the `--`; everything after `--` is the
command Claude Code runs:

```bash
claude mcp add videobug \
  -s project \
  -e BUNDLE_DIR=/Users/you/workspace/framesleuth/bug-reports \
  -e DATABASE_PATH=/Users/you/workspace/framesleuth/bug-reports/jobs.db \
  -- /Users/you/workspace/framesleuth/.venv/bin/framesleuth-mcp
```

Scopes: `-s local` (default; private to you, this project), `-s project` (shared
via a checked-in `.mcp.json`), `-s user` (all your projects).

Verify and use:

```bash
claude mcp list           # videobug should be listed
claude                    # then ask it to analyze a video (see §4)
```

### Checked-in project config (`.mcp.json`)

For a team, commit a `.mcp.json` in the repo root. Top-level key is **`mcpServers`**:

```json
{
  "mcpServers": {
    "videobug": {
      "type": "stdio",
      "command": "/Users/you/workspace/framesleuth/.venv/bin/framesleuth-mcp",
      "env": {
        "BUNDLE_DIR": "/Users/you/workspace/framesleuth/bug-reports",
        "DATABASE_PATH": "/Users/you/workspace/framesleuth/bug-reports/jobs.db"
      }
    }
  }
}
```

Teammates get an approval prompt on first use.

---

## 3. Claude Desktop

Edit the config file (create it if missing), then fully quit and reopen Claude
Desktop.

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Top-level key is **`mcpServers`**; stdio is the default (no `type` needed):

```json
{
  "mcpServers": {
    "videobug": {
      "command": "/Users/you/workspace/framesleuth/.venv/bin/framesleuth-mcp",
      "args": [],
      "env": {
        "BUNDLE_DIR": "/Users/you/workspace/framesleuth/bug-reports",
        "DATABASE_PATH": "/Users/you/workspace/framesleuth/bug-reports/jobs.db"
      }
    }
  }
}
```

After restart, the `videobug` tools appear under the tools/🔌 menu.

---

## 4. Drive it (same in any client)

In an agent chat:

> Analyze the video at `samples/flash_bug.mp4` for this repo — the Save button
> hangs, find why and fix it, and add a retry.

The agent calls `analyze_video(path, repo_root=<this repo>, intent=…)`, which
writes `bug-reports/<id>/bundle.json`, then:

> Read report `<id>`: get its repro steps, error evidence, and fix-prompt, then
> propose a grounded code change.

Because the agent has your repo open, it turns the bundle's grounded
`code_candidates` + the evidence-only fix prompt into edits through the client's
normal review flow.

Want a different outcome? Steer it with an **action**:

> Analyze `samples/flash_bug.mp4` with `action="test"` — write a failing
> regression test instead of fixing it.

> …with `action="triage"` — just assess severity and tell me which component
> owns it.

Or skip the analysis and turn an existing report into an artifact:

> Call `render(report_id="<id>", format="issue")` and open a GitHub issue with
> that body.

---

## 5. End-to-end test in VS Code (copy/paste)

A concrete pass you can run top-to-bottom to confirm Copilot can analyze a video
**and act on it**. Uses the bundled `samples/flash_bug.mp4`.

1. **(Optional) Start the vision model for `full` quality.** Skip to get a
   `degraded`/`partial` bundle from sidecars only — the flow still works.
   ```bash
   ollama serve &            # or your llama.cpp server on :8080
   ollama pull qwen2.5vl
   ```
2. **Install + smoke-test the server** (see §0). `which framesleuth-mcp` resolves
   and the smoke test prints 14 tools.
3. **Start the server in VS Code.** `code .`, open `.vscode/mcp.json`, click
   **Start** above `videobug` (or **MCP: List Servers → Start**). The code lens
   should flip to **Running**.
4. **Analyze + act, in Copilot Chat → Agent mode:**
   > Use the videobug tools. Call `analyze_video` with
   > `path="samples/flash_bug.mp4"`, `repo_root` set to this workspace, and
   > `intent="find why the Save button hangs and propose a fix"`. Then read the
   > report's summary, repro steps, error evidence, and fix-prompt, and propose a
   > grounded code change.
5. **Verify the run landed** (any shell):
   ```bash
   ls bug-reports/*/bundle.json | tail -1            # a fresh bundle exists
   # quality, resolved action, and the next-step menu the agent received:
   python -c "import json,glob,os; b=json.load(open(max(glob.glob('bug-reports/*/bundle.json'),key=os.path.getmtime))); print('quality=',b['analysis_quality']['level'],'action=',b.get('action'),'suggested=',[s['action'] for s in b.get('suggested_actions',[])])"
   ```

**Pass criteria:** the agent invoked `analyze_video`, a new `bundle.json` was
written with an auto-picked `action` and a non-empty `suggested_actions` menu,
and it proposed edits referencing real files/lines from the bundle's
`code_candidates` — gated by the fix-prompt's confidence block (it should refuse
to fabricate on a `degraded` run). The same chat prompt works verbatim in Claude
Code / Claude Desktop once the server is connected (§2–§3).

---

## 6. Tune latency & quality (optional)

The analysis pipeline is configured via env vars (set them in the server's `env`
block in `mcp.json`/`.mcp.json`, or a `.env` next to the repo). Defaults are
tuned for a single local GPU; the ones worth knowing:

| Var | Default | Effect |
|---|---|---|
| `VLM_MAX_CONCURRENCY` | `3` | Keyframes analyzed in parallel. Raise only if your engine serves concurrently (`OLLAMA_NUM_PARALLEL` / llama.cpp `--parallel`); otherwise it harmlessly serializes. |
| `VLM_MAX_TOKENS` | `768` | Per-frame generation cap (lower = faster). |
| `VLM_JSON_MODE` | `true` | OpenAI-style JSON output for reliable parsing. Set `false` for an engine that rejects `response_format`. |
| `VLM_SEND_JPEG` | `true` | Send frames as JPEG (smaller upload, fewer vision tokens). Stored keyframes stay PNG. |
| `MAX_RESAMPLE_RETRIES` | `2` | Bounded resample around the failure window on ambiguous runs (`0` disables). |
| `CLASSIFY_USE_MODEL` | `true` | Break ambiguous-band classification ties with a model call. |

See [`.env.example`](../.env.example) for the full list. All are optional —
the server runs with sensible defaults out of the box.

---

## 7. Notes & troubleshooting

- **Degraded vs full.** Without the model servers (llama.cpp/Ollama) the pipeline
  runs in sidecar/degraded mode: no `keyframes/`, and `analysis_quality.level`
  will be `partial` or `degraded`. Add the models (see
  [runbook.md](../runbook.md)) for real frame OCR/captions and `full` quality.
- **Shared store.** Reports created via the HTTP backend (`POST /v1/analyze`) and
  via the `analyze_video` tool land in the same `BUNDLE_DIR`, so either client
  sees both. Keep `BUNDLE_DIR`/`DATABASE_PATH` consistent across clients.
- **Read-only.** The server never edits files; all changes go through the agent's
  reviewed apply flow.

| Symptom | Fix |
|---|---|
| `framesleuth-mcp: command not found` | Activate the venv, `uv pip install -e ".[dev]"`, re-check `which framesleuth-mcp`. |
| Tools don't appear (Copilot) | Use **Agent mode**, and Start + trust the server. |
| Tools don't appear (Claude) | Use an absolute `command` path; restart the client; `claude mcp list` (CLI). |
| `get_keyframe_image` returns nothing | Keyframes exist only after a **full-mode** analysis. |
| A client can't find a report | Ensure its `BUNDLE_DIR` matches where the analysis wrote. |
| `analyze_video` errors / empty captions after a model 400 | Your VLM engine rejects `response_format`. Set `VLM_JSON_MODE=false` in the server `env`. |
| Frame analysis feels slow | Lower `VLM_MAX_TOKENS`, and only raise `VLM_MAX_CONCURRENCY` if the engine serves requests in parallel (§6). |
