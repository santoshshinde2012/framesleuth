# Framesleuth — Postman collection

Test the local HTTP API end-to-end without writing any code.

## Files

| File | What it is |
|---|---|
| `Framesleuth.postman_collection.json` | All API requests with example bodies and test scripts |
| `Framesleuth.postman_environment.json` | `baseUrl` (default `http://127.0.0.1:8010`) and `jobId` |

## 1. Start the backend

```bash
source .venv/bin/activate
framesleuth-api            # binds 127.0.0.1:8010
```

(No model servers needed — the pipeline degrades gracefully and the bundle's
`analysis_quality` tells you what was skipped.)

## 2. Import into Postman

1. **Import** → drop both JSON files in.
2. Select the **Framesleuth Local** environment (top-right).
3. Run the requests in order:
   - **Health** → confirms the backend is up.
   - **Analyze video** → in the request **Body → form-data**, set the `video`
     field to a file (e.g. `samples/flash_bug.mp4`). It returns `202` with a
     `job_id`; the test script saves it into the `jobId` variable automatically.
   - **Get job status** → re-run until `state` is `done` (analysis runs in the
     background).
   - **Get report / Get source video / Get preview GIF** → reuse `jobId`.

## 3. Or run headless with Newman (CI-friendly)

`video` is a file field, so pass the path with `--form-data` overrides:

```bash
npm install -g newman

newman run postman/Framesleuth.postman_collection.json \
  -e postman/Framesleuth.postman_environment.json \
  --folder "Health"

# Analyze a specific file, then the report request reuses the captured jobId:
newman run postman/Framesleuth.postman_collection.json \
  -e postman/Framesleuth.postman_environment.json \
  --form-data "video=@samples/flash_bug.mp4"
```

## Endpoints at a glance

| Request | Method + path | Notes |
|---|---|---|
| Health | `GET /v1/healthz` | overall + per-service (`vlm`, `coder`, `storage`); plus a `render` block reporting optional HTML→video readiness |
| List skills | `GET /v1/skills` | built-in summary styles + the default |
| List actions | `GET /v1/actions` | built-in action modes (`fix`/`explain`/`triage`/`test`/`report`/`reproduce`) + default |
| Analyze video | `POST /v1/analyze` | multipart: `video` (file), `intent?`, `skill?`, `system_prompt?`, `action?`, `action_prompt?`, `sidecars?`, `capture_options?`. **Async** — returns `202 {job_id, status: "queued"}`; poll **Get job status**. Idempotent on the video's SHA-256. |
| Get job status | `GET /v1/jobs/{job_id}` | lifecycle state + progress; poll until `state` is `done` |
| Get report | `GET /v1/report/{job_id}` | the Bug Context Bundle (incl. `analysis_quality`); `409` until ready |
| Get source video | `GET /v1/video/{job_id}` | streams the stored recording |
| Get preview GIF | `GET /v1/gif/{job_id}` | animated `image/gif` preview; optional `fps`/`width`/`start`/`end`; cached on disk per params |
| Render HTML to video | `POST /v1/render-html` | render a self-contained HTML animation to `mp4`/`gif`/`webm`; **optional** (needs `render` extra + `ffmpeg`) — `200` with the file or `503` when unavailable |

> **Idempotency gotcha:** re-posting the *same bytes* returns the existing job
> (`idempotent: true`) without re-running. Use a different file, or clear the
> scratch store (`rm -rf bug-reports/*`, keeps `.gitkeep`) to force a fresh run.
