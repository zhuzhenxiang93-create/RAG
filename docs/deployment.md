# Deployment and operations

## Local virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lite.txt
.\scripts\run-demo.ps1
```

The demo script opens the workbench and stops its child server when the script exits.

## Docker Compose

```powershell
docker compose up --build
```

The container:

- runs as non-root UID `10001`;
- exposes port `8000`;
- has an HTTP health check;
- writes runtime data to the `docmind-data` named volume;
- applies `no-new-privileges`;
- keeps the bundled evaluation corpus inside the image.

Stop it with:

```powershell
docker compose down
```

Add `-v` only when the runtime volume should also be deleted.

## Operational endpoints

- `GET /api/health`: liveness and optional capability availability.
- `GET /api/diagnostics`: uptime, document/chunk totals and in-memory index state.
- Every response includes `X-Request-ID` and `Server-Timing`.

Clients may supply an `X-Request-ID` containing 1-64 letters, digits, dots, underscores
or hyphens. Invalid identifiers are replaced. Access logs never include request bodies,
document contents, API keys or filesystem paths.

## Structured logging

`DOCMIND_LOG_FORMAT=json` is the default. Set it to `text` for local development.

Example event:

```json
{
  "level": "INFO",
  "logger": "docmind.access",
  "message": "request_completed",
  "request_id": "interview-demo-01",
  "method": "POST",
  "path": "/api/chat",
  "status_code": 200,
  "duration_ms": 4.218
}
```

## CI

`.github/workflows/ci.yml` performs:

1. Lite dependency installation and `pip check`;
2. Python and frontend JavaScript syntax checks;
3. standard-library unit/integration tests;
4. Lite Docker image build.

The workflow was verified on GitHub Actions run `30195841724`: the test job completed
successfully in 18 seconds and the Lite container build completed successfully in
23 seconds. GitHub emitted a Node 20 deprecation compatibility warning for official
actions, but it did not fail either job.
