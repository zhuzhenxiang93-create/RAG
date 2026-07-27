# Project handoff

Repository: `https://github.com/zhuzhenxiang93-create/RAG`

## Verified entry points

- Workbench: `http://127.0.0.1:8000/ui/`
- OpenAPI: `http://127.0.0.1:8000/docs`
- Health: `GET /api/health`
- Diagnostics: `GET /api/diagnostics`
- Demo corpus: `POST /api/demo/bootstrap` in Lite mode
- Evaluation: `POST /api/evaluation/run`
- Intent status: `GET /api/plugins/intent/status`
- Intent classification: `POST /api/plugins/intent/classify`

## Reproduction sequence

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lite.txt
.\scripts\test.ps1
.\scripts\run-demo.ps1
```

For evaluation and load artifacts:

```powershell
.\.venv\Scripts\python.exe -B scripts\evaluate.py --benchmark lite_v1
.\.venv\Scripts\python.exe -B scripts\load_test.py --bootstrap --scenario chat --requests 100 --concurrency 8
```

## Verified versus pending

Verified:

- Lite document ingestion, retrieval, selective QA, demo bootstrap and evaluation;
- Lite intent classification, low-confidence fallback and adaptive-search integration;
- MASSIVE preparation, LoRA training and dependency-free metric scripts pass syntax checks;
- optional legal asset validation without importing heavy dependencies;
- Windows local virtual environment and browser workflow;
- deterministic tests, syntax checks and credential pattern scan.
- GitHub-hosted CI test job and Docker image build on run `30195841724`.

Pending:

- MASSIVE full download, GPU LoRA training and held-out Accuracy/Macro-F1/ECE;
- measured RAG retrieval improvement from predicted intent routing;
- real legal model inference and held-out classification score;
- production-grade embedding and Cross-Encoder evaluation;
- OCR provider integration;
- large independent corpus and calibrated confidence threshold.

## Original project safety

The legacy directories remain read-only:

- `D:\毕业设计\file-tools-system`
- `D:\LLM\law`

The verified backup remains:

- `D:\project-backups\legal-platform-20260726-102902`
