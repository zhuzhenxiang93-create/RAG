# Unified Case Analysis API and CLI

## Build data and train

```bash
python -m legalmind.data.build_dataset --config configs/data/processed_v2_1_1.yaml
python -m legalmind.cli train-sentencing
python -m legalmind.cli build-sentencing-index
python -m legalmind.cli evaluate-sentencing
```

For a smoke run, pass `--limit 2000`. A limited run is not a final evaluation.

## Analyze

```bash
python -m legalmind.cli analyze \
  --fact "当事人盗窃财物价值人民币5000元，已经退赃并取得谅解。" \
  --accusation "盗窃" \
  --top-k 3
```

`--accusation` is optional and repeatable. When it is omitted, the unified pipeline loads the
existing QLoRA classifier and preserves its real probabilities through retrieval and sentencing.
When supplied, it is an explicit user override. Case retrieval is restricted to the selected
charge partitions and returns at most three thresholded, deduplicated cases.

## Python and HTTP

Build a `LegalMindPipeline` with `legalmind.pipeline.analyze_case.build_pipeline` and call
`LegalMindPipeline.analyze`. The optional FastAPI factory is
`legalmind.pipeline.api.create_app(pipeline)` and exposes `GET /health` plus
`POST /v1/analyze`. The former sentencing-only import remains a compatibility alias, but the
sentencing-only HTTP route and CLI command are no longer exposed. Install the `api` optional
dependency before serving.

The API always includes a research disclaimer. Do not expose it as automated legal advice.
