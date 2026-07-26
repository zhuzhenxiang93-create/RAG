# Evaluation

## Built-in benchmark

`lite_v1` contains seven manually authored Markdown documents and fourteen questions:

- twelve answerable retrieval questions;
- one cross-version conflict;
- two unanswerable/out-of-scope questions.

The benchmark records gold documents, sections and expected selective-answer decisions.
It is intentionally small and exists to validate the experiment pipeline. Its metrics
must not be described as production performance or general model quality.

Run:

```powershell
python -B scripts/evaluate.py --benchmark lite_v1
```

Each run stores:

- environment and provider names;
- index fingerprint;
- aggregate metrics;
- per-question retrieved/cited documents;
- decisions and confidence;
- a prominent small-benchmark warning.

## Retrieval metrics

- Precision@1/3/5
- Recall@1/3/5
- MRR
- nDCG@1/3/5
- mean/P50/P95 end-to-end retrieval latency

## Trust metrics

- decision accuracy;
- abstention precision/recall;
- conflict precision/recall;
- citation document precision/recall.

The next benchmark version should add page/chunk-level gold evidence, paraphrases from
independent annotators, OCR perturbations and a larger held-out corpus.
