# Enterprise router sample data

`enterprise_router_train.sample.jsonl` contains a tiny, manually authored fictional
company set for unit tests and training-path smoke tests only.

It is not part of EnterpriseRAG-Bench, is not large enough for a meaningful LoRA
experiment, and must not be used to report model-quality metrics.

For the real experiment:

1. Use the EnterpriseRAG-Bench generation framework to create a separate training
   company and its question set.
2. Keep the official Redwood `questions.jsonl` in an evaluation-only directory.
3. Run `scripts/prepare_enterprise_router.py`; it will fail on exact question-ID or
   normalized-text overlap.
4. Train on the generated company and evaluate once on the official benchmark.
