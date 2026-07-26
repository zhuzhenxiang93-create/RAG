# Migration status

No legacy module has been copied verbatim.

Stage 4 reimplemented the useful behavior of the legacy document parser and
Parent-Child chunker behind format-specific interfaces. It intentionally excludes
hard-coded OCR credentials, import-time model loading and Pickle persistence.

Stage 5 reimplemented sparse retrieval, a deterministic dense baseline, weighted RRF,
parent lookup and reranking. The old benchmark showed that fixed Hybrid retrieval was
not always superior to Dense + Reranker, so the new Query Router exposes dynamic weights
as a testable algorithm instead of a fixed configuration.

Stage 6 adds evidence-only generation, explicit citations, deterministic confidence and
selective answering. It does not migrate the legacy multi-sampling sentence aggregation,
because averaging generated penalties or factual values is unsafe without calibrated
uncertainty and verified evidence.

Stage 7 wraps the useful legal LoRA sequence-classification path as an optional plugin.
It removes hard-coded model paths, forced CUDA, import-time loading and default Pickle
database loading. Existing model weights remain external and no unverified accuracy is
carried into the new project.

Stage 8 replaces ad-hoc console-only evaluation with an isolated, repeatable benchmark
runner that stores environment, configuration, raw predictions and aggregate metrics.
The legacy 588-question result remains historical evidence but is not mixed with the new
Lite benchmark or presented as a directly comparable result.

Stage 9 adds an offline workbench for document ingestion, evidence inspection,
selective QA and evaluation. The Lite demo bootstrap is content-idempotent and disabled
in Full mode.

Stage 10 adds request correlation, safe structured access logs, non-sensitive runtime
diagnostics, non-root container defaults, CI configuration and a dependency-free HTTP
concurrency baseline. GitHub CI and Docker execution remain pending external validation.
Legacy projects remain read-only references:

- `D:\毕业设计\file-tools-system`
- `D:\LLM\law`
