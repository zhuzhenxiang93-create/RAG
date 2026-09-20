# LegalMind-RAG Code Audit — 2026-09-20

## Scope

Audited branch: `codex/data-v3-school` at commit `c2b499919d51f631b9fb006e6b518f4ef716d593`.

This audit treats source code and executable configuration as the primary truth, README/interview
material as secondary truth, and historical chat descriptions as non-authoritative unless matched by
the repository.

A cleanup branch was created from the audited branch:

`audit/bf16-lora-cleanup-20260919`

## Executive summary

The current school classifier is **BF16 LoRA, not QLoRA**. The model config sets
`quantization.load_in_4bit: false`, so Qwen3-4B is loaded in BF16 and only LoRA adapters plus the
202-class `score` head are trained. Historical 4-bit QLoRA experiments remain in the repository
for reproducibility.

The grounded generation path is real: case evidence, statute evidence, and the structured sentencing
baseline are assembled into `EvidencePacketV1` and passed to an OpenAI-compatible generator.
Generated claims are checked against a strict Pydantic schema and an evidence-ID allowlist.

The largest correctness problem found was in evaluation: `scripts/run_classifier_eval.py` and the
regression diagnostic unconditionally reloaded the base model in 4-bit mode. That means a BF16 LoRA
adapter could be evaluated on a quantized base different from the base precision used for training.
This branch fixes the loader so training, evaluation, regression diagnosis, and inference respect the
same `load_in_4bit` flag.

## Findings

### A1 — High — BF16 LoRA was evaluated on an unconditional 4-bit base

**Status: fixed on audit branch.**

School training config:

- `configs/model_qwen3_4b_school_lora.yaml`
- `quantization.load_in_4bit: false`
- BF16 base + LoRA `r=16`, alpha 32, dropout 0.05, `all-linear`
- classifier head `score` saved with the adapter

Before the audit, both `scripts/run_classifier_eval.py` and
`scripts/diagnose_classifier_regression.py` always constructed
`BitsAndBytesConfig(load_in_4bit=True)`.

Impact: test/validation logits were not guaranteed to represent the exact trained BF16 LoRA model.

Fix:

- added `src/legalmind/models/peft_classifier.py`;
- training/evaluation/diagnosis now dispatch from model config;
- `load_in_4bit=false` loads BF16 LoRA;
- `load_in_4bit=true` keeps historical QLoRA reproducibility;
- corrected evaluation output is isolated under
  `artifacts/results/qwen3_4b_lora_school_full_eval_bf16`.

**Interview rule:** do not report an old independent-test score as the final BF16 LoRA score until the
corrected evaluator has been run.

### A2 — High — Full LoRA checkpoint used historical 10K QLoRA thresholds

**Status: fixed in pipeline configuration; corrected threshold artifact still needs to be generated.**

Before the audit:

`qwen3_4b_lora_school_full/checkpoint-7530`
was paired with
`artifacts/results/classifier_10k_2k/thresholds.json`.

Those thresholds belong to a different historical experiment identity.

Fix:

- pipeline uses the full LoRA adapter directory;
- label mapping points to `processed_v2_1_1`;
- thresholds point to the corrected BF16 full-evaluation output.

If the new threshold file is absent, online inference falls back to 0.5 thresholds rather than
silently using the historical QLoRA thresholds.

### A3 — Medium — QLoRA names obscured the actual training method

**Status: fixed for the current main path; historical names retained only for compatibility.**

The school Slurm job was named `legalmind-lora` and used a non-quantized model config, but called
`scripts/train_qlora.py` and `build_qlora_classifier`.

Fix:

- canonical school entrypoint is now `scripts/train_lora.py`;
- generic PEFT builder is `build_adapter_classifier`;
- `src/legalmind/models/qlora.py` is now an explicit backwards-compatibility shim;
- experiment manifests record `adapter_method=lora` or `qlora` from config.

### A4 — Medium — Two competing pipeline implementations existed

**Status: fixed on audit branch.**

The repository contained both:

- legacy `src/legalmind/pipeline.py`: classifier → HybridRetriever → optional generator;
- current `src/legalmind/pipeline/`: classifier → charge-partitioned retrieval → statute filtering →
  sentencing → grounded generation → evidence firewall.

The legacy top-level module was removed to make the package implementation the single source of truth.

### A5 — High documentation risk — Default online retrieval is not the Hybrid pipeline

**Status: open by design; documentation corrected.**

The repository contains working experimental components for:

- BM25;
- Dense embeddings;
- FAISS;
- RRF;
- CrossEncoder/API reranking.

However `legalmind analyze` currently loads `ChargePartitionedRetriever` when the partitioned
index exists. Its online score is:

- 0.55 normalized BM25;
- 0.25 predicted-charge confidence;
- 0.15 structured similarity (amount, sentencing factors, year);
- 0.05 single/multi-charge cardinality match.

Therefore the safe claim is:

> The project implemented and evaluated BM25/Dense/RRF/Reranker modules, while the current default
> online path uses charge-gated BM25 with deterministic structured reranking.

Do **not** say the current default E2E path is
`BM25 + Dense → RRF → Qwen Reranker`.

### A6 — Confirmed strength — RAG evidence is genuinely injected into the LLM

**Status: implemented and tested.**

`LegalMindPipeline.analyze` constructs an `EvidencePacketV1` containing:

- de-identified current case facts;
- predicted accusations;
- Top-K retrieved case evidence;
- case penalty metadata;
- effective verified statute evidence;
- structured sentencing baseline.

`GroundedAnalysisService` serializes this packet with the output JSON schema and passes it to the
OpenAI-compatible model.

`tests/test_pipeline_v2.py::test_pipeline_sends_retrieval_and_sentencing_context_to_grounded_llm`
checks that the LLM-facing packet includes both retrieved penalties and the sentencing baseline.

This is a real grounded-generation loop, not “retrieve and display only”.

### A7 — Confirmed strength — Citation and privacy constraints are hard validation, not prompt-only

**Status: implemented.**

The generated analysis is validated for:

- strict Pydantic schema;
- unknown evidence IDs;
- case/statute evidence-type misuse;
- statute source/date validity;
- direct identifier leakage;
- analyzed outputs with no supporting claims.

The final response contract performs another cross-field evidence allowlist check.

This is one of the strongest defensible project highlights.

### A8 — Safety/coverage trade-off — Current statute gate is intentionally strict

**Status: open operational dependency.**

A statute is accepted only when:

- source URL is from an authoritative government/judicial domain;
- source status is `human_verified_official` or `verified_official`;
- legal status is effective;
- an `as_of_date` is supplied;
- effective/expiry dates allow use on that date.

Automatically downloaded but unreviewed statute chunks are rejected. Therefore a demo without
verified statute metadata or without `--as-of-date` can legitimately return no applicable statute
and require manual review.

This is safer than silently using unverified law, but the verified statute asset must exist for a
full legal-basis demo.

### A9 — Medium — Long-case classifier still uses Head+Tail, not overlap inference/training

**Status: open.**

Training token audit reports:

- median 248 tokens;
- p99 1,904;
- 1,059 / 120,468 = 0.879% exceed 2,048;
- maximum 43,754.

The classifier currently uses 2,048-token Head+Tail truncation in training, evaluation, and online
inference. Sliding-window/overlap case-level aggregation is not implemented.

Safe claim:

> Head+Tail is the current reproducible baseline; sliding-window overlap is a planned long-document
> improvement and should be evaluated first on the >2,048-token subset.

### A10 — Medium — SimHash near duplicates are audited, not split-grouped

**Status: open.**

The data builder merges exact/normalized duplicates before multilabel group splitting. This supports
claims of zero exact/normalized/dedup-group overlap.

SimHash near-duplicate candidates are generated **after** the split for audit and are not used as a
grouping key.

Therefore do not claim “all near duplicates are guaranteed isolated across splits.”

### A11 — Medium — GitHub does not contain the final BF16 LoRA evaluation artifact

**Status: open.**

The repository contains the school LoRA config and pipeline references, but the adapter weights,
processed JSONL data, and corrected full test report are intentionally excluded from Git.

Consequently the repository by itself does not prove a final BF16 LoRA test Micro-F1. The next
authoritative artifact should be generated by the corrected evaluation job and saved with:

- adapter identity/hash;
- dataset manifest/hash;
- label mapping hash;
- validation-derived thresholds;
- untouched test metrics.

### A12 — Confirmed limitation — Structured sentencing is a statistical baseline, not legal reasoning

**Status: implemented and appropriately caveated.**

The sentencing model uses character 2–4 gram `HashingVectorizer` features with hierarchical
`SGDClassifier` / `SGDRegressor` heads.

Untouched-test results committed in the repository include:

- sentence type accuracy ≈ 0.938;
- sentence type Macro-F1 ≈ 0.545;
- fixed-term month MAE ≈ 23.63 months;
- median absolute error 9 months;
- fine-imposed F1 ≈ 0.769;
- fine amount median absolute error CNY 5,000.

The model cannot perform Transformer-style long-context reasoning. Its correct role is a reproducible
statistical baseline and anomaly/reference signal for grounded generation.

## Security and repository hygiene

- no real API key is committed in `.env.example`;
- `.env`, model weights, raw local datasets, FAISS/PKL/NPY artifacts and trained adapters are ignored;
- retrieved case text is de-identified before presentation/LLM packaging;
- untrusted pickle-style BM25 assets should never be loaded from external sources;
- raw data license/provenance is still not fully closed and remains a release blocker.

## What the project can safely claim today

1. Qwen3-4B **BF16 LoRA** multi-label charge classification is the current school-model design.
2. Data construction performs target leakage masking, normalized deduplication and group-isolated
   train/validation/test splitting.
3. Experimental Hybrid retrieval modules exist and have pilot results.
4. The current online route is charge-partitioned BM25 + structured reranking.
5. Structured sentencing is a separate statistical baseline.
6. Retrieved cases, penalties, verified statutes and the baseline can be injected into an LLM
   evidence packet.
7. Pydantic + evidence-ID allowlists + statute temporal/source checks + privacy checks constrain the
   final grounded output.
8. Low-confidence or unsupported states explicitly trigger manual review.

## What the project should not claim yet

- “The current online retriever is BM25 + Dense + RRF + Qwen Reranker.”
- “All near-duplicate leakage is eliminated.”
- “The final BF16 LoRA test score is X” until the corrected evaluator runs.
- “The statute corpus is fully legally reviewed.”
- “The sentencing baseline understands full legal context.”
- “Sliding-window overlap classification has been implemented.”
- “The system is production legal advice.”

## Immediate next actions

1. Run the corrected BF16 LoRA independent evaluation and archive its metrics/manifests.
2. Point production/demo thresholds only to that exact evaluation artifact.
3. Run three grounded E2E acceptance cases with a verified statute index and `as_of_date`.
4. Add a long-text >2,048-token overlap-inference experiment before considering retraining.
5. If Hybrid retrieval is intended to be a product highlight, add a configuration switch that
   actually instantiates `HybridRetriever` in `build_pipeline`, then evaluate it against the same
   reviewed qrels.
6. Human-review a retrieval qrels subset and the statute source/temporal metadata.
