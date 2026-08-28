# English Resume Entry

**LegalMind-RAG: Evidence- and Statute-Grounded Criminal Case Analysis | LLM Algorithm Engineer**

- Rebuilt a CAIL2018-derived data pipeline with Unicode normalization, target-leakage masking, case-family deduplication, and group-aware splitting, producing 120,393/15,032/15,097 multi-label records across 202 charges with zero exact, normalized-text, or case-family overlap between splits.
- Implemented reproducible Qwen3-4B QLoRA classification with dynamic padding, length-grouped batching, and Head+Tail/sentence-aware long-context strategies; profiled batch sizes 2/4/8 on an RTX 4090D and selected batch 4 with gradient accumulation 8, while a historical stratified 10K run achieved 0.8296 Micro-F1 on its independent test subset.
- Built a train-only case corpus with 128,107 chunks and a 452-clause statute index, then evaluated BM25, Qwen3 dense retrieval, RRF fusion, and Qwen3 reranking on the same pilot corpus; used the finding that dense retrieval outperformed the fused variants to keep routing configurable instead of assuming additional stages always help.
- Added strict JSON contracts, retrieval-grounded citation validation, low-confidence fallback, and mandatory human-review controls; converted generation failures into explicit bad-case categories and defined evidence-based entry criteria for full SFT and any later DPO work.

> The 0.8296 result belongs to the preserved historical stratified 10K experiment, not the unrun v2 full-data experiment. Dataset-license and proxy-qrels limitations are documented in the repository.
