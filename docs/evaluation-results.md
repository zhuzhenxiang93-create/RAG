# Lite V1 evaluation results

Run ID: `20260726T043925Z-1cb9c719`
Date: 2026-07-26
Environment: Windows, Python 3.11 project virtual environment
Corpus: 7 manually authored Markdown documents
Questions: 14 total, 12 answerable

These are small engineering-baseline results. They are not production metrics and must
not be described as general embedding, reranker or LLM quality.

## Retrieval

| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | nDCG@5 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 1.41 ms |
| Lite Dense | 0.9583 | 0.8750 | 1.0000 | 1.0000 | 0.9692 | 1.24 ms |
| RRF | 0.9583 | 0.8750 | 1.0000 | 1.0000 | 0.9692 | 1.29 ms |
| RRF + Lite Reranker | 0.9583 | 0.8750 | 1.0000 | 1.0000 | 0.9692 | 1.45 ms |
| Adaptive | 0.9583 | 0.8750 | 1.0000 | 1.0000 | 0.9692 | 1.51 ms |

BM25 is strongest on this small corpus because the questions and documents share exact
identifiers and terminology. The hashing-based Lite Dense baseline is not a semantic
embedding model, and neither RRF nor Adaptive improves the top rank here. This result is
reported rather than hidden.

## Trustworthy QA

| Metric | Result |
| --- | ---: |
| Decision accuracy | 0.7143 |
| Abstention precision | 1.0000 |
| Abstention recall | 1.0000 |
| Conflict precision | 1.0000 |
| Conflict recall | 1.0000 |
| Citation document precision | 1.0000 |
| Citation document recall | 1.0000 |

Four answerable questions were conservatively returned as `clarify` despite retrieving
and citing the correct source. Their confidence scores were 0.4800, 0.5148, 0.5078 and
0.4800, below the fixed 0.52 answer threshold. The threshold was not lowered after
seeing this test set because that would be test-set tuning.

## Required next evaluation

- split threshold calibration and final test sets;
- add independent paraphrases and more distractor documents;
- add page/chunk-level evidence labels;
- add OCR corruption cases;
- evaluate Qwen3 Embedding and the real Cross-Encoder reranker;
- report confidence intervals over a substantially larger held-out set.
