# Structured Sentencing Baseline v1.1 Evaluation

## Reproducibility

- Dataset: `processed_v2_1_1`, source status `legacy_local_file_unverified`
- Train / validation / test: 120,468 / 14,994 / 15,031
- Labels: 202 accusations; sentence outcomes include death, life, months and fine
- Seed: 42
- Model: character 2-4 gram hashing encoder + hierarchical SGD heads
- Training device: CPU (16 allocated cores); elapsed time 135.175 seconds
- Available accelerator: NVIDIA GeForce RTX 4090D, 24,564 MiB (not used by this baseline)
- Runtime: Python 3.11.15, scikit-learn 1.9.0
- Training warning: at least one SGD head reached its configured maximum iteration count
- Leakage audit: 0 residual explicit sentence markers in every split; 0 normalized cross-split overlap

## Untouched test metrics

| Task | Metric | Result |
|---|---:|---:|
| Sentence type | Accuracy | 0.9380 |
| Sentence type | Macro-F1 | 0.5411 |
| Fixed-term months | MAE | 23.63 months |
| Fixed-term months | Median AE | 9 months |
| Fixed-term interval | Bucket accuracy | 0.3404 |
| Fine imposed | Precision | 0.7164 |
| Fine imposed | Recall | 0.8305 |
| Fine imposed | F1 | 0.7693 |
| Positive fine amount | MAE | CNY 33,130.90 |
| Positive fine amount | Median AE | CNY 5,000 |
| Positive fine interval | Bucket accuracy | 0.3842 |

Sentence type was evaluated on 15,031 rows. Fixed-term duration metrics used all 14,010 true
fixed-term rows, regardless of the preceding type prediction. Fine-imposed metrics used all 15,031
rows. Fine amount metrics used all 6,515 true positive-fine rows, regardless of the preceding
fine-imposed prediction. This avoids selective evaluation on easy, correctly routed examples.

The complete confusion matrix and per-accusation breakdown are in `evaluation_v1_1.json` on the
remote server.

## Charge classifier context

Charge prediction remains the existing Qwen3-4B QLoRA component and is injected when the request
does not provide an accusation. Its historical independent 2,000-row coverage-first test reported:
Micro-F1 0.8296, Macro-F1 0.6052, micro precision 0.8434 and micro recall 0.8162. These are not a
full-distribution 15,031-row test and must not be presented as the structured-sentencing benchmark.

## Interpretation

This is a functioning baseline, not a production-grade sentencing model. Fine-imposed prediction
is useful as a first-stage signal, and median errors are substantially more informative than means
because fine values are heavy-tailed. Interval accuracies around 34-38% show that calibration needs
improvement. The source cannot reliably distinguish fixed-term imprisonment, detention and control,
and contains no trustworthy probation label, so those outputs are explicitly degraded rather than
fabricated.
