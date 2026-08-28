# Reranker hard-negative mining v1

This completed CPU data-mining experiment uses 5,000 queries from the processed-v2 train split and train-only case chunks. It produced 14,536 unreviewed triplets covering 4,882 queries. Validation and test records were not used. The complete type distribution and hashes are stored in `data/reranker/manifest.json`.

Reproduction command:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/build_reranker_data.py \
  --queries 5000 --negatives-per-query 3 --seed 42
```
