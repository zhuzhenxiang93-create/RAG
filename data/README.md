# Data

Place the original CAIL files under `data/raw/` or update `configs/data.yaml`.
Large datasets, trained weights, and indexes are intentionally excluded from Git.

Expected raw JSONL fields:

```json
{"fact":"案件事实", "meta":{"accusation":["盗窃"], "relevant_articles":[264], "term_of_imprisonment":{"imprisonment":8}}}
```

The legacy local files may also use top-level `accusation`; the normalizer supports both.
Run `python scripts/audit_data.py --config configs/data.yaml` before preprocessing.
