# Structured Sentencing Research Baseline

## Scope

The feature predicts a structured historical-outcome estimate from case facts and an optional
accusation. It is a research prototype, not legal advice and not a substitute for a judge or lawyer.
An accusation alone is insufficient: the service returns `insufficient_information` instead of a
fabricated term or fine.

## Data flow

```text
legacy CAIL JSONL
  -> normalization
  -> charge + sentencing conclusion masking
  -> normalized-fact deduplication and penalty-conflict audit
  -> grouped train/validation/test split
  -> processed_v2_1_1 (fine retained; partially anonymized verdict leakage removed)
  -> hierarchical baseline + sentencing evidence index
  -> consistency-constrained Pydantic response
```

The outcome labels are `death_penalty`, `life_imprisonment`, `imprisonment_months`, and `fine`.
`fine=null`, `fine=0`, and `fine>0` remain distinct. Invalid and negative fines become missing.
Duplicate facts with conflicting outcomes are listed in
`reports/data_v2_1/duplicate_penalty_conflicts.json`; a deterministic modal outcome is used for the
single deduplicated training row.

## Model

The first reproducible baseline uses one character n-gram hashing encoder and separate linear
heads for sentence type, imprisonment bucket, log imprisonment months, fine imposed, fine bucket,
and log fine amount. Fine is explicitly two-stage. Accusations are deterministically omitted from
half of training inputs so the same model supports known- and unknown-accusation modes.

Inference applies hierarchy rules: death/life/exempt outcomes cannot carry fixed-term months, and
`fine.imposed=false` cannot carry an amount. Low-confidence or accusation-unknown results require
manual review.

## Known limitations

- Source provenance is `legacy_local_file_unverified`.
- Legacy CAIL labels do not distinguish fixed-term imprisonment, detention and control. Positive
  month labels are conservatively exposed as `fixed_term`.
- No reliable probation label is present, so probation remains unknown rather than guessed.
- Historical outcomes can encode temporal, regional and social bias.
- Intervals are learned buckets, not legal guarantees or calibrated judicial uncertainty.
- The existing QLoRA charge classifier is injectable but evaluated separately from sentencing.
