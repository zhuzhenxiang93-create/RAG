# Charge classifier regression diagnosis

## Scope

This diagnosis uses the complete current `processed_v2` validation split and never reads the test
split for threshold selection or model choice. The 30-case regression suite is entirely synthetic,
contains no direct identifiers, and is not a substitute for a human legal benchmark.

Rebuild the report with:

```bash
PYTHONPATH=src python scripts/diagnose_classifier_regression.py \
  --output reports/classification/classifier_regression_diagnosis.json
```

## Experiment identity result

The 202-label mapping is contiguous and its SHA-256 is identical in `processed_v2` and
`processed_v2_1_1`. Adapter, threshold, mapping, validation, and configuration hashes are recorded
in the JSON report. The identity gate nevertheless fails:

- the historical evaluation reports 15,081 source validation rows;
- the current manifest and file contain 15,032 validation rows;
- the historical 2,000-row subset hash cannot be reproduced from the current file;
- the threshold artifact is not cryptographically linked to the checkpoint manifest.

Therefore historical metrics remain valid only for their historical artifact snapshot and must not
be presented as a fully reproducible score on the current file.

## Complete current validation result

Using the historical thresholds on all 15,032 current validation rows produced Micro-F1 0.8501,
Macro-F1 0.6425, precision 0.8553, and recall 0.8449. Fixed 0.5 produced Micro-F1 0.8563 and
Macro-F1 0.6462. Validation-only retuning produced Micro-F1 0.8598 and Macro-F1 0.6737, but that is
an in-sample diagnostic and is not test performance. Retuning changed 155 of 202 thresholds.

The old validation logits can still be replayed coherently with the old thresholds (Micro-F1
0.8507, Macro-F1 0.6183). This and the full current result show that the LoRA weights load and the
classifier has not globally collapsed.

## Synthetic regression result

The 30-case suite passed 20 strict threshold checks (66.7%); the expected charge appeared in Top-5
for 88.9% of non-abstention cases. All negation and insufficient-information cases abstained, but
both multi-label cases failed strict completeness. Main failures were:

- short synthetic theft, robbery, snatch, unlawful detention, and intentional homicide facts ranked
  the expected charge but remained below the historical threshold;
- the gambling example was confused with operating a casino;
- both multi-label examples returned only one expected charge;
- the prior-charge example ranked the current intentional-injury charge first but abstained;
- the attempted-theft example did not rank theft in Top-5.

This suite is intentionally difficult and should be kept fixed as a regression gate, not used as a
threshold-tuning set.

## Preprocessing correction

Training and batch evaluation use 2,048-token head-tail truncation. Online inference previously used
plain tokenizer head truncation. Online inference now calls the same `truncate_fact` contract and
the behavior is covered by a unit test.

## Decision

Do not retrain solely because of the original short theft example. First create a new validation
threshold candidate tied to the exact current manifest, then evaluate it once on an untouched test
split. Do not overwrite the historical threshold file or compare a validation-tuned result directly
with a test score.
