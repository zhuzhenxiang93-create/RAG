# Grounded JSON SFT v1

This stage adds an auditable large-model analysis layer without replacing the existing
classifier, retrieval, sentencing baseline, evidence firewall, or deterministic fallback.

## Runtime contract

`EvidencePacketV1` is the exclusive input boundary. It contains a de-identified fact,
`as_of_date`, upstream predictions, and a catalog of retrieved case/statute evidence with
stable `evidence_id` values. `LegalAnalysisV1` is a strict, extra-fields-forbidden JSON
contract. Every legal-basis, analogous-case, and sentencing claim must cite catalog IDs.

The post-generation validator rejects:

- malformed or schema-invalid JSON;
- unknown evidence IDs;
- case evidence used as a legal rule, or statute evidence used as an analogous case;
- non-authoritative, unverified, not-yet-effective, or expired statute citations;
- direct personal identifiers;
- an `analyzed` result with no cited support.

The service makes at most one repair attempt and then returns a deterministic
`insufficient_evidence` result. The main pipeline exposes the SFT result in `shadow` mode;
the existing deterministic output remains authoritative until the gate passes.

## Reproducible commands

```bash
python scripts/build_grounded_sft.py \
  --output data/sft_grounded_v1_smoke --rows 200 --seed 42
python scripts/audit_grounded_sft.py \
  --dataset data/sft_grounded_v1_smoke \
  --output reports/generation/sft_grounded_v1_smoke_audit.json
python -m legalmind.training.train_generator \
  --config configs/generation/qwen3_4b_grounded_sft_smoke.yaml
python scripts/evaluate_grounded_sft.py \
  --adapter artifacts/experiments/qwen3_4b_grounded_sft_smoke \
  --data data/sft_grounded_v1_smoke/test.jsonl \
  --output reports/generation/qwen3_4b_grounded_sft_smoke_eval.json
```

## Current gate status

The 200/2,000 records are artificial contract tests, not cases and not current law. They
prove formatting, abstention, citation, privacy, training, and evaluation plumbing only.
The five-step smoke adapter must not be presented as a production legal-reasoning model.
The 2,000-row pilot training remains blocked until the 200-row human legal review queue is
completed. Real-law training also remains blocked until authoritative statute versions and
effective dates are human verified.

The model is a research prototype, does not provide legal advice, and cannot replace a
judge or lawyer. Historical cases support statistical learning; current-law retrieval must
use rules verified as effective on the requested date.
