# Deterministic Pydantic Pipeline

The default runtime does not require a generative SFT model. Classification, retrieval,
temporal statute filtering, structured sentencing, citation checks, and the evidence
firewall produce a versioned `LegalCaseAnalysisResponse` directly.

```text
de-identified fact
  -> charge classification
  -> case retrieval and statute retrieval
  -> temporal/source filtering
  -> structured sentencing baseline
  -> deterministic grounded analysis
  -> citation/privacy/cross-field validation
  -> LegalCaseAnalysisResponse.model_dump_json()
```

The final model forbids unknown fields and verifies that:

- case and statute evidence remain in separate collections;
- every cited case and article exists in the returned evidence;
- returned case summaries contain no detected direct identifier;
- a passed evidence firewall has valid citations, an as-of date, and applicable statutes;
- a component that requires review cannot be hidden by the top-level response;
- death, life, and exempt sentences never contain fabricated imprisonment months;
- a negative or missing fine decision cannot silently carry an amount;
- numeric ranges are ordered and all timing values are non-negative.

Export the public JSON Schema with:

```bash
python scripts/export_pipeline_schema.py \
  --output schemas/legal-case-analysis-v1.schema.json
```

The historical generation/SFT modules and adapters remain available for reproducibility,
but they are not loaded by `configs/pipeline/default.yaml` and are not part of the default
analysis path.
