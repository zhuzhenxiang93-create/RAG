# DocMind-RAG architecture

## System view

```mermaid
flowchart TB
    U["Web Workbench / REST Client"] --> M["FastAPI Middleware<br/>Request ID · Server Timing · Safe Logs"]
    M --> API["API Layer"]

    subgraph DATA["Document data plane"]
        API --> INGEST["Upload validation"]
        INGEST --> PARSER["PDF · DOCX · XLSX · TXT · Markdown Parsers"]
        PARSER --> CHUNK["Heading-aware Parent-Child Chunking"]
        CHUNK --> STORE["SQLite Metadata + JSON Artifacts + Uploaded Files"]
    end

    subgraph RETRIEVAL["Retrieval plane"]
        API --> ROUTER["Explainable Query Router"]
        STORE --> BM25["BM25 Index"]
        STORE --> DENSE["Lite Hashing Dense Index"]
        ROUTER --> BM25
        ROUTER --> DENSE
        BM25 --> RRF["Weighted RRF"]
        DENSE --> RRF
        RRF --> RERANK["Optional Lite / Full Reranker"]
    end

    subgraph TRUST["Trustworthy decision plane"]
        RERANK --> EVIDENCE["Evidence Relation Analysis"]
        EVIDENCE --> CONFLICT["Numeric · Version · Negation Conflicts"]
        CONFLICT --> CONFIDENCE["Five-factor Confidence"]
        CONFIDENCE --> DECISION["Answer · Clarify · Abstain"]
        DECISION --> CITATION["Location-aware Citations"]
    end

    subgraph EXT["Optional domain plugins"]
        LEGAL["Legal LoRA Classifier<br/>Lazy Transformers / PEFT Loading"]
        FULL["Full Embedding / Cross-Encoder / LLM<br/>Interfaces reserved, results pending"]
    end

    subgraph QUALITY["Quality and operations"]
        EVAL["Lite V1 Evaluation<br/>Raw Predictions + Metrics"]
        LOAD["Concurrent HTTP Baseline"]
        DIAG["Health + Diagnostics"]
        CI["GitHub Actions<br/>Syntax · Tests · Container Build"]
    end

    API --> LEGAL
    ROUTER -. provider upgrade .-> FULL
    API --> EVAL
    API --> DIAG
    LOAD --> API
    CI --> API
```

## Design decisions

### Keep Lite mode independently runnable

The default path requires no API key, GPU, FAISS, Transformers or OCR. Optional model
providers are delayed until their endpoint is used. This keeps development, CI and the
interview demo reproducible on a CPU machine.

### Preserve document location and hierarchy

Parsed elements retain page, section, content type and format-specific metadata. Parent
chunks preserve context while child chunks are retrieved. Child text is prefixed with
its section so that headings and entities are not lost during retrieval.

### Route before fusion

The rule-first Query Router returns an inspectable plan. Exact identifiers favor BM25;
semantic or comparison questions increase the dense channel; table-like questions use
both exact and contextual signals. Standard weighted RRF only rewards channels where a
candidate actually appears.

### Treat uncertainty as a product decision

Confidence is calculated from evidence relevance, channel agreement, ranking margin,
citation coverage and conflict penalty. It is not an LLM self-reported probability.
Conflicting evidence produces `clarify`; missing evidence produces `abstain`.

### Separate immutable benchmark data from runtime state

The repository contains only the authored Lite corpus and labels. Uploads, SQLite,
indexes, logs, model weights and evaluation artifacts are ignored. The container stores
runtime state in `/var/lib/docmind`, separate from `/app/data/eval`.

## Current boundaries

- Lite Dense is deterministic feature hashing, not a semantic embedding model.
- Lite reranking is a transparent lexical baseline, not a trained Cross-Encoder.
- Conflict analysis is rule-based and is not a trained NLI model.
- The 7-document, 14-question benchmark is a regression suite, not production evidence.
- Legal model assets were structurally validated; full inference quality is still pending.
