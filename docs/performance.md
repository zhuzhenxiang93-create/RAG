# Performance baseline

## Reproduce

Start the Lite service, then run:

```powershell
.\.venv\Scripts\python.exe -B scripts\load_test.py `
  --bootstrap `
  --scenario chat `
  --requests 100 `
  --concurrency 8
```

Available scenarios are `health`, `search` and `chat`. Each request carries a unique
request ID. The JSON artifact records success rate, throughput, mean/P50/P95/max latency,
errors and the run configuration.

## Interpretation rules

- This is an HTTP concurrency smoke baseline on one local machine.
- It does not model production documents, network latency, multiple workers or GPUs.
- Results from different hardware or background load are not directly comparable.
- Run the command at least three times before using it for capacity decisions.
- Do not place a throughput number on a resume unless the machine, scenario, concurrency
  and success rate are stated.

## Latest verified target-machine run

Artifact: `artifacts/load/20260726T050737Z-chat.json`

| Field | Result |
| --- | ---: |
| Environment | Windows 10 build 26200, Python 3.11.3, 32 logical CPUs |
| Scenario | Lite extractive `/api/chat` |
| Requests / concurrency | 100 / 8 |
| Success | 100 / 100 |
| Throughput | 323.360 requests/s |
| Mean latency | 23.544 ms |
| P50 latency | 21.097 ms |
| P95 latency | 38.307 ms |
| Max latency | 41.606 ms |

This run used the 7-document Lite corpus, deterministic hashing retrieval and no external
LLM, embedding service or network hop. It is useful for catching HTTP concurrency and
thread-safety regressions only. It must not be described as Full-mode RAG performance or
production capacity.
