"""Small dependency-free HTTP concurrency baseline for a running Lite service."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


PAYLOADS = {
    "health": ("GET", "/api/health", None),
    "search": (
        "POST",
        "/api/search",
        {
            "query": "SEV-1 生产事故要求多少分钟内响应？",
            "strategy": "adaptive",
            "top_k": 5,
        },
    ),
    "chat": (
        "POST",
        "/api/chat",
        {
            "query": "SEV-1 生产事故要求多少分钟内响应？",
            "strategy": "adaptive",
            "top_k": 5,
        },
    ),
}


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999) - 1))
    return ordered[index]


def send(base_url: str, scenario: str, timeout: float) -> Dict:
    method, path, payload = PAYLOADS[scenario]
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": "load-{}".format(uuid4().hex[:16]),
        },
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
            error: Optional[str] = None
    except HTTPError as exc:
        status = exc.code
        error = "http_{}".format(exc.code)
    except (URLError, TimeoutError) as exc:
        status = 0
        error = type(exc).__name__
    return {
        "status": status,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "error": error,
    }


def bootstrap(base_url: str, timeout: float) -> None:
    request = Request(
        base_url.rstrip("/") + "/api/demo/bootstrap",
        data=b"",
        method="POST",
        headers={"X-Request-ID": "load-bootstrap"},
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("Demo bootstrap failed with HTTP {}".format(response.status))


def run(
    base_url: str,
    scenario: str,
    requests: int,
    concurrency: int,
    timeout: float,
) -> Dict:
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send, base_url, scenario, timeout) for _ in range(requests)
        ]
        results = [future.result() for future in as_completed(futures)]
    elapsed = perf_counter() - started
    successful = [item for item in results if 200 <= item["status"] < 300]
    latencies = [item["latency_ms"] for item in successful]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "scenario": scenario,
        "base_url": base_url,
        "requests": requests,
        "concurrency": concurrency,
        "successful": len(successful),
        "failed": requests - len(successful),
        "success_rate": round(len(successful) / requests, 4),
        "throughput_rps": round(requests / elapsed, 3),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "errors": [item for item in results if item["error"]][:20],
        "warning": (
            "Single-machine Lite engineering baseline; do not present as production "
            "capacity or compare across different hardware."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=sorted(PAYLOADS), default="chat")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be positive")
    if args.bootstrap:
        bootstrap(args.base_url, args.timeout)
    result = run(
        args.base_url,
        args.scenario,
        args.requests,
        args.concurrency,
        args.timeout,
    )
    output = args.output or (
        Path("artifacts")
        / "load"
        / "{}-{}.json".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), args.scenario
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["artifact_path"] = str(output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
