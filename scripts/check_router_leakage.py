"""Standalone guard that fails if training questions overlap the benchmark."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.plugins.enterprise_router.data import assert_no_benchmark_leakage, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    assert_no_benchmark_leakage(
        list(read_jsonl(args.train)),
        list(read_jsonl(args.benchmark)),
    )
    print("PASS: no exact ID or normalized-text overlap found.")


if __name__ == "__main__":
    main()
