"""Run the built-in Lite benchmark."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="lite_v1", choices=["lite_v1"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation",
    )
    arguments = parser.parse_args()
    result = EvaluationRunner(PROJECT_ROOT, arguments.output_dir).run(
        arguments.benchmark
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
