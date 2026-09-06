from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.pipeline.contracts import LegalCaseAnalysisResponse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("schemas/legal-case-analysis-v1.schema.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(LegalCaseAnalysisResponse.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
