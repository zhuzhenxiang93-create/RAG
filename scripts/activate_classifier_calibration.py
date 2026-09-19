from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-labels", type=int, default=202)
    args = parser.parse_args()
    values = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if len(values) != args.expected_labels:
        raise ValueError(f"expected {args.expected_labels} thresholds, got {len(values)}")
    replacement = f"thresholds: {args.thresholds.as_posix()}"
    for config in args.configs:
        text = config.read_text(encoding="utf-8")
        lines = text.splitlines()
        matches = [index for index, line in enumerate(lines) if line.strip().startswith("thresholds:")]
        if len(matches) != 1:
            raise ValueError(f"expected one thresholds entry in {config}, got {len(matches)}")
        index = matches[0]
        indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
        lines[index] = indent + replacement
        output = "\n".join(lines) + "\n"
        handle, temporary = tempfile.mkstemp(prefix=config.name, dir=config.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(output)
            os.replace(temporary, config)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(f"activated {args.thresholds} in {config}")


if __name__ == "__main__":
    main()
