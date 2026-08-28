from __future__ import annotations

import json

from legalmind.experiments.manifest import file_sha256, write_run_manifest


def test_experiment_manifest_and_hash(tmp_path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("stable", encoding="utf-8")
    output = tmp_path / "experiment"
    path = write_run_manifest(
        output, {"experiment_name": "unit-test", "data_hash": file_sha256(source)}
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["experiment_name"] == "unit-test"
    assert len(value["data_hash"]) == 64
