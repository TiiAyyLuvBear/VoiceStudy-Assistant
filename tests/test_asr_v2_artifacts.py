from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils import sha256_file


CONFIG = Path("reports/asr/v2/asr_test_config.json")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_asr_v2_uses_all_valid_validation_and_test_rows() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validation_config = config["datasets"]["validation"]
    test_config = config["datasets"]["test"]
    validation_path = Path(validation_config["path"])
    test_path = Path(test_config["path"])
    validation = _rows(validation_path)
    test = _rows(test_path)

    assert len(validation) == validation_config["row_count"] == 322
    assert len(test) == test_config["row_count"] == 249
    assert all(row["project_split"] == "VALIDATION" for row in validation)
    assert all(row["project_split"] == "TEST" for row in test)
    assert {row["audio_path"].casefold() for row in validation}.isdisjoint(
        {row["audio_path"].casefold() for row in test}
    )
    assert sha256_file(validation_path) == validation_config["sha256"]
    assert sha256_file(test_path) == test_config["sha256"]


def test_asr_v2_locked_checksums_match_workspace() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    for path, expected in config["locked_files"].items():
        assert sha256_file(path) == expected
    manifest = config["datasets"]["split_manifest"]
    assert sha256_file(manifest["path"]) == manifest["sha256"]
