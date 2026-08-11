"""Test the ASR split builder against the current inventory format."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_asr_splits import _load_inventory, _select


def test_selects_by_project_split_and_preserves_original_split(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "data_inventory.csv"
    fields = (
        "audio_id",
        "audio_path",
        "original_split",
        "project_split",
        "speaker_id",
        "transcript",
        "intent",
        "is_valid",
    )
    rows = [
        {
            "audio_id": f"id{index}",
            "audio_path": f"dev/{index}.wav",
            "original_split": "validation",
            "project_split": project_split,
            "speaker_id": f"spk{index % 2}",
            "transcript": f"sample {index}",
            "intent": "sample",
            "is_valid": valid,
        }
        for index, (project_split, valid) in enumerate(
            [
                ("VALIDATION", "true"),
                ("VALIDATION", "true"),
                ("VALIDATION", "false"),
                ("TEST", "true"),
                ("TEST", "true"),
            ]
        )
    ]
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    loaded, columns = _load_inventory(inventory)
    validation = _select(loaded, columns, {"validation"}, 2, 42)
    test = _select(loaded, columns, {"test"}, 2, 43)

    assert {row["audio_id"] for row in validation} == {"id0", "id1"}
    assert {row["audio_id"] for row in test} == {"id3", "id4"}
    assert {row["original_split"] for row in validation + test} == {"validation"}
    assert {row["project_split"] for row in validation} == {"VALIDATION"}
    assert {row["project_split"] for row in test} == {"TEST"}


def test_resolves_inventory_paths_from_audio_root(tmp_path: Path) -> None:
    audio_root = tmp_path / "data" / "audio"
    audio_path = audio_root / "dev" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    inventory = tmp_path / "data_inventory.csv"
    fields = (
        "audio_path",
        "original_split",
        "project_split",
        "transcript",
        "is_valid",
    )
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": "dev/sample.wav",
                "original_split": "validation",
                "project_split": "VALIDATION",
                "transcript": "sample",
                "is_valid": "true",
            }
        )

    loaded, columns = _load_inventory(inventory)
    selected = _select(loaded, columns, {"validation"}, 1, 42, audio_root)

    assert Path(selected[0]["audio_path"]) == audio_path


def test_none_size_selects_all_usable_rows(tmp_path: Path) -> None:
    inventory = tmp_path / "data_inventory.csv"
    fields = (
        "audio_path",
        "original_split",
        "project_split",
        "transcript",
        "is_valid",
    )
    rows = [
        {
            "audio_path": f"dev/{index}.wav",
            "original_split": "validation",
            "project_split": "VALIDATION",
            "transcript": f"sample {index}",
            "is_valid": "true" if index < 3 else "false",
        }
        for index in range(4)
    ]
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    loaded, columns = _load_inventory(inventory)
    selected = _select(loaded, columns, {"validation"}, None, 42)

    assert len(selected) == 3
    assert {row["audio_id"] for row in selected} == {"0", "1", "2"}
