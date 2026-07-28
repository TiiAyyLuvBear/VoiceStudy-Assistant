"""Test deterministic ASR split selection bằng inventory giả lập."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_asr_splits import _load_inventory, _select


def test_selects_only_valid_official_split_deterministically(tmp_path: Path) -> None:
    inventory = tmp_path / "data_inventory.csv"
    fields = (
        "audio_id",
        "audio_path",
        "original_split",
        "speaker_id",
        "transcript",
        "intent",
        "is_valid",
    )
    rows = [
        {
            "audio_id": f"id{index}",
            "audio_path": f"audio/{index}.wav",
            "original_split": split,
            "speaker_id": f"spk{index % 2}",
            "transcript": f"câu số {index}",
            "intent": "sample",
            "is_valid": valid,
        }
        for index, (split, valid) in enumerate(
            [
                ("validation", "true"),
                ("validation", "true"),
                ("validation", "false"),
                ("test", "true"),
                ("test", "true"),
            ]
        )
    ]
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    loaded, columns = _load_inventory(inventory)
    first = _select(loaded, columns, {"validation"}, 2, 42)
    second = _select(loaded, columns, {"validation"}, 2, 42)

    assert first == second
    assert {row["original_split"] for row in first} == {"validation"}
    assert {row["audio_id"] for row in first} == {"id0", "id1"}
