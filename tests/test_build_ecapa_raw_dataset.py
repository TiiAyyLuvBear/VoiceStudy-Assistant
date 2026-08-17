"""Tests for metadata-only ECAPA raw dataset creation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_ecapa_raw_dataset import METADATA_FILES, build_ecapa_raw_dataset
from src.utils.files import sha256_file


FIELDS = (
    "audio_id",
    "audio_path",
    "original_split",
    "locale",
    "speaker_id",
    "speaker_sex",
    "speaker_age",
    "transcript",
    "intent",
    "scenario_str",
    "duration_sec",
    "sample_rate",
    "num_channels",
    "is_valid",
)


def _source_dataset(root: Path) -> tuple[Path, Path]:
    audio_root = root / "audio"
    rows = []
    for speaker_index, audio_count in enumerate((6, 6, 4)):
        for audio_index in range(audio_count):
            audio_id = f"s{speaker_index}-a{audio_index}"
            relative = Path("dev") / f"{audio_id}.wav"
            source = audio_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"audio-{audio_id}".encode())
            rows.append(
                {
                    "audio_id": audio_id,
                    "audio_path": relative.as_posix(),
                    "original_split": "validation",
                    "locale": "vi-VN",
                    "speaker_id": f"original-{speaker_index}",
                    "speaker_sex": "Female",
                    "speaker_age": "30",
                    "transcript": f"sample {audio_id}",
                    "intent": "sample",
                    "scenario_str": "sample",
                    "duration_sec": "3.0",
                    "sample_rate": "48000",
                    "num_channels": "1",
                    "is_valid": "True",
                }
            )
    inventory = root / "data_inventory.csv"
    with inventory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return inventory, audio_root


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_builds_large_balanced_metadata_without_copying_audio(tmp_path: Path) -> None:
    inventory, audio_root = _source_dataset(tmp_path / "source")
    output = tmp_path / "dataset"
    manifest = build_ecapa_raw_dataset(
        inventory_path=inventory,
        audio_root=audio_root,
        output_root=output,
        split_counts={"train": 3, "validation": 1, "test": 1},
    )

    assert manifest["selection"] == {
        "inventory_rows": 16,
        "metadata_eligible_rows": 16,
        "selected_speakers": 2,
        "selected_rows": 10,
    }
    expected = {"train": 6, "validation": 2, "test": 2}
    selected_ids: set[str] = set()
    for split, count in expected.items():
        rows = _read(output / METADATA_FILES[split])
        assert len(rows) == count
        assert {row["split"] for row in rows} == {split}
        assert {row["sample_rate"] for row in rows} == {"48000"}
        for row in rows:
            assert row["audio_id"] not in selected_ids
            selected_ids.add(row["audio_id"])
            source = audio_root / row["audio_path"]
            assert source.is_file()
            assert row["checksum"] == sha256_file(source)

    assert not (output / "audio").exists()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    report = _read(output / "selection_report.csv")
    rejected = [row for row in report if row["status"] == "rejected_insufficient_speaker_audio"]
    assert len(rejected) == 4


def test_refuses_to_overwrite_existing_dataset(tmp_path: Path) -> None:
    inventory, audio_root = _source_dataset(tmp_path / "source")
    output = tmp_path / "dataset"
    output.mkdir()

    try:
        build_ecapa_raw_dataset(
            inventory_path=inventory,
            audio_root=audio_root,
            output_root=output,
            split_counts={"train": 3, "validation": 1, "test": 1},
        )
    except FileExistsError as error:
        assert "Refusing to overwrite" in str(error)
    else:
        raise AssertionError("Expected overwrite refusal")
