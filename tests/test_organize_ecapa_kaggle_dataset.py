"""Tests for task-aware ECAPA Kaggle dataset organizer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.organize_ecapa_kaggle_dataset import (
    PROTOCOL_LAYOUT,
    organize_kaggle_dataset,
)
from src.utils.files import sha256_file


FIELDS = ("audio_id", "audio_path", "normalized_speaker_id", "checksum")


def _portable_fixture(root: Path) -> Path:
    source = root / "source"
    metadata = source / "metadata"
    metadata.mkdir(parents=True)
    for index, metadata_name in enumerate(PROTOCOL_LAYOUT):
        audio = source / "audio" / "raw" / f"{index}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"audio-{index}".encode())
        with (metadata / metadata_name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "audio_id": str(index),
                    "audio_path": audio.relative_to(source).as_posix(),
                    "normalized_speaker_id": f"speaker_{index}",
                    "checksum": sha256_file(audio),
                }
            )
    (source / "manifest.json").write_text(
        json.dumps({"dataset": "ecapa_kaggle_v2", "speaker_audio": 7}) + "\n",
        encoding="utf-8",
    )
    return source


def test_organizes_protocols_into_self_describing_folders(tmp_path: Path) -> None:
    source = _portable_fixture(tmp_path)
    output = tmp_path / "organized"

    manifest = organize_kaggle_dataset(source_root=source, output_root=output)

    assert manifest["dataset"] == "ecapa_kaggle_split_v1"
    assert manifest["speaker_audio"] == 7
    copied_paths = []
    for metadata_name, leaf in PROTOCOL_LAYOUT.items():
        metadata_path = output / leaf / "metadata.csv"
        with metadata_path.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        copied = output / row["audio_path"]
        assert copied.is_file()
        assert sha256_file(copied) == row["checksum"]
        assert manifest["protocols"][metadata_name]["metadata_path"] == (
            leaf / "metadata.csv"
        ).as_posix()
        copied_paths.append(copied)
    assert len(set(copied_paths)) == 7
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source = _portable_fixture(tmp_path)
    output = tmp_path / "organized"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        organize_kaggle_dataset(source_root=source, output_root=output)
