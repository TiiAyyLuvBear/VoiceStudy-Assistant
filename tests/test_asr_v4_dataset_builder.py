"""Tests for exact ASR v3-to-v4 dataset versioning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_asr_v4_dataset import build_dataset
from src.utils import canonical_csv_sha256, sha256_file


FIELDS = ("audio_id", "audio_path", "transcript", "speaker_id", "audio_sha256")


def _write_split(root: Path, split: str, speaker: str) -> Path:
    audio = root.parent / "audio" / f"{split}.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(split.encode())
    path = root / "metadata" / f"asr_finetune_{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "audio_id": split,
                "audio_path": audio.as_posix(),
                "transcript": f"câu {split}",
                "speaker_id": speaker,
                "audio_sha256": sha256_file(audio),
            }
        )
    return path


def _source_v3(tmp_path: Path) -> Path:
    root = tmp_path / "v3"
    datasets = {}
    for split in ("train", "validation", "test"):
        path = _write_split(root, split, f"speaker-{split}")
        datasets[split] = {"canonical_csv_sha256": canonical_csv_sha256(path)}
    (root / "asr_finetune_manifest.json").write_text(
        json.dumps(
            {"dataset_version": "v3", "freeze_status": "FROZEN", "datasets": datasets}
        ),
        encoding="utf-8",
    )
    return root


def test_builds_identical_v4_splits(tmp_path: Path) -> None:
    source = _source_v3(tmp_path)
    output = tmp_path / "v4"
    manifest = build_dataset(source, output)
    assert manifest["freeze_status"] == "DEVELOPMENT"
    assert manifest["experiment_protocol"]["test_is_fresh_holdout"] is False
    for split in ("train", "validation", "test"):
        source_csv = source / "metadata" / f"asr_finetune_{split}.csv"
        output_csv = output / "metadata" / f"asr_finetune_{split}.csv"
        assert canonical_csv_sha256(source_csv) == canonical_csv_sha256(output_csv)


def test_requires_frozen_v3(tmp_path: Path) -> None:
    source = _source_v3(tmp_path)
    manifest_path = source / "asr_finetune_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["freeze_status"] = "DEVELOPMENT"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN"):
        build_dataset(source, tmp_path / "v4")
