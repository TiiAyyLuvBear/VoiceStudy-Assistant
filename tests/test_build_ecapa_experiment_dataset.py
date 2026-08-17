"""Tests for ASR-disjoint ECAPA SID/SV experiment protocols."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_ecapa_experiment_dataset import (
    PROTOCOL_FILES,
    build_ecapa_dataset,
)
from scripts.extract_all_embeddings import load_protocol_rows
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


def _source_dataset(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    audio_root = root / "audio"
    rows = []
    for speaker_index in range(7):
        for audio_index in range(6):
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

    asr_paths = []
    for split in ("validation", "test"):
        path = root / f"asr_{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("audio_id", "audio_path"))
            writer.writeheader()
            if split == "validation":
                writer.writerow(
                    {
                        "audio_id": "s6-a5",
                        "audio_path": "data/audio/dev/s6-a5.wav",
                    }
                )
        asr_paths.append(path)
    return inventory, audio_root, tuple(asr_paths)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_builds_asr_disjoint_sid_and_sv_protocols(tmp_path: Path) -> None:
    inventory, audio_root, asr_paths = _source_dataset(tmp_path / "source")
    output = tmp_path / "experiment"
    manifest = build_ecapa_dataset(
        inventory_path=inventory,
        audio_root=audio_root,
        asr_metadata_paths=asr_paths,
        output_root=output,
        sid_speaker_count=2,
        sid_counts={"train": 3, "validation": 1, "test": 1},
        sv_enrolled_speakers_per_eval=1,
        sv_unknown_speakers_per_eval=1,
        sv_enrollment_audio=1,
        sv_query_audio=1,
        sv_unknown_audio=2,
    )

    expected_counts = {
        "sid_train": 6,
        "sid_validation": 2,
        "sid_test": 2,
        "sv_validation_enrollment": 1,
        "sv_validation_query": 1,
        "sv_validation_unknown": 2,
        "sv_test_enrollment": 1,
        "sv_test_query": 1,
        "sv_test_unknown": 2,
    }
    all_rows = []
    speakers_by_protocol = {}
    for protocol_key, expected in expected_counts.items():
        rows = _read(output / "metadata" / PROTOCOL_FILES[protocol_key])
        assert len(rows) == expected
        speakers_by_protocol[protocol_key] = {
            row["normalized_speaker_id"] for row in rows
        }
        all_rows.extend(rows)
        for row in rows:
            source = audio_root / row["audio_path"]
            assert source.is_file()
            assert row["checksum"] == sha256_file(source)

    assert len({row["audio_id"] for row in all_rows}) == len(all_rows) == 18
    assert "s6-a5" not in {row["audio_id"] for row in all_rows}
    sid_speakers = set().union(
        speakers_by_protocol["sid_train"],
        speakers_by_protocol["sid_validation"],
        speakers_by_protocol["sid_test"],
    )
    assert speakers_by_protocol["sid_train"] == sid_speakers
    assert speakers_by_protocol["sid_validation"] == sid_speakers
    assert speakers_by_protocol["sid_test"] == sid_speakers
    sv_groups = [
        speakers_by_protocol["sv_validation_enrollment"],
        speakers_by_protocol["sv_validation_unknown"],
        speakers_by_protocol["sv_test_enrollment"],
        speakers_by_protocol["sv_test_unknown"],
    ]
    assert all(not sid_speakers & group for group in sv_groups)
    assert all(
        not left & right
        for index, left in enumerate(sv_groups)
        for right in sv_groups[index + 1 :]
    )
    assert not (output / "audio").exists()
    loaded = load_protocol_rows(output / "metadata", tuple(PROTOCOL_FILES.values()))
    assert len(loaded) == 18
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_refuses_existing_output(tmp_path: Path) -> None:
    inventory, audio_root, asr_paths = _source_dataset(tmp_path / "source")
    output = tmp_path / "experiment"
    output.mkdir()

    try:
        build_ecapa_dataset(
            inventory_path=inventory,
            audio_root=audio_root,
            asr_metadata_paths=asr_paths,
            output_root=output,
            sid_speaker_count=2,
            sid_counts={"train": 3, "validation": 1, "test": 1},
            sv_enrolled_speakers_per_eval=1,
            sv_unknown_speakers_per_eval=1,
            sv_enrollment_audio=1,
            sv_query_audio=1,
            sv_unknown_audio=2,
        )
    except FileExistsError as error:
        assert "Refusing to overwrite" in str(error)
    else:
        raise AssertionError("Expected overwrite refusal")
