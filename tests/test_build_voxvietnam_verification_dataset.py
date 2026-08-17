"""Tests for compact VoxVietnam three-task dataset builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_voxvietnam_verification_dataset import (
    materialize_voxvietnam_subset,
    select_speaker_splits,
)
from src.utils.files import sha256_file


def _records() -> list[dict]:
    rows = []
    for partition_index, (partition, speakers) in enumerate((
        ("train_small", [f"train-speaker-{index}" for index in range(4)]),
        ("test", [f"test-speaker-{index}" for index in range(2)]),
    )):
        for speaker_index, speaker in enumerate(speakers):
            for audio_index in range(4):
                samples = np.full(
                    320 + 7 * audio_index,
                    0.1
                    + 0.05 * partition_index
                    + 0.01 * speaker_index
                    + 0.001 * audio_index,
                    dtype=np.float32,
                )
                rows.append(
                    {
                        "speaker": speaker,
                        "audio": {"array": samples, "sampling_rate": 16_000},
                        "_source_partition": partition,
                    }
                )
    return rows


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _speaker_splits() -> dict[str, list[str]]:
    training_counts = {f"train-speaker-{index}": 4 for index in range(4)}
    test_counts = {f"test-speaker-{index}": 4 for index in range(2)}
    return select_speaker_splits(
        training_counts,
        test_counts,
        train_speakers=2,
        validation_speakers=2,
        test_speakers=2,
        train_audio_per_speaker=3,
        evaluation_audio_per_speaker=3,
        seed=42,
    )


def test_builds_speaker_disjoint_dataset_and_binary_trials(tmp_path: Path) -> None:
    output = tmp_path / "voxvietnam"
    manifest = materialize_voxvietnam_subset(
        _records(),
        _speaker_splits(),
        output_root=output,
        train_audio_per_speaker=3,
        evaluation_audio_per_speaker=3,
        enrollment_audio_per_speaker=1,
        negative_trials_per_query=1,
        closed_set_train_audio_per_speaker=1,
        closed_set_validation_audio_per_speaker=1,
        open_set_known_speakers=1,
        max_bytes=10 * 1024**2,
        seed=42,
    )

    expected_audio = {"train": 6, "validation": 6, "test": 6}
    speaker_sets = {}
    all_checksums = set()
    for split, expected in expected_audio.items():
        rows = _read(output / split / "metadata.csv")
        assert len(rows) == expected
        assert {row["split"] for row in rows} == {split}
        speaker_sets[split] = {row["normalized_speaker_id"] for row in rows}
        for row in rows:
            audio = output / row["audio_path"]
            assert audio.is_file()
            assert row["sample_rate"] == "16000"
            assert row["num_channels"] == "1"
            assert row["checksum"] == sha256_file(audio)
            all_checksums.add(row["checksum"])

    assert not speaker_sets["train"] & speaker_sets["validation"]
    assert not speaker_sets["train"] & speaker_sets["test"]
    assert not speaker_sets["validation"] & speaker_sets["test"]
    assert len(all_checksums) == sum(expected_audio.values())

    for split in ("validation", "test"):
        rows = _read(output / split / "metadata.csv")
        assert sum(row["role"] == "ENROLLMENT" for row in rows) == 2
        assert sum(row["role"] == "QUERY" for row in rows) == 4
        trials = _read(output / split / "verification_trials.csv")
        assert len(trials) == 8
        assert sum(row["label"] == "1" for row in trials) == 4
        assert sum(row["label"] == "0" for row in trials) == 4
        assert all(
            (row["enrollment_speaker_id"] == row["query_speaker_id"])
            == (row["label"] == "1")
            for row in trials
        )

    assert manifest["total_audio"] == 18
    assert manifest["total_audio_bytes"] <= manifest["maximum_audio_bytes"]
    assert manifest["invariants"]["speaker_disjoint"] is True
    assert manifest["dataset"] == "voxvietnam_ecapa_three_task_v1"
    assert json.loads((output / "manifest.json").read_text()) == manifest

    closed_train = _read(output / "protocols/closed_set/classifier_train.csv")
    closed_valid = _read(output / "protocols/closed_set/validation_queries.csv")
    closed_test = _read(output / "protocols/closed_set/test_queries.csv")
    assert [len(closed_train), len(closed_valid), len(closed_test)] == [2, 2, 2]
    assert {row["speaker_id"] for row in closed_train} == {
        row["speaker_id"] for row in closed_valid
    } == {row["speaker_id"] for row in closed_test}
    closed_paths = [
        {row["audio_path"] for row in rows}
        for rows in (closed_train, closed_valid, closed_test)
    ]
    closed_checksums = [
        {row["checksum"] for row in rows}
        for rows in (closed_train, closed_valid, closed_test)
    ]
    assert all(
        not left & right
        for index, left in enumerate(closed_paths)
        for right in closed_paths[index + 1 :]
    )
    assert all(
        not left & right
        for index, left in enumerate(closed_checksums)
        for right in closed_checksums[index + 1 :]
    )

    for split in ("validation", "test"):
        enrollment = _read(output / f"protocols/verification/{split}_enrollment.csv")
        trials = _read(output / f"protocols/verification/{split}_trials.csv")
        assert all((output / row["audio_path"]).is_file() for row in enrollment)
        assert all((output / row["query_audio_path"]).is_file() for row in trials)
        assert all(
            (row["enrollment_speaker_id"] == row["query_speaker_id"])
            == (row["label"] == "1")
            for row in trials
        )

        gallery = _read(output / f"protocols/open_set/{split}_gallery.csv")
        queries = _read(output / f"protocols/open_set/{split}_queries.csv")
        gallery_speakers = {row["speaker_id"] for row in gallery}
        assert len(gallery_speakers) == 1
        assert all((output / row["audio_path"]).is_file() for row in gallery)
        assert all((output / row["query_audio_path"]).is_file() for row in queries)
        assert all(
            (row["query_speaker_id"] in gallery_speakers)
            == (row["is_known"] == "1")
            for row in queries
        )

    validation_open = _read(output / "protocols/open_set/validation_queries.csv")
    test_open = _read(output / "protocols/open_set/test_queries.csv")
    assert {row["query_speaker_id"] for row in validation_open}.isdisjoint(
        {row["query_speaker_id"] for row in test_open}
    )


def test_protocol_generation_is_deterministic(tmp_path: Path) -> None:
    manifests = []
    protocol_bytes = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifests.append(materialize_voxvietnam_subset(
            _records(),
            _speaker_splits(),
            output_root=output,
            train_audio_per_speaker=3,
            evaluation_audio_per_speaker=3,
            enrollment_audio_per_speaker=1,
            negative_trials_per_query=1,
            closed_set_train_audio_per_speaker=1,
            closed_set_validation_audio_per_speaker=1,
            open_set_known_speakers=1,
            max_bytes=10 * 1024**2,
            seed=7,
        ))
        protocol_bytes.append({
            path.relative_to(output).as_posix(): path.read_bytes()
            for path in (output / "protocols").rglob("*.csv")
        })
    assert protocol_bytes[0] == protocol_bytes[1]
    assert manifests[0]["protocols"] == manifests[1]["protocols"]


def test_rejects_invalid_protocol_partitions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="closed-set partitions"):
        materialize_voxvietnam_subset(
            _records(),
            _speaker_splits(),
            output_root=tmp_path / "bad-closed",
            train_audio_per_speaker=3,
            evaluation_audio_per_speaker=3,
            enrollment_audio_per_speaker=1,
            closed_set_train_audio_per_speaker=2,
            closed_set_validation_audio_per_speaker=1,
        )
    assert not (tmp_path / "bad-closed").exists()

    with pytest.raises(ValueError, match="open_set_known_speakers"):
        materialize_voxvietnam_subset(
            _records(),
            _speaker_splits(),
            output_root=tmp_path / "bad-open",
            train_audio_per_speaker=3,
            evaluation_audio_per_speaker=3,
            enrollment_audio_per_speaker=1,
            closed_set_train_audio_per_speaker=1,
            closed_set_validation_audio_per_speaker=1,
            open_set_known_speakers=2,
        )
    assert not (tmp_path / "bad-open").exists()


def test_refuses_source_speaker_overlap() -> None:
    with pytest.raises(ValueError, match="source partitions overlap"):
        select_speaker_splits(
            {"same-speaker": 10},
            {"same-speaker": 10},
            train_speakers=1,
            validation_speakers=1,
            test_speakers=1,
            train_audio_per_speaker=1,
            evaluation_audio_per_speaker=1,
        )


def test_validation_speakers_only_require_evaluation_audio_count() -> None:
    splits = select_speaker_splits(
        {
            "train-a": 4,
            "train-b": 4,
            "validation-a": 2,
            "validation-b": 2,
        },
        {"test-a": 2},
        train_speakers=2,
        validation_speakers=2,
        test_speakers=1,
        train_audio_per_speaker=4,
        evaluation_audio_per_speaker=2,
        seed=42,
    )

    assert set(splits["train"]) == {"train-a", "train-b"}
    assert set(splits["validation"]) == {"validation-a", "validation-b"}
    assert splits["test"] == ["test-a"]


def test_byte_budget_failure_removes_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "too-large"
    with pytest.raises(ValueError, match="byte budget exceeded"):
        materialize_voxvietnam_subset(
            _records(),
            _speaker_splits(),
            output_root=output,
            train_audio_per_speaker=3,
            evaluation_audio_per_speaker=3,
            enrollment_audio_per_speaker=1,
            negative_trials_per_query=1,
            closed_set_train_audio_per_speaker=1,
            closed_set_validation_audio_per_speaker=1,
            open_set_known_speakers=1,
            max_bytes=1,
        )
    assert not output.exists()


def test_refuses_to_overwrite_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize_voxvietnam_subset(
            _records(),
            _speaker_splits(),
            output_root=output,
            train_audio_per_speaker=3,
            evaluation_audio_per_speaker=3,
        )
