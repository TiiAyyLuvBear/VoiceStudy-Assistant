from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.utils import canonical_csv_sha256


ROOT = Path("data/processed/v2")
METADATA = ROOT / "metadata"
MANIFEST = ROOT / "split_manifest.json"
SVM_TARGETS = {
    "svm_closed_set_enrollment.csv": 45,
    "svm_closed_set_train.csv": 600,
    "svm_closed_set_validation.csv": 129,
    "svm_closed_set_test.csv": 128,
}
COSINE_FILES = (
    "cosine_test_enrollment.csv",
    "cosine_test_query.csv",
    "cosine_test_unknown.csv",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _audio_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/").casefold()
    marker = "/data/audio/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    elif normalized.startswith("data/audio/"):
        normalized = normalized[len("data/audio/") :]
    return normalized.lstrip("./")


def test_speaker_v2_balanced_counts_and_disjoint_speakers() -> None:
    svm_speakers: set[str] = set()
    for name, expected in SVM_TARGETS.items():
        rows = _rows(METADATA / name)
        counts = Counter(row["normalized_speaker_id"] for row in rows)
        assert len(counts) == 9
        assert sum(counts.values()) == expected
        svm_speakers.update(row["speaker_id"] for row in rows)
    assert set(
        Counter(
            row["normalized_speaker_id"]
            for row in _rows(METADATA / "svm_closed_set_enrollment.csv")
        ).values()
    ) == {5}
    train_counts = Counter(
        row["normalized_speaker_id"]
        for row in _rows(METADATA / "svm_closed_set_train.csv")
    )
    validation_counts = Counter(
        row["normalized_speaker_id"]
        for row in _rows(METADATA / "svm_closed_set_validation.csv")
    )
    test_counts = Counter(
        row["normalized_speaker_id"]
        for row in _rows(METADATA / "svm_closed_set_test.csv")
    )
    assert (min(train_counts.values()), max(train_counts.values())) == (64, 72)
    assert (min(validation_counts.values()), max(validation_counts.values())) == (
        13,
        16,
    )
    assert (min(test_counts.values()), max(test_counts.values())) == (14, 15)

    enrollment = _rows(METADATA / "cosine_test_enrollment.csv")
    query = _rows(METADATA / "cosine_test_query.csv")
    unknown = _rows(METADATA / "cosine_test_unknown.csv")
    enrollment_counts = Counter(row["normalized_speaker_id"] for row in enrollment)
    query_counts = Counter(row["normalized_speaker_id"] for row in query)
    unknown_counts = Counter(row["normalized_speaker_id"] for row in unknown)

    assert len(enrollment_counts) == 8
    assert set(enrollment_counts.values()) == {5}
    assert len(query_counts) == 8
    assert set(query_counts.values()) == {25}
    assert set(enrollment_counts) == set(query_counts)
    assert len(unknown_counts) == 8
    assert len(unknown) == 109
    assert set(enrollment_counts).isdisjoint(unknown_counts)
    cosine_speakers = {row["speaker_id"] for row in enrollment + query + unknown}
    assert svm_speakers.isdisjoint(cosine_speakers)


def test_speaker_v2_has_no_audio_overlap_with_asr_v2_or_itself() -> None:
    asr_paths = {
        _audio_key(row["audio_path"])
        for name in ("asr_validation.csv", "asr_test.csv")
        for row in _rows(METADATA / name)
    }
    speaker_paths = [
        _audio_key(row["audio_path"])
        for name in (*SVM_TARGETS, *COSINE_FILES)
        for row in _rows(METADATA / name)
    ]

    assert len(speaker_paths) == len(set(speaker_paths))
    assert set(speaker_paths).isdisjoint(asr_paths)


def test_speaker_v2_preserves_v1_svm_split_roles() -> None:
    selected = {
        row["speaker_id"]
        for row in _rows(METADATA / "selected_svm_experimental_speakers.csv")
    }
    for name in SVM_TARGETS:
        v1_paths = {
            _audio_key(row["audio_path"])
            for row in _rows(Path("data/processed/v1/metadata") / name)
            if row["speaker_id"] in selected
        }
        v2_paths = {
            _audio_key(row["audio_path"])
            for row in _rows(METADATA / name)
        }
        assert v1_paths <= v2_paths


def test_speaker_v2_manifest_checksums_match() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    component = manifest["components"]["speaker"]

    assert manifest["dataset_version"] == "v2"
    assert manifest["freeze_status"] == "FROZEN"
    assert component["invariants"]["audio_overlap_with_asr_v2"] == 0
    for split in component["splits"].values():
        assert canonical_csv_sha256(split["path"]) == split["checksum"]
    for selection in component["selection_files"].values():
        assert canonical_csv_sha256(selection["path"]) == selection["checksum"]
