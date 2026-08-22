from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.database.user_repository import get_user
from src.speaker.application import (
    delete_application_user,
    enroll_user,
    identify_application_user,
    verify_speaker,
)


class _Extractor:
    def extract(self, audio, *, sample_rate):
        vector = np.asarray(audio, dtype=np.float32).reshape(-1)[:2]
        vector = vector / np.linalg.norm(vector)
        return vector, vector.size, 0.1


class _SequenceExtractor:
    def __init__(self, vectors):
        self.vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        self.calls = 0

    def extract(self, audio, *, sample_rate):
        vector = self.vectors[self.calls]
        self.calls += 1
        return vector, vector.size, 0.1


def _audio_loader(path: str | Path) -> tuple[np.ndarray, int]:
    name = Path(path).stem
    return (
        np.asarray([0.8, 0.6], dtype=np.float32)
        if "genuine" in name or "enroll" in name
        else np.asarray([0.0, 1.0], dtype=np.float32),
        16000,
    )


def _config(tmp_path: Path) -> Path:
    centroid_dir = tmp_path / "centroids"
    sid = tmp_path / "sid.json"
    sv = tmp_path / "sv.json"
    sid.write_text(json.dumps({"threshold": 0.68}), encoding="utf-8")
    sv.write_text(json.dumps({"threshold": 0.72}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "speaker:",
                f"  application_centroid_dir: {centroid_dir.as_posix()}",
                f"  application_sid_threshold_path: {sid.as_posix()}",
                f"  application_verification_threshold_path: {sv.as_posix()}",
            )
        ),
        encoding="utf-8",
    )
    return config


def _quality_config(tmp_path: Path) -> Path:
    centroid_dir = tmp_path / "centroids"
    sid = tmp_path / "sid.json"
    sv = tmp_path / "sv.json"
    sid.write_text(json.dumps({"threshold": 0.68}), encoding="utf-8")
    sv.write_text(json.dumps({"threshold": 0.72}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            (
                "speaker:",
                f"  application_centroid_dir: {centroid_dir.as_posix()}",
                f"  application_sid_threshold_path: {sid.as_posix()}",
                f"  application_verification_threshold_path: {sv.as_posix()}",
                "  enrollment_quality:",
                "    enabled: true",
                "    min_duration_seconds: 2.0",
                "    min_speech_duration_seconds: 2.0",
                "    min_speech_ratio: 0.25",
                "    min_rms: 0.01",
                "    max_clipping_ratio: 0.02",
                "    max_silence_ratio: 0.65",
                "    min_mean_pairwise_cosine: 0.70",
                "    min_pairwise_cosine: 0.45",
                "    min_samples_for_outlier_detection: 3",
                "    min_centroid_similarity: 0.45",
                "    min_accepted_enrollment_samples: 3",
            )
        ),
        encoding="utf-8",
    )
    return config


def _five_audio(tmp_path: Path) -> list[Path]:
    paths = []
    for index in range(5):
        path = tmp_path / f"enroll-{index}.wav"
        path.write_bytes(f"audio-{index}".encode("ascii"))
        paths.append(path)
    return paths


def _quality_audio_loader(path: str | Path) -> tuple[np.ndarray, int, dict]:
    audio = np.full(32000, 0.5, dtype=np.float32)
    return audio, 16000, {
        "duration_seconds": 2.0,
        "speech_duration_seconds": 2.0,
        "speech_ratio": 1.0,
        "rms": 0.5,
        "clipping_ratio": 0.0,
        "sample_rate": 16000,
        "num_channels": 1,
    }


def _short_speech_audio_loader(path: str | Path) -> tuple[np.ndarray, int, dict]:
    audio = np.full(32000, 0.5, dtype=np.float32)
    return audio, 16000, {
        "duration_seconds": 2.0,
        "speech_duration_seconds": 0.8,
        "speech_ratio": 0.4,
        "rms": 0.5,
        "clipping_ratio": 0.0,
        "sample_rate": 16000,
        "num_channels": 1,
    }


def test_application_api_contract_and_database_enrollment(tmp_path: Path) -> None:
    database = tmp_path / "application.db"
    config = _config(tmp_path)
    result = enroll_user(
        "user_003",
        "User 003",
        _five_audio(tmp_path),
        secret_phrase="hoa sen xanh an toàn",
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )

    assert result["success"] is True
    assert result["status"] == "ENROLLED"
    assert result["secret_phrase_configured"] is True
    assert result["audio_count"] == 5
    assert result["embedding_dim"] == 2
    assert np.isclose(result["l2_norm"], 1.0)
    assert result["latency_ms"] >= 0.0
    assert all(item["valid"] for item in result["file_results"])
    assert get_user("user_003", database)["embedding_path"] == result["embedding_path"]
    assert Path(result["centroid_path"]).is_file()

    query = tmp_path / "genuine-query.wav"
    query.write_bytes(b"audio")
    sid = identify_application_user(
        query,
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    assert sid["status"] == "KNOWN"
    assert sid["candidate_user_id"] == "user_003"
    assert sid["cosine_similarity"] == sid["similarity"]
    assert sid["unknown_threshold"] == 0.68
    assert sid["latency_ms"] >= 0.0

    sv = verify_speaker(
        query,
        "user_003",
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    assert sv["status"] == "VERIFIED"
    assert sv["verified"] is True
    assert sv["verification_threshold"] == 0.72
    assert sv["latency_ms"] >= 0.0


def test_application_api_unknown_impostor_and_delete(tmp_path: Path) -> None:
    database = tmp_path / "application.db"
    config = _config(tmp_path)
    enrolled = enroll_user(
        "user_003",
        "User 003",
        _five_audio(tmp_path),
        secret_phrase="hoa sen xanh an toàn",
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    query = tmp_path / "impostor-query.wav"
    query.write_bytes(b"audio")

    sid = identify_application_user(
        query,
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    assert sid["success"] is True
    assert sid["status"] == "UNKNOWN"
    assert sid["candidate_user_id"] is None

    sv = verify_speaker(
        query,
        "user_003",
        database_path=database,
        config_path=config,
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    assert sv["success"] is True
    assert sv["status"] == "REJECTED"
    assert sv["verified"] is False

    deleted = delete_application_user(
        "user_003", database_path=database, config_path=config
    )
    assert deleted["success"] is True
    assert get_user("user_003", database) is None
    assert not (config.parent / enrolled["centroid_path"]).exists()


def test_application_api_rejects_duplicate_enrollment_audio(tmp_path: Path) -> None:
    paths = _five_audio(tmp_path)
    result = enroll_user(
        "user_003",
        "User 003",
        paths[:4] + [paths[0]],
        secret_phrase="hoa sen xanh an toàn",
        database_path=tmp_path / "application.db",
        config_path=_config(tmp_path),
        extractor=_Extractor(),
        audio_loader=_audio_loader,
    )
    assert result["success"] is False
    assert result["error"] == "DUPLICATE_ENROLLMENT_AUDIO"


def test_enrollment_rejects_outlier_sample_but_saves_centroid_from_accepted_samples(tmp_path: Path) -> None:
    extractor = _SequenceExtractor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )

    result = enroll_user(
        "user_003",
        "User 003",
        _five_audio(tmp_path),
        secret_phrase="hoa sen xanh an toàn",
        database_path=tmp_path / "application.db",
        config_path=_quality_config(tmp_path),
        extractor=extractor,
        audio_loader=_quality_audio_loader,
    )

    assert result["success"] is True
    assert result["accepted_samples"] == 4
    assert result["rejected_samples"] == 1
    assert result["file_results"][4]["valid"] is False
    assert result["file_results"][4]["accepted"] is False
    assert result["file_results"][4]["error"] == "VOICE_INCONSISTENT_WITH_OTHER_SAMPLES"
    assert result["file_results"][4]["rejection_reasons"] == ["embedding_outlier"]
    assert result["embedding_consistency"]["accepted"]["valid"] is True
    assert np.isclose(result["l2_norm"], 1.0)


def test_enrollment_accepts_prompt_variation_around_shared_centroid(tmp_path: Path) -> None:
    extractor = _SequenceExtractor(
        [
            [0.64278764, -0.76604444],
            [0.8660254, -0.5],
            [1.0, 0.0],
            [0.8660254, 0.5],
            [0.64278764, 0.76604444],
        ]
    )

    result = enroll_user(
        "user_003",
        "User 003",
        _five_audio(tmp_path),
        secret_phrase="hoa sen xanh an toàn",
        database_path=tmp_path / "application.db",
        config_path=_quality_config(tmp_path),
        extractor=extractor,
        audio_loader=_quality_audio_loader,
    )

    assert result["success"] is True
    assert result["accepted_samples"] == 5
    assert result["embedding_consistency"]["accepted"]["valid"] is True
    assert result["embedding_consistency"]["accepted"]["valid_by_pairwise"] is False
    assert result["embedding_consistency"]["accepted"]["valid_by_centroid"] is True


def test_enrollment_requires_minimum_accepted_speech_samples(tmp_path: Path) -> None:
    result = enroll_user(
        "user_003",
        "User 003",
        _five_audio(tmp_path),
        secret_phrase="hoa sen xanh an toàn",
        database_path=tmp_path / "application.db",
        config_path=_quality_config(tmp_path),
        extractor=_SequenceExtractor([[1.0, 0.0]] * 5),
        audio_loader=_short_speech_audio_loader,
    )

    assert result["success"] is False
    assert result["error"] == "INSUFFICIENT_ACCEPTED_SAMPLES"
    assert result["requires_more_samples"] is True
    assert result["accepted_samples"] == 0
    assert result["file_results"][0]["error"] == "AUDIO_QUALITY_FAILED"
    assert "insufficient_speech" in result["file_results"][0]["rejection_reasons"]
