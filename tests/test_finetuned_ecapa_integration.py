from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.speaker import application
from src.database.user_repository import create_user
from src.security.secret_phrase import hash_secret_phrase
from src.speaker.embedding import (
    CheckpointValidationError,
    ECAPAEmbeddingExtractor,
    clear_embedding_extractor_cache,
    get_embedding_extractor,
)
from src.pipeline.orchestrator import process_audio_request


class _RuntimeClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_model = torch.nn.Linear(2, 2, bias=False)
        self.mods = SimpleNamespace(embedding_model=self.embedding_model)

    def encode_batch(self, waveform, normalize=False):
        values = torch.arange(1, 193, dtype=torch.float32)
        return values.reshape(1, 1, 192)


class _SequenceExtractor:
    model_version = "ecapa-voxvietnam-epoch-9"

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = iter(vectors)

    def extract(self, audio, *, sample_rate):
        vector = np.asarray(next(self.vectors), dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector, vector.size, 1.0


def _write_checkpoint(path: Path, state_dict=None, *, key: str = "encoder") -> str:
    classifier = _RuntimeClassifier()
    torch.save(
        {
            key: state_dict or classifier.embedding_model.state_dict(),
            "epoch": 9,
            "validation": {"eer": 0.052},
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_hash_schema_and_strict_encoder_loading(tmp_path: Path) -> None:
    path = tmp_path / "fine-tuned.pt"
    expected = {
        "weight": torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    }
    digest = _write_checkpoint(path, expected)
    classifier = _RuntimeClassifier()

    extractor = ECAPAEmbeddingExtractor(
        classifier=classifier,
        expected_dimension=192,
        checkpoint_path=path,
        checkpoint_key="encoder",
        checkpoint_sha256=digest,
        checkpoint_strict=True,
        model_version="ecapa-voxvietnam-epoch-9",
    )

    assert torch.equal(classifier.embedding_model.weight, expected["weight"])
    assert extractor.checkpoint_metadata["epoch"] == 9
    assert extractor.model_version == "ecapa-voxvietnam-epoch-9"
    assert extractor.is_frozen


@pytest.mark.parametrize("failure", ["missing", "wrong_hash", "wrong_key", "incompatible"])
def test_checkpoint_validation_fails_closed(tmp_path: Path, failure: str) -> None:
    path = tmp_path / "fine-tuned.pt"
    digest = _write_checkpoint(path)
    key = "encoder"
    if failure == "missing":
        path.unlink()
    elif failure == "wrong_hash":
        digest = "0" * 64
    elif failure == "wrong_key":
        key = "missing_encoder"
    elif failure == "incompatible":
        digest = _write_checkpoint(path, {"wrong.weight": torch.ones(1)})

    with pytest.raises(CheckpointValidationError):
        ECAPAEmbeddingExtractor(
            classifier=_RuntimeClassifier(),
            checkpoint_path=path,
            checkpoint_key=key,
            checkpoint_sha256=digest,
            checkpoint_strict=True,
            model_version="ecapa-voxvietnam-epoch-9",
        )


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fine-tuned.pt"
    path.write_bytes(b"not a torch checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(CheckpointValidationError):
        ECAPAEmbeddingExtractor(
            classifier=_RuntimeClassifier(),
            checkpoint_path=path,
            checkpoint_sha256=digest,
            model_version="ecapa-voxvietnam-epoch-9",
        )


def test_config_resolves_checkpoint_and_cached_extractor_loads_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    digest = _write_checkpoint(checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "speaker": {
                    "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
                    "device": "cpu",
                    "cache_dir": "cache",
                    "embedding_dimension": 192,
                    "evaluation_mode": True,
                    "freeze_parameters": True,
                    "fine_tune": False,
                    "model_version": "ecapa-voxvietnam-epoch-9",
                    "checkpoint_path": "model.pt",
                    "checkpoint_encoder_key": "encoder",
                    "checkpoint_sha256": digest,
                    "checkpoint_strict": True,
                }
            }
        ),
        encoding="utf-8",
    )
    loads = 0

    def fake_load(self):
        nonlocal loads
        loads += 1
        return _RuntimeClassifier()

    monkeypatch.setattr(ECAPAEmbeddingExtractor, "_load_classifier", fake_load)
    clear_embedding_extractor_cache()
    first = get_embedding_extractor(config)
    second = get_embedding_extractor(config)

    assert first is second
    assert loads == 1
    assert first.checkpoint_path == checkpoint.resolve()
    clear_embedding_extractor_cache()


def test_application_enrollment_verification_and_stale_centroid_gate(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"database": "users.db"},
                "speaker": {
                    "application_centroid_dir": "centroids",
                    "model_version": "ecapa-voxvietnam-epoch-9",
                    "enrollment_audio_count": 5,
                    "application_verification_threshold": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    audio_paths = []
    for index in range(5):
        path = tmp_path / f"enroll-{index}.wav"
        path.write_bytes(f"audio-{index}".encode("ascii"))
        audio_paths.append(path)
    loader = lambda path: (np.ones(16, dtype=np.float32), 16000)
    extractor = _SequenceExtractor([np.array([1.0, 0.0])] * 6)
    database = tmp_path / "users.db"

    enrolled = application.enroll_user(
        "user_001",
        "Student",
        audio_paths,
        secret_phrase="hoa sen xanh",
        database_path=database,
        config_path=config,
        extractor=extractor,
        audio_loader=loader,
    )
    verified = application.verify_speaker(
        audio_paths[0],
        "user_001",
        database_path=database,
        config_path=config,
        extractor=extractor,
        audio_loader=loader,
    )

    centroid = Path(enrolled["centroid_path"])
    metadata = centroid.with_suffix(".meta.json")
    assert enrolled["success"] is True
    assert json.loads(metadata.read_text())["model_version"] == "ecapa-voxvietnam-epoch-9"
    assert verified["success"] is True
    assert verified["verified"] is True

    document = json.loads(metadata.read_text())
    document["model_version"] = "baseline"
    metadata.write_text(json.dumps(document), encoding="utf-8")
    stale = application.verify_speaker(
        audio_paths[0],
        "user_001",
        database_path=database,
        config_path=config,
        extractor=_SequenceExtractor([np.array([1.0, 0.0])]),
        audio_loader=loader,
    )
    assert stale["success"] is False
    assert stale["error"] == "CENTROID_MODEL_MISMATCH"

    duplicate = application.enroll_user(
        "user_002",
        "Duplicate",
        [audio_paths[0]] * 5,
        secret_phrase="hoa sen xanh",
        database_path=database,
        config_path=config,
        extractor=_SequenceExtractor([np.array([1.0, 0.0])] * 5),
        audio_loader=loader,
    )
    assert duplicate["success"] is False
    assert duplicate["error"] == "DUPLICATE_ENROLLMENT_AUDIO"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.499, False), (0.5, True), (0.501, True)],
)
def test_verification_threshold_boundary(tmp_path: Path, score: float, expected: bool) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "speaker": {
                    "application_centroid_dir": "centroids",
                    "model_version": "test-model",
                    "application_verification_threshold": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "query.wav"
    audio.write_bytes(b"audio")
    centroid_dir = tmp_path / "centroids"
    centroid_dir.mkdir()
    centroid = centroid_dir / "user_001.npy"
    np.save(centroid, [1.0, 0.0])
    centroid.with_suffix(".meta.json").write_text(
        json.dumps({"model_version": "test-model", "embedding_dimension": 2}),
        encoding="utf-8",
    )
    from src.database.user_repository import create_user

    database = tmp_path / "users.db"
    create_user("user_001", "Student", str(centroid), database)
    query = np.array([score, np.sqrt(1.0 - score**2)])
    result = application.verify_speaker(
        audio,
        "user_001",
        database_path=database,
        config_path=config,
        extractor=_SequenceExtractor([query]),
        audio_loader=lambda path: (np.ones(16, dtype=np.float32), 16000),
    )
    assert result["success"] is True
    assert result["verified"] is expected
    assert result["identified"] is None


def test_orchestrator_blocks_private_note_when_verification_fails(tmp_path: Path) -> None:
    database = tmp_path / "users.db"
    secret_hash, secret_salt = hash_secret_phrase("hoa sen xanh")
    create_user(
        "user_001",
        "Student",
        database_path=database,
        secret_phrase_hash=secret_hash,
        secret_phrase_salt=secret_salt,
    )
    pipeline_result = {
        "success": True,
        "transcript": "đọc ghi chú riêng tư mật khẩu hoa sen xanh",
        "normalized_transcript": "đọc ghi chú riêng tư mật khẩu hoa sen xanh",
        "intent": "VIEW_PRIVATE_NOTE",
        "entities": {},
        "missing_fields": [],
        "error": None,
    }

    result = process_audio_request(
        tmp_path / "query.wav",
        database_path=database,
        asr_nlu_runner=lambda *args, **kwargs: pipeline_result,
        identifier=lambda *args, **kwargs: {
            "success": True,
            "candidate_user_id": "user_001",
            "similarity": 0.8,
            "identified": True,
            "centroid_path": "centroid.npy",
            "error": None,
        },
        verifier=lambda *args, **kwargs: {
            "success": True,
            "verified": False,
            "similarity": 0.4,
            "error": None,
        },
    )

    assert result["success"] is False
    assert result["speaker"]["secret_phrase_verified"] is True
    assert result["speaker"]["verified"] is False
    assert result["error"] == "VERIFICATION_FAILED"
    assert "thất bại" in result["response"]


def test_orchestrator_skips_identification_for_public_intent(tmp_path: Path) -> None:
    calls: list[str] = []

    def identify(*args, **kwargs):
        calls.append("sid")
        return {
            "success": True,
            "candidate_user_id": "user_001",
            "similarity": 0.81,
            "identified": True,
            "centroid_path": "centroid.npy",
            "error": None,
        }

    def verify(*args, **kwargs):
        calls.append("sv")
        return {
            "protocol": "APPLICATION_SV",
            "success": True,
            "user_id": None,
            "candidate_user_id": "user_001",
            "embedding_count": None,
            "centroid_path": "centroid.npy",
            "similarity": 0.72,
            "identified": False,
            "verified": True,
            "error": None,
        }

    def run_asr(*args, **kwargs):
        calls.append("asr")
        return {
            "success": True,
            "transcript": "bây giờ là mấy giờ",
            "normalized_transcript": "bây giờ là mấy giờ",
            "intent": "GET_TIME",
            "entities": {},
            "missing_fields": [],
            "error": None,
        }

    result = process_audio_request(
        tmp_path / "query.wav",
        database_path=tmp_path / "users.db",
        asr_nlu_runner=run_asr,
        identifier=identify,
        verifier=verify,
    )

    assert calls == ["asr"]
    assert result["speaker"]["candidate_user_id"] is None
    assert result["speaker"]["identified"] is None
    assert result["speaker"]["verified"] is None


def test_orchestrator_skips_identification_when_asr_fails(tmp_path: Path) -> None:
    result = process_audio_request(
        tmp_path / "query.wav",
        database_path=tmp_path / "users.db",
        asr_nlu_runner=lambda *args, **kwargs: {
            "success": False,
            "transcript": "",
            "intent": "OUT_OF_SCOPE",
            "error": "ASR_FAILED",
        },
        identifier=lambda *args, **kwargs: {
            "success": True,
            "candidate_user_id": "user_001",
            "similarity": 0.81,
            "identified": True,
            "centroid_path": "centroid.npy",
            "error": None,
        },
        verifier=lambda *args, **kwargs: {
            "success": True,
            "similarity": 0.72,
            "verified": True,
            "error": None,
        },
    )

    assert result["success"] is False
    assert result["speaker"]["candidate_user_id"] is None
    assert result["speaker"]["identified"] is None
    assert "verification" not in result["speaker"]
    assert result["speaker"]["verified"] is None
