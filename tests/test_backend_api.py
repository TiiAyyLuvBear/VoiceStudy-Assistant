from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.main import create_app


def test_health_endpoint_reports_backend_and_model_version() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "voicestudy-backend",
        "model_version": "ecapa-voxvietnam-epoch-9",
    }


def test_process_endpoint_passes_temporary_wav_and_removes_it() -> None:
    observed: dict = {}

    def processor(path, *, reference_date=None, config_path="config.yaml"):
        audio_path = Path(path)
        observed["path"] = audio_path
        observed["exists_during_call"] = audio_path.is_file()
        observed["bytes"] = audio_path.read_bytes()
        observed["reference_date"] = reference_date
        return {
            "success": True,
            "intent": "GET_TIME",
            "speaker": {},
            "response": "ok",
            "error": None,
        }

    client = TestClient(create_app(pipeline_processor=processor))
    response = client.post(
        "/api/v1/process",
        data={"reference_date": "2026-08-17"},
        files={"audio": ("command.wav", b"RIFF-test", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "GET_TIME"
    assert observed["exists_during_call"] is True
    assert observed["bytes"] == b"RIFF-test"
    assert observed["reference_date"] == "2026-08-17"
    assert observed["path"].exists() is False


def test_enrollment_requires_five_files_and_calls_service() -> None:
    observed: dict = {}

    def enroll(user_id, name, paths, *, config_path="config.yaml"):
        observed["user_id"] = user_id
        observed["name"] = name
        observed["count"] = len(paths)
        observed["all_exist"] = all(Path(path).is_file() for path in paths)
        return {"success": True, "user_id": user_id, "centroid_path": "centroid.npy"}

    client = TestClient(create_app(enroller=enroll))
    four = [("audio_files", (f"{index}.wav", b"wav", "audio/wav")) for index in range(4)]
    rejected = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student"},
        files=four,
    )
    five = [("audio_files", (f"{index}.wav", b"wav", "audio/wav")) for index in range(5)]
    accepted = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student"},
        files=five,
    )

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "Exactly 5 WAV files are required"
    assert accepted.status_code == 200
    assert accepted.json()["success"] is True
    assert observed == {
        "user_id": "user_001",
        "name": "Student",
        "count": 5,
        "all_exist": True,
    }


def test_user_endpoints_use_repository_contract() -> None:
    client = TestClient(
        create_app(
            user_lister=lambda: [{"user_id": "user_001", "name": "Student"}],
            user_deleter=lambda user_id: user_id == "user_001",
        )
    )

    listed = client.get("/api/v1/users")
    deleted = client.delete("/api/v1/users/user_001")
    missing = client.delete("/api/v1/users/user_999")

    assert listed.status_code == 200
    assert listed.json()[0]["user_id"] == "user_001"
    assert deleted.status_code == 200
    assert deleted.json() == {"success": True, "user_id": "user_001"}
    assert missing.status_code == 404


def test_upload_limit_rejects_large_audio() -> None:
    client = TestClient(create_app(max_upload_bytes=4))

    response = client.post(
        "/api/v1/process",
        files={"audio": ("large.wav", b"12345", "audio/wav")},
    )

    assert response.status_code == 413


def test_backend_startup_preloads_models_and_prints_each_field(capsys) -> None:
    speaker = SimpleNamespace(
        model_version="ecapa-voxvietnam-epoch-9",
        device="cpu",
        checkpoint_metadata={"epoch": 9},
    )
    asr = SimpleNamespace(
        config=SimpleNamespace(model_name="whisper-small", device="cpu", compute_type="int8"),
        load_model=lambda: object(),
    )
    application = create_app(
        preload_models=True,
        speaker_loader=lambda config_path: speaker,
        asr_loader=lambda config_path: asr,
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200

    output = capsys.readouterr().out
    assert "event: backend_startup" in output
    assert "speaker_loaded: true" in output
    assert "speaker_model: ecapa-voxvietnam-epoch-9" in output
    assert "speaker_epoch: 9" in output
    assert "asr_loaded: true" in output
    assert "asr_model: whisper-small" in output
    assert "device: cpu" in output
