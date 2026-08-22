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

    def processor(
        path,
        *,
        secret_audio_path=None,
        reference_date=None,
        config_path="config.yaml",
    ):
        audio_path = Path(path)
        observed["path"] = audio_path
        observed["exists_during_call"] = audio_path.is_file()
        observed["bytes"] = audio_path.read_bytes()
        observed["secret_audio_path"] = secret_audio_path
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
    assert observed["secret_audio_path"] is None
    assert observed["reference_date"] == "2026-08-17"
    assert observed["path"].exists() is False


def test_process_endpoint_accepts_secret_audio_for_private_verification() -> None:
    observed: dict = {}

    def processor(
        path,
        *,
        secret_audio_path=None,
        reference_date=None,
        config_path="config.yaml",
    ):
        observed["command_exists"] = Path(path).is_file()
        observed["secret_exists"] = Path(secret_audio_path).is_file()
        observed["secret_bytes"] = Path(secret_audio_path).read_bytes()
        return {
            "success": True,
            "intent": "VIEW_PRIVATE_NOTE",
            "speaker": {"secret_phrase_verified": True},
            "response": "ok",
            "error": None,
        }

    client = TestClient(create_app(pipeline_processor=processor))
    response = client.post(
        "/api/v1/process",
        files={
            "audio": ("command.wav", b"RIFF-command", "audio/wav"),
            "secret_audio": ("secret.wav", b"RIFF-secret", "audio/wav"),
        },
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "VIEW_PRIVATE_NOTE"
    assert observed == {
        "command_exists": True,
        "secret_exists": True,
        "secret_bytes": b"RIFF-secret",
    }


def test_command_catalog_endpoint_returns_fixed_scripts() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/commands")

    assert response.status_code == 200
    commands = response.json()
    assert commands
    phrases = {command["phrase"] for command in commands}
    intents = {command["intent"] for command in commands}
    assert any(command["requires_secret"] for command in commands)
    assert "ADD_PRIVATE_NOTE" in intents
    assert "Thêm ghi chú riêng tư <nội dung>." in phrases
    assert all("thống kê" not in phrase.lower() for phrase in phrases)


def test_tts_endpoint_returns_backend_vietnamese_audio() -> None:
    client = TestClient(create_app(tts_synthesizer=lambda text: b"mp3-bytes" if text else None))

    response = client.post("/api/v1/tts", json={"text": "Xin chào"})

    assert response.status_code == 200
    assert response.content == b"mp3-bytes"
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["cache-control"] == "no-store"


def test_tts_endpoint_reports_unavailable_backend_model() -> None:
    client = TestClient(create_app(tts_synthesizer=lambda text: None))

    response = client.post("/api/v1/tts", json={"text": "Xin chào"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Vietnamese TTS is unavailable"


def test_enrollment_accepts_configured_file_range_and_calls_service() -> None:
    observed: dict = {}

    def enroll(
        user_id,
        name,
        paths,
        *,
        secret_phrase,
        enrollment_prompts,
        config_path="config.yaml",
    ):
        observed["user_id"] = user_id
        observed["name"] = name
        observed["count"] = len(paths)
        observed["all_exist"] = all(Path(path).is_file() for path in paths)
        observed["secret_phrase"] = secret_phrase
        observed["prompt_count"] = len(enrollment_prompts)
        return {"success": True, "user_id": user_id, "centroid_path": "centroid.npy"}

    secret_transcriber = lambda *args, **kwargs: {
        "success": True,
        "transcript": "hoa sen xanh",
        "error": None,
    }
    client = TestClient(create_app(enroller=enroll, secret_transcriber=secret_transcriber))
    two = [("audio_files", (f"{index}.wav", b"wav", "audio/wav")) for index in range(2)]
    rejected = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student", "secret_phrase": "hoa sen xanh"},
        files=two + [("secret_audio", ("secret.wav", b"secret", "audio/wav"))],
    )
    five = [("audio_files", (f"{index}.wav", b"wav", "audio/wav")) for index in range(5)]
    accepted = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student", "secret_phrase": "hoa sen xanh"},
        files=five + [("secret_audio", ("secret.wav", b"secret", "audio/wav"))],
    )

    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "Enrollment requires 3-10 audio files"
    assert accepted.status_code == 200
    assert accepted.json()["success"] is True
    assert observed == {
        "user_id": "user_001",
        "name": "Student",
        "count": 5,
        "all_exist": True,
        "secret_phrase": "hoa sen xanh",
        "prompt_count": 5,
    }


def test_enrollment_accepts_flac_uploads_and_preserves_suffix() -> None:
    observed: dict = {}

    def enroll(
        user_id,
        name,
        paths,
        *,
        secret_phrase,
        enrollment_prompts,
        config_path="config.yaml",
    ):
        observed["suffixes"] = [Path(path).suffix for path in paths]
        return {"success": True, "user_id": user_id, "centroid_path": "centroid.npy"}

    client = TestClient(
        create_app(
            enroller=enroll,
            secret_transcriber=lambda *args, **kwargs: {
                "success": True,
                "transcript": "hoa sen xanh",
                "error": None,
            },
        )
    )
    files = [
        ("audio_files", (f"{index}.flac", b"fLaC", "audio/flac"))
        for index in range(5)
    ]

    response = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student", "secret_phrase": "hoa sen xanh"},
        files=files + [("secret_audio", ("secret.flac", b"secret", "audio/flac"))],
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert observed["suffixes"] == [".flac"] * 5


def test_enrollment_rejects_secret_audio_transcript_mismatch() -> None:
    client = TestClient(
        create_app(
            enroller=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("enroller must not run on secret mismatch")
            ),
            secret_transcriber=lambda *args, **kwargs: {
                "success": True,
                "transcript": "mat trang bac",
                "error": None,
            },
        )
    )
    files = [
        ("audio_files", (f"{index}.wav", b"wav", "audio/wav"))
        for index in range(5)
    ]

    response = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student", "secret_phrase": "hoa sen xanh"},
        files=files + [("secret_audio", ("secret.wav", b"secret", "audio/wav"))],
    )

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["error"] == "SECRET_PHRASE_TRANSCRIPT_MISMATCH"
    assert response.json()["secret_phrase_transcript"] == "mat trang bac"
    assert response.json()["message_vi"] == (
        "Audio câu bí mật không khớp transcript đã nhập. Hãy sửa transcript hoặc đọc lại câu bí mật."
    )
    assert response.json()["failed_stage"] == "secret_audio"


def test_enrollment_reports_failed_voice_prompt_for_retry(capsys) -> None:
    def enroll(
        user_id,
        name,
        paths,
        *,
        secret_phrase,
        enrollment_prompts,
        config_path="config.yaml",
    ):
        return {
            "success": False,
            "user_id": user_id,
            "error": "AUDIO_QUALITY_FAILED",
            "file_results": [
                {"audio_path": str(paths[0]), "valid": True, "error": None},
                {"audio_path": str(paths[1]), "valid": True, "error": None},
                {
                    "audio_path": str(paths[2]),
                    "valid": False,
                    "error": "AUDIO_QUALITY_FAILED",
                    "message_vi": (
                        "Voice có quá nhiều khoảng lặng. Hãy bấm thu rồi đọc ngay, "
                        "dừng khi đọc xong."
                    ),
                    "quality": {
                        "issues": ["too_much_silence"],
                        "message_vi": (
                            "Voice có quá nhiều khoảng lặng. Hãy bấm thu rồi đọc ngay, "
                            "dừng khi đọc xong."
                        ),
                    },
                },
            ],
        }

    client = TestClient(
        create_app(
            enroller=enroll,
            secret_transcriber=lambda *args, **kwargs: {
                "success": True,
                "transcript": "hoa sen xanh",
                "error": None,
            },
        )
    )
    files = [
        ("audio_files", (f"{index}.wav", b"wav", "audio/wav"))
        for index in range(5)
    ]

    response = client.post(
        "/api/v1/enroll",
        data={"user_id": "user_001", "name": "Student", "secret_phrase": "hoa sen xanh"},
        files=files + [("secret_audio", ("secret.wav", b"secret", "audio/wav"))],
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert payload["message_vi"] == (
        "Voice có quá nhiều khoảng lặng. Hãy bấm thu rồi đọc ngay, dừng khi đọc xong."
    )
    assert payload["failed_stage"] == "speaker_sample"
    assert payload["failed_sample_index"] == 3
    assert payload["failed_prompt"] == "Thêm lịch học thống kê lúc tám giờ sáng."
    assert payload["sample_prompt"] == "Thêm lịch học thống kê lúc tám giờ sáng."

    cached = client.get("/api/v1/enrollment-cache/user_001")
    assert cached.status_code == 200
    assert cached.json()["status"] == "failed"
    assert cached.json()["prompts_source"] == "default"
    assert cached.json()["client_prompt_count"] == 0
    assert cached.json()["audio_files"][2]["sample_prompt"] == (
        "Thêm lịch học thống kê lúc tám giờ sáng."
    )
    assert cached.json()["last_error"] == "AUDIO_QUALITY_FAILED"

    output = capsys.readouterr().out
    assert "event: enrollment_received" in output
    assert "prompts_source: default" in output
    assert "client_prompt_count: 0" in output
    assert "audio_files.3.sample_prompt: Thêm lịch học thống kê lúc tám giờ sáng." in output
    assert "event: enrollment_failed" in output
    assert "message_vi: Voice có quá nhiều khoảng lặng." in output
    assert "failed_sample_index: 3" in output
    assert "sample_prompt: Thêm lịch học thống kê lúc tám giờ sáng." in output
    assert "file_results:" not in output
    assert "file_results.3.error: AUDIO_QUALITY_FAILED" in output
    assert "file_results.3.quality.issues.1: too_much_silence" in output


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


def test_user_schedule_and_note_endpoints_are_owner_scoped(monkeypatch) -> None:
    state = {
        "schedules": [
            {
                "schedule_id": 1,
                "user_id": "user_001",
                "title": "Học thống kê",
                "date": "2026-08-20",
                "time": "08:00",
            }
        ],
        "notes": [
            {
                "note_id": 1,
                "user_id": "user_001",
                "content": "Ghi chú riêng",
                "is_private": 1,
            }
        ],
    }

    import backend.main as backend_main

    monkeypatch.setattr(
        backend_main,
        "get_user",
        lambda user_id: {"user_id": user_id, "name": "Student"}
        if user_id == "user_001"
        else None,
    )
    monkeypatch.setattr(
        backend_main,
        "get_schedules",
        lambda user_id, date=None: [
            row
            for row in state["schedules"]
            if row["user_id"] == user_id and (date is None or row["date"] == date)
        ],
    )
    monkeypatch.setattr(
        backend_main,
        "add_schedule",
        lambda user_id, title, date, time, description=None: {
            "schedule_id": 2,
            "user_id": user_id,
            "title": title,
            "date": date,
            "time": time,
            "description": description,
        },
    )
    monkeypatch.setattr(
        backend_main,
        "delete_schedule",
        lambda user_id, schedule_id: user_id == "user_001" and schedule_id == 1,
    )
    monkeypatch.setattr(
        backend_main,
        "get_notes",
        lambda user_id: [row for row in state["notes"] if row["user_id"] == user_id],
    )
    monkeypatch.setattr(
        backend_main,
        "add_note",
        lambda user_id, content, is_private=True: {
            "note_id": 2,
            "user_id": user_id,
            "content": content,
            "is_private": int(is_private),
        },
    )
    monkeypatch.setattr(
        backend_main,
        "delete_note",
        lambda user_id, note_id: user_id == "user_001" and note_id == 1,
    )
    client = TestClient(create_app())

    schedules = client.get("/api/v1/users/user_001/schedules")
    created_schedule = client.post(
        "/api/v1/users/user_001/schedules",
        json={"title": "Thi", "date": "2026-08-21", "time": "09:00"},
    )
    deleted_schedule = client.delete("/api/v1/users/user_001/schedules/1")
    notes = client.get("/api/v1/users/user_001/notes")
    created_note = client.post(
        "/api/v1/users/user_001/notes",
        json={"content": "Nội dung", "is_private": True},
    )
    deleted_note = client.delete("/api/v1/users/user_001/notes/1")
    missing_user = client.get("/api/v1/users/user_999/notes")

    assert schedules.status_code == 200
    assert schedules.json()[0]["title"] == "Học thống kê"
    assert created_schedule.status_code == 200
    assert created_schedule.json()["user_id"] == "user_001"
    assert deleted_schedule.json() == {
        "success": True,
        "user_id": "user_001",
        "schedule_id": 1,
    }
    assert notes.status_code == 200
    assert notes.json()[0]["content"] == "Ghi chú riêng"
    assert created_note.status_code == 200
    assert created_note.json()["is_private"] == 1
    assert deleted_note.json() == {"success": True, "user_id": "user_001", "note_id": 1}
    assert missing_user.status_code == 404


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
