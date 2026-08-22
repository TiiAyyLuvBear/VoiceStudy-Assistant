from __future__ import annotations

from pathlib import Path

import pytest

from src.database.user_repository import create_user
from src.pipeline.orchestrator import process_audio_request
from src.security.secret_phrase import (
    extract_secret_phrase,
    hash_secret_phrase,
    verify_secret_phrase,
    verify_transcript_secret_phrase,
)
from src.tasks.note_tasks import add_note, get_private_notes
from src.tasks.schedule_tasks import add_schedule, get_schedules


def _pipeline(
    intent: str,
    entities: dict | None = None,
    missing_fields: list[str] | None = None,
    transcript: str = "lệnh kiểm thử",
):
    def run(*args, **kwargs):
        return {
            "success": True,
            "transcript": transcript,
            "normalized_transcript": transcript,
            "model": "test-asr",
            "language": "vi",
            "latency_ms": 1.0,
            "intent": intent,
            "entities": entities or {},
            "missing_fields": missing_fields or [],
            "can_execute": not missing_fields,
            "can_write_database": intent in {"ADD_SCHEDULE", "ADD_PRIVATE_NOTE"} and not missing_fields,
            "error": None,
        }
    return run


def _sid(user_id: str = "user_001", known: bool = True):
    def identify(*args, **kwargs):
        return {
            "protocol": "APPLICATION_SID",
            "success": True,
            "candidate_user_id": user_id if known else None,
            "cosine_similarity": 0.91 if known else 0.2,
            "similarity": 0.91 if known else 0.2,
            "unknown_threshold": 0.68,
            "status": "KNOWN" if known else "UNKNOWN",
            "identified": known,
            "centroid_path": "managed-centroid.npy" if known else None,
            "latency_ms": 2.0,
            "error": None,
        }
    return identify


def _sv(verified: bool = True):
    def verify(*args, **kwargs):
        return {
            "protocol": "APPLICATION_SV",
            "success": True,
            "candidate_user_id": args[1],
            "similarity": 0.9 if verified else 0.1,
            "verification_threshold": 0.72,
            "verified": verified,
            "latency_ms": 1.5,
            "error": None,
        }
    return verify


@pytest.fixture
def system(tmp_path: Path):
    database = tmp_path / "system.db"
    audio = tmp_path / "command.wav"
    audio.write_bytes(b"audio")
    secret_hash, secret_salt = hash_secret_phrase("hoa sen xanh")
    create_user(
        "user_001",
        "User One",
        database_path=database,
        secret_phrase_hash=secret_hash,
        secret_phrase_salt=secret_salt,
    )
    secret_hash, secret_salt = hash_secret_phrase("mat trang bac")
    create_user(
        "user_002",
        "User Two",
        database_path=database,
        secret_phrase_hash=secret_hash,
        secret_phrase_salt=secret_salt,
    )
    add_schedule("user_001", "Lịch user 1", "2026-08-13", "08:00", database_path=database)
    add_schedule("user_002", "Lịch user 2", "2026-08-13", "09:00", database_path=database)
    add_note("user_001", "Ghi chú bí mật user 1", database_path=database)
    add_note("user_002", "Ghi chú bí mật user 2", database_path=database)
    return audio, database


def test_get_time_is_public_and_never_calls_sid_or_sv(system) -> None:
    audio, database = system
    def forbidden(*args, **kwargs):
        raise AssertionError("speaker API must not be called")
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("GET_TIME"),
        identifier=forbidden,
        verifier=forbidden,
    )
    assert result["success"] is True
    assert result["policy"] == "PUBLIC"
    assert result["speaker"]["candidate_user_id"] is None


def test_out_of_scope_rejects_without_sid_or_database(system) -> None:
    audio, database = system
    def forbidden(*args, **kwargs):
        raise AssertionError("speaker API must not be called")
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("OUT_OF_SCOPE"),
        identifier=forbidden,
        verifier=forbidden,
    )
    assert result["policy"] == "REJECT"
    assert result["error"] == "OUT_OF_SCOPE"


def test_view_schedule_uses_only_sid_candidate_user(system) -> None:
    audio, database = system
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline(
            "VIEW_SCHEDULE",
            {"date": "2026-08-13", "user_id": "user_002"},
        ),
        identifier=_sid("user_001"),
    )
    assert result["speaker"]["candidate_user_id"] == "user_001"
    assert "Lịch user 1" in result["response"]
    assert "Lịch user 2" not in result["response"]


def test_add_schedule_writes_to_sid_candidate_only(system) -> None:
    audio, database = system
    entities = {
        "title": "Lịch mới",
        "date": "2026-08-14",
        "time": "10:30",
        "user_id": "user_002",
    }
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("ADD_SCHEDULE", entities),
        identifier=_sid("user_001"),
    )
    assert result["error"] is None
    assert any(row["title"] == "Lịch mới" for row in get_schedules("user_001", database_path=database))
    assert not any(row["title"] == "Lịch mới" for row in get_schedules("user_002", database_path=database))


def test_add_schedule_missing_fields_does_not_write(system) -> None:
    audio, database = system
    before = len(get_schedules("user_001", database_path=database))
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline(
            "ADD_SCHEDULE",
            {"title": "Thiếu giờ", "date": "2026-08-14"},
            ["time"],
        ),
        identifier=_sid(),
    )
    assert result["error"] == "MISSING_FIELDS"
    assert len(get_schedules("user_001", database_path=database)) == before


def test_unknown_user_is_rejected_before_database_access(system) -> None:
    audio, database = system
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_SCHEDULE"),
        identifier=_sid(known=False),
    )
    assert result["error"] == "UNKNOWN_SPEAKER"
    assert result["speaker"]["status"] == "UNKNOWN"
    assert "database" not in result["stage_latency_ms"]


def test_private_note_requires_successful_sid_and_sv(system) -> None:
    audio, database = system
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_PRIVATE_NOTE", transcript="mở ghi chú riêng tư mật khẩu hoa sen xanh"),
        identifier=_sid("user_001"),
        verifier=_sv(True),
    )
    assert result["speaker"]["verified"] is True
    assert result["speaker"]["secret_phrase_verified"] is True
    assert "Ghi chú bí mật user 1" in result["response"]
    assert "Ghi chú bí mật user 2" not in result["response"]


def test_private_note_accepts_separate_secret_audio(system, tmp_path: Path) -> None:
    audio, database = system
    secret_audio = tmp_path / "secret.wav"
    secret_audio.write_bytes(b"secret-audio")
    result = process_audio_request(
        audio,
        secret_audio_path=secret_audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_PRIVATE_NOTE", transcript="mở ghi chú riêng tư"),
        secret_transcriber=lambda path, config_path: {
            "success": True,
            "transcript": "hoa sen xanh",
            "model": "test-asr",
            "language": "vi",
            "latency_ms": 1.0,
            "error": None,
        },
        identifier=_sid("user_001"),
        verifier=_sv(True),
    )
    assert result["speaker"]["secret_phrase_verified"] is True
    assert result["speaker"]["verified"] is True
    assert "Ghi chú bí mật user 1" in result["response"]


def test_add_private_note_writes_after_secret_and_sv(system) -> None:
    audio, database = system
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline(
            "ADD_PRIVATE_NOTE",
            {"content": "mã wifi ở trong tủ"},
            transcript="thêm ghi chú riêng tư mã wifi ở trong tủ mật khẩu hoa sen xanh",
        ),
        identifier=_sid("user_001"),
        verifier=_sv(True),
    )

    assert result["error"] is None
    assert result["speaker"]["secret_phrase_verified"] is True
    assert "Đã thêm ghi chú riêng tư" in result["response"]
    user_one_notes = get_private_notes("user_001", database_path=database)
    user_two_notes = get_private_notes("user_002", database_path=database)
    assert any(row["content"] == "mã wifi ở trong tủ" for row in user_one_notes)
    assert not any(row["content"] == "mã wifi ở trong tủ" for row in user_two_notes)


def test_private_note_impostor_is_rejected_before_database(system) -> None:
    audio, database = system
    result = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_PRIVATE_NOTE", transcript="mở ghi chú riêng tư mật khẩu sai hoàn toàn"),
        identifier=_sid("user_002"),
        verifier=_sv(False),
    )
    assert result["error"] == "SECRET_PHRASE_FAILED"
    assert result["speaker"]["secret_phrase_verified"] is False
    assert "Ghi chú" not in result["response"]
    assert "database" not in result["stage_latency_ms"]


def test_empty_transcript_and_bad_audio_are_stable(system, tmp_path: Path) -> None:
    audio, database = system
    empty = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("OUT_OF_SCOPE", transcript=""),
    )
    missing = process_audio_request(tmp_path / "missing.wav", database_path=database)
    assert empty["error"] == "EMPTY_TRANSCRIPT"
    assert empty["success"] is False
    assert missing["success"] is False
    assert missing["error"]


def test_sid_and_sv_exceptions_are_stable(system) -> None:
    audio, database = system
    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    sid_error = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_SCHEDULE"),
        identifier=broken,
    )
    sv_error = process_audio_request(
        audio,
        database_path=database,
        asr_nlu_runner=_pipeline("VIEW_PRIVATE_NOTE", transcript="mở ghi chú riêng tư mật khẩu hoa sen xanh"),
        identifier=_sid(),
        verifier=broken,
    )
    assert sid_error["success"] is False
    assert sid_error["error"].startswith("SID_ERROR")
    assert sv_error["success"] is False
    assert sv_error["error"].startswith("SV_ERROR")


def test_secret_phrase_hash_and_transcript_verification(tmp_path: Path) -> None:
    digest, salt = hash_secret_phrase("Hoa Sen Xanh")
    assert verify_secret_phrase("hoa sen xanh", digest, salt) is True
    assert verify_secret_phrase("hoa sen đỏ", digest, salt) is False
    assert extract_secret_phrase("mở ghi chú riêng tư mật khẩu hoa sen xanh") == "hoa sen xanh"

    database = tmp_path / "users.db"
    create_user(
        "user_003",
        "Secret User",
        database_path=database,
        secret_phrase_hash=digest,
        secret_phrase_salt=salt,
    )

    ok, error = verify_transcript_secret_phrase(
        "user_003",
        "mở ghi chú riêng tư mật khẩu hoa sen xanh",
        database_path=database,
    )
    assert ok is True
    assert error is None

    ok, error = verify_transcript_secret_phrase(
        "user_003",
        "mở ghi chú riêng tư mật khẩu sai hoàn toàn",
        database_path=database,
    )
    assert ok is False
    assert error == "SECRET_PHRASE_FAILED"

