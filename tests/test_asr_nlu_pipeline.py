"""Tests for the stable ASR/NLU pipeline without loading Whisper."""

from pathlib import Path

from src.asr.whisper_model import ASRResult
from src.pipeline.asr_nlu import run_asr_nlu_pipeline


def _successful_transcriber(
    audio_path: str | Path,
    config_path: str | Path,
) -> ASRResult:
    return {
        "transcript": (
            "Th\u00eam l\u1ecbch h\u1ecdc m\u00e1y l\u00fac "
            "8 gi\u1edd s\u00e1ng mai"
        ),
        "model": "whisper-small",
        "language": "vi",
        "latency_ms": 100.0,
        "success": True,
        "error": None,
    }


def _failed_transcriber(
    audio_path: str | Path,
    config_path: str | Path,
) -> ASRResult:
    return {
        "transcript": "",
        "model": "whisper-small",
        "language": "vi",
        "latency_ms": 1.0,
        "success": False,
        "error": "missing audio",
    }


def test_pipeline_returns_nlu_and_execution_gates() -> None:
    result = run_asr_nlu_pipeline(
        "sample.wav",
        reference_date="2026-07-28",
        transcriber=_successful_transcriber,
    )

    assert result["success"]
    assert result["intent"] == "ADD_SCHEDULE"
    assert result["entities"] == {
        "title": "h\u1ecdc m\u00e1y",
        "date": "2026-07-29",
        "time": "08:00",
    }
    assert result["missing_fields"] == []
    assert result["command_text"] == "Thêm lịch học máy vào 2026-07-29 lúc 08:00."
    assert result["can_execute"]
    assert result["can_write_database"]


def test_pipeline_stops_after_asr_failure() -> None:
    result = run_asr_nlu_pipeline(
        "missing.wav",
        transcriber=_failed_transcriber,
    )

    assert not result["success"]
    assert result["intent"] == "OUT_OF_SCOPE"
    assert not result["can_execute"]
    assert not result["can_write_database"]
    assert result["error"] == "missing audio"


def test_pipeline_snaps_noisy_asr_to_fixed_command() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "bây giờ là mấy giờ rùi",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "GET_TIME"
    assert result["command_text"] == "Bây giờ là mấy giờ rồi?"
    assert result["asr_postprocessed"] is True


def test_pipeline_does_not_snap_slot_command_and_preserves_note_content() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "thêm ghi chú riêng tư mua thuốc chiều nay",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "ADD_PRIVATE_NOTE"
    assert result["entities"] == {"content": "mua thuốc chiều nay"}
    assert result["command_text"] == "Thêm ghi chú riêng tư mua thuốc chiều nay."
    assert result["asr_postprocessed"] is False


def test_pipeline_displays_processed_noisy_slot_command_not_raw_asr() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "thêm ghi chủ riêng từ mả sổ thể trên giấy",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "ADD_PRIVATE_NOTE"
    assert result["transcript"] == "thêm ghi chủ riêng từ mả sổ thể trên giấy"
    assert result["command_text"] == "Thêm ghi chú riêng tư mả sổ thể trên giấy."
    assert result["normalized_transcript"] == "thêm ghi chú riêng tư mả sổ thể trên giấy"


def test_pipeline_does_not_semantic_rewrite_private_note_content() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "Thêm ghi chỗ riêng tư họp thống tế",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "ADD_PRIVATE_NOTE"
    assert result["entities"] == {"content": "họp thống tế"}
    assert result["raw_content"] == "họp thống tế"
    assert result["normalized_content"] == "Họp thống tế."
    assert result["command_text"] == "Thêm ghi chú riêng tư họp thống tế."
    assert result["normalized_transcript"] == "thêm ghi chú riêng tư họp thống tế"


def test_pipeline_keeps_noisy_view_private_note_as_view() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "huyển thị ghi chú riêng tư mới nhất.",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "VIEW_PRIVATE_NOTE"
    assert result["entities"] == {}
    assert result["command_text"] == "Mở ghi chú riêng tư gần nhất của tôi."


def test_pipeline_supports_public_note_command() -> None:
    def transcriber(audio_path: str | Path, config_path: str | Path) -> ASRResult:
        return {
            "transcript": "thêm ghi chú gọi cho an",
            "model": "whisper-small",
            "language": "vi",
            "latency_ms": 1.0,
            "success": True,
            "error": None,
        }

    result = run_asr_nlu_pipeline("sample.wav", transcriber=transcriber)

    assert result["success"]
    assert result["intent"] == "ADD_NOTE"
    assert result["entities"] == {"content": "gọi cho an"}
    assert result["raw_content"] == "gọi cho an"
    assert result["normalized_content"] == "Gọi cho an."
    assert result["command_text"] == "Thêm ghi chú gọi cho an."
