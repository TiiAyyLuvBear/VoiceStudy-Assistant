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
