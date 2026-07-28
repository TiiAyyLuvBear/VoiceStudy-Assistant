"""Unit test cho ASR wrapper mà không tải model thật."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.asr.whisper_model import WhisperASR, WhisperConfig, load_whisper_config


class FakeWhisperModel:
    def __init__(self, texts: list[str] | None = None, error: Exception | None = None):
        self.texts = texts if texts is not None else [" Xin chào ", " các bạn. "]
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def transcribe(self, audio: str, **kwargs: Any):
        self.calls.append((audio, kwargs))
        if self.error is not None:
            raise self.error
        return (iter(SimpleNamespace(text=text) for text in self.texts), object())


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 1600)


def test_project_config_selects_whisper_small_cpu() -> None:
    config = load_whisper_config("config.yaml")
    assert config.model_size == "small"
    assert config.model_name == "whisper-small"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.language == "vi"


def test_transcribe_returns_contract_and_vietnamese_options(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)
    fake_model = FakeWhisperModel()
    asr = WhisperASR(WhisperConfig(), model=fake_model)

    result = asr.transcribe(audio_path)

    assert result["transcript"] == "Xin chào các bạn."
    assert result["model"] == "whisper-small"
    assert result["language"] == "vi"
    assert result["success"] is True
    assert result["error"] is None
    assert result["latency_ms"] >= 0
    assert fake_model.calls[0][1]["language"] == "vi"
    assert fake_model.calls[0][1]["task"] == "transcribe"
    assert fake_model.calls[0][1]["beam_size"] == 5


def test_missing_audio_is_reported_without_loading_model(tmp_path: Path) -> None:
    fake_model = FakeWhisperModel()
    asr = WhisperASR(WhisperConfig(), model=fake_model)

    result = asr.transcribe(tmp_path / "missing.wav")

    assert result["success"] is False
    assert result["transcript"] == ""
    assert "does not exist" in str(result["error"])
    assert fake_model.calls == []


def test_empty_transcript_is_failure(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    _write_silent_wav(audio_path)
    asr = WhisperASR(WhisperConfig(), model=FakeWhisperModel(texts=[" "]))

    result = asr.transcribe(audio_path)

    assert result["success"] is False
    assert "did not detect speech" in str(result["error"])


def test_inference_error_is_returned_to_orchestrator(tmp_path: Path) -> None:
    audio_path = tmp_path / "broken.wav"
    _write_silent_wav(audio_path)
    failing_model = FakeWhisperModel(error=RuntimeError("decoder failure"))
    asr = WhisperASR(WhisperConfig(), model=failing_model)

    result = asr.transcribe(audio_path)

    assert result["success"] is False
    assert result["error"] == "ASR inference failed: decoder failure"
