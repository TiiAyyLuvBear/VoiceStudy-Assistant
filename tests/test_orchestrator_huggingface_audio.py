from __future__ import annotations

from pathlib import Path

from src.pipeline import orchestrator


def _public_pipeline() -> dict:
    return {
        "success": True,
        "transcript": "mấy giờ rồi",
        "normalized_transcript": "mấy giờ rồi",
        "intent": "GET_TIME",
        "entities": {},
        "missing_fields": [],
        "error": None,
    }


def test_orchestrator_passes_resolved_audio_to_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cached = tmp_path / "cached.wav"
    cached.write_bytes(b"audio")
    calls: list[Path] = []

    monkeypatch.setattr(
        orchestrator,
        "resolve_audio_path",
        lambda value: cached,
    )

    def runner(audio_path, **kwargs):
        calls.append(Path(audio_path))
        return _public_pipeline()

    result = orchestrator.process_audio_request(
        "data/commands/audio/test/remote.wav",
        asr_nlu_runner=runner,
    )

    assert result["success"] is True
    assert calls == [cached]


def test_orchestrator_reports_remote_audio_resolution_failure(
    monkeypatch,
) -> None:
    def fail(value):
        raise FileNotFoundError(f"missing: {value}")

    monkeypatch.setattr(orchestrator, "resolve_audio_path", fail)
    result = orchestrator.process_audio_request(
        "data/commands/audio/test/missing.wav"
    )

    assert result["success"] is False
    assert "missing" in str(result["error"])
