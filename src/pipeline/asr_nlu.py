"""Stable audio-to-ASR-to-NLU pipeline with side-effect gates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from src.asr.whisper_model import ASRResult, transcribe_audio
from src.nlu.command_parser import parse_command
from src.nlu.intent_schema import Entities, Intent
from src.nlu.missing_fields import can_execute_command, can_write_database
from src.nlu.text_normalizer import normalize_text


class ASRNLUPipelineResult(TypedDict):
    success: bool
    transcript: str
    normalized_transcript: str
    model: str
    language: str
    latency_ms: float
    intent: str
    entities: Entities
    missing_fields: list[str]
    can_execute: bool
    can_write_database: bool
    error: str | None


Transcriber = Callable[[str | Path, str | Path], ASRResult]


def run_asr_nlu_pipeline(
    audio_path: str | Path,
    *,
    reference_date: str | None = None,
    config_path: str | Path = "config.yaml",
    transcriber: Transcriber = transcribe_audio,
) -> ASRNLUPipelineResult:
    """Transcribe audio, parse the command, and expose execution decisions."""

    asr = transcriber(audio_path, config_path)
    if not asr["success"]:
        return {
            "success": False,
            "transcript": asr["transcript"],
            "normalized_transcript": "",
            "model": asr["model"],
            "language": asr["language"],
            "latency_ms": asr["latency_ms"],
            "intent": Intent.OUT_OF_SCOPE.value,
            "entities": {},
            "missing_fields": [],
            "can_execute": False,
            "can_write_database": False,
            "error": asr["error"],
        }

    nlu = parse_command(asr["transcript"], reference_date)
    return {
        "success": True,
        "transcript": asr["transcript"],
        "normalized_transcript": normalize_text(asr["transcript"]),
        "model": asr["model"],
        "language": asr["language"],
        "latency_ms": asr["latency_ms"],
        "intent": nlu["intent"],
        "entities": nlu["entities"],
        "missing_fields": nlu["missing_fields"],
        "can_execute": can_execute_command(nlu["intent"], nlu["missing_fields"]),
        "can_write_database": can_write_database(nlu["intent"], nlu["entities"]),
        "error": None,
    }
