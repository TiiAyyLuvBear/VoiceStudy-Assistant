"""Stable audio-to-ASR-to-NLU pipeline with side-effect gates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from src.asr.whisper_model import ASRResult, transcribe_audio
from src.nlu.command_catalog import DEFAULT_POSTPROCESSOR
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
    command_text: str
    asr_postprocessed: bool
    detected_command_text: str | None
    normalized_command_text: str | None
    raw_content: str | None
    normalized_content: str | None
    final_content: str | None
    command_match_score: float | None
    requires_user_confirmation: bool


Transcriber = Callable[[str | Path, str | Path], ASRResult]


def _display_command_text(intent: str, entities: Entities, fallback: str) -> str:
    """Build production-safe command text after NLU processing."""

    if intent == Intent.GET_TIME.value:
        return "Bây giờ là mấy giờ rồi?"
    if intent == Intent.VIEW_PRIVATE_NOTE.value:
        return "Mở ghi chú riêng tư gần nhất của tôi."
    if intent == Intent.ADD_PRIVATE_NOTE.value:
        content = str(entities.get("content", "")).strip()
        return f"Thêm ghi chú riêng tư {content}." if content else "Thêm ghi chú riêng tư."
    if intent == Intent.ADD_NOTE.value:
        content = str(entities.get("content", "")).strip()
        return f"Thêm ghi chú {content}." if content else "Thêm ghi chú."
    if intent == Intent.VIEW_SCHEDULE.value:
        date_value = str(entities.get("date", "")).strip()
        return f"Cho tôi xem lịch ngày {date_value}." if date_value else "Cho tôi xem lịch."
    if intent == Intent.ADD_SCHEDULE.value:
        title = str(entities.get("title", "")).strip()
        date_value = str(entities.get("date", "")).strip()
        time_value = str(entities.get("time", "")).strip()
        if title and date_value and time_value:
            return f"Thêm lịch {title} vào {date_value} lúc {time_value}."
    return normalize_text(fallback)


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
            "command_text": "",
            "asr_postprocessed": False,
            "detected_command_text": None,
            "normalized_command_text": None,
            "raw_content": None,
            "normalized_content": None,
            "final_content": None,
            "command_match_score": None,
            "requires_user_confirmation": False,
        }

    processed = DEFAULT_POSTPROCESSOR.process(asr["transcript"])
    if processed.intent is None:
        command_text = asr["transcript"]
    elif processed.raw_content:
        command_text = f"{processed.normalized_command_text} {processed.raw_content}"
    else:
        command_text = processed.normalized_command_text or asr["transcript"]
    nlu = parse_command(command_text, reference_date)
    display_text = _display_command_text(
        nlu["intent"],
        nlu["entities"],
        command_text,
    )
    return {
        "success": True,
        "transcript": asr["transcript"],
        "normalized_transcript": normalize_text(display_text),
        "model": asr["model"],
        "language": asr["language"],
        "latency_ms": asr["latency_ms"],
        "intent": nlu["intent"],
        "entities": nlu["entities"],
        "missing_fields": nlu["missing_fields"],
        "can_execute": can_execute_command(nlu["intent"], nlu["missing_fields"]),
        "can_write_database": can_write_database(nlu["intent"], nlu["entities"]),
        "error": None,
        "command_text": display_text,
        "asr_postprocessed": (
            normalize_text(asr["transcript"]) != normalize_text(display_text)
        ),
        "detected_command_text": processed.detected_command_text,
        "normalized_command_text": processed.normalized_command_text,
        "raw_content": processed.raw_content,
        "normalized_content": processed.normalized_content,
        "final_content": processed.final_content,
        "command_match_score": processed.command_match_score,
        "requires_user_confirmation": (
            processed.requires_user_confirmation or bool(nlu["missing_fields"])
        ),
    }
