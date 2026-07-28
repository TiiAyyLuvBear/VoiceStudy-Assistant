"""Kết hợp normalization, intent classification và entity extraction."""

from __future__ import annotations

from datetime import date, datetime

from .entity_extractor import extract_entities
from .intent_classifier import classify_intent
from .intent_schema import NLUResult, REQUIRED_ENTITIES, Intent
from .text_normalizer import normalize_text


def parse_command(
    transcript: str,
    reference_date: date | datetime | str | None = None,
) -> NLUResult:
    """Phân tích transcript thành output JSON-safe thống nhất."""

    normalized = normalize_text(transcript)
    intent_value = classify_intent(normalized)
    intent = Intent(intent_value)
    entities = extract_entities(normalized, intent, reference_date)
    missing_fields = [
        field for field in REQUIRED_ENTITIES[intent] if not entities.get(field)
    ]
    return {
        "intent": intent.value,
        "entities": entities,
        "missing_fields": missing_fields,
    }
