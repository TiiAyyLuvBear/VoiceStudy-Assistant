"""Required-entity checks and execution gates for NLU commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .intent_schema import REQUIRED_ENTITIES, Intent


def get_missing_fields(
    intent: str | Intent,
    entities: Mapping[str, str],
) -> list[str]:
    """Return required entity names that are absent or blank."""

    intent_value = intent if isinstance(intent, Intent) else Intent(intent)
    return [
        field
        for field in REQUIRED_ENTITIES[intent_value]
        if not str(entities.get(field, "")).strip()
    ]


def can_execute_command(intent: str | Intent, missing_fields: Sequence[str]) -> bool:
    """Reject out-of-scope and incomplete commands before side effects."""

    intent_value = intent if isinstance(intent, Intent) else Intent(intent)
    return intent_value is not Intent.OUT_OF_SCOPE and not missing_fields


def can_write_database(
    intent: str | Intent,
    entities: Mapping[str, str],
) -> bool:
    """Allow an ADD_SCHEDULE database write only when all fields are present."""

    intent_value = intent if isinstance(intent, Intent) else Intent(intent)
    return (
        intent_value is Intent.ADD_SCHEDULE
        and not get_missing_fields(intent_value, entities)
    )
