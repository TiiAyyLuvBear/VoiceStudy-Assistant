"""Định nghĩa intent chính thức và schema đầu ra NLU."""

from __future__ import annotations

from enum import Enum
from typing import Final, TypedDict


class Intent(str, Enum):
    """Năm nhãn intent duy nhất được phép trong hệ thống."""

    GET_TIME = "GET_TIME"
    VIEW_SCHEDULE = "VIEW_SCHEDULE"
    ADD_SCHEDULE = "ADD_SCHEDULE"
    ADD_NOTE = "ADD_NOTE"
    ADD_PRIVATE_NOTE = "ADD_PRIVATE_NOTE"
    VIEW_PRIVATE_NOTE = "VIEW_PRIVATE_NOTE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


SUPPORTED_INTENTS: Final[tuple[str, ...]] = tuple(intent.value for intent in Intent)


class Entities(TypedDict, total=False):
    """Các entity mà command parser có thể trả về."""

    title: str
    date: str
    time: str
    content: str


class NLUResult(TypedDict):
    """Schema đầu ra thống nhất của NLU."""

    intent: str
    entities: Entities
    missing_fields: list[str]


REQUIRED_ENTITIES: Final[dict[Intent, tuple[str, ...]]] = {
    Intent.GET_TIME: (),
    Intent.VIEW_SCHEDULE: (),
    Intent.ADD_SCHEDULE: ("title", "date", "time"),
    Intent.ADD_NOTE: ("content",),
    Intent.ADD_PRIVATE_NOTE: ("content",),
    Intent.VIEW_PRIVATE_NOTE: (),
    Intent.OUT_OF_SCOPE: (),
}


def is_valid_intent(value: str) -> bool:
    """Trả về ``True`` nếu value là một intent chính thức."""

    return value in SUPPORTED_INTENTS
