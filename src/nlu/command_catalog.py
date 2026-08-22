"""Fixed user command catalog and ASR post-processing helpers."""

from __future__ import annotations

from typing import Final, TypedDict

from src.nlu.asr_postprocessor import ASRPostProcessor, CommandSpec
from src.nlu.intent_schema import Intent
from src.nlu.text_normalizer import normalize_text
from src.utils.fuzzy_match import fuzzy_match


class SystemCommand(TypedDict, total=False):
    intent: str
    phrase: str
    requires_secret: bool
    slots: list[str]


SYSTEM_COMMANDS: Final[tuple[SystemCommand, ...]] = (
    {
        "intent": Intent.GET_TIME.value,
        "phrase": "Bây giờ là mấy giờ rồi?",
        "requires_secret": False,
    },
    {
        "intent": Intent.VIEW_SCHEDULE.value,
        "phrase": "Cho tôi xem lịch hôm nay.",
        "requires_secret": False,
    },
    {
        "intent": Intent.VIEW_SCHEDULE.value,
        "phrase": "Cho tôi xem lịch ngày mai.",
        "requires_secret": False,
    },
    {
        "intent": Intent.ADD_SCHEDULE.value,
        "phrase": "Thêm lịch <tiêu đề> vào <ngày> lúc <giờ>.",
        "requires_secret": False,
        "slots": ["title", "date", "time"],
    },
    {
        "intent": Intent.ADD_PRIVATE_NOTE.value,
        "phrase": "Thêm ghi chú riêng tư <nội dung>.",
        "requires_secret": True,
        "slots": ["content"],
    },
    {
        "intent": Intent.ADD_NOTE.value,
        "phrase": "Thêm ghi chú <nội dung>.",
        "requires_secret": False,
        "slots": ["content"],
    },
    {
        "intent": Intent.VIEW_PRIVATE_NOTE.value,
        "phrase": "Mở ghi chú riêng tư gần nhất của tôi.",
        "requires_secret": True,
    },
)

COMMAND_REGISTRY: Final[tuple[CommandSpec, ...]] = (
    {
        "intent": Intent.GET_TIME.value,
        "patterns": ("bây giờ là mấy giờ", "mấy giờ rồi"),
        "has_content": False,
        "requires_secret": False,
    },
    {
        "intent": Intent.VIEW_SCHEDULE.value,
        "patterns": ("cho tôi xem lịch hôm nay", "cho tôi xem lịch ngày mai"),
        "has_content": False,
        "requires_secret": False,
    },
    {
        "intent": Intent.ADD_SCHEDULE.value,
        "patterns": (
            "thêm lịch",
            "tạo lịch",
            "lập lịch",
            "đặt lịch",
            "tạo lời nhắc",
            "nhắc tôi",
        ),
        "has_content": True,
        "requires_secret": False,
    },
    {
        "intent": Intent.ADD_PRIVATE_NOTE.value,
        "patterns": (
            "thêm ghi chú riêng tư",
            "tạo ghi chú riêng tư",
            "lưu ghi chú riêng tư",
        ),
        "has_content": True,
        "requires_secret": True,
    },
    {
        "intent": Intent.ADD_NOTE.value,
        "patterns": ("thêm ghi chú", "tạo ghi chú", "lưu ghi chú"),
        "has_content": True,
        "requires_secret": False,
    },
    {
        "intent": Intent.VIEW_PRIVATE_NOTE.value,
        "patterns": (
            "mở ghi chú riêng tư",
            "xem ghi chú riêng tư",
            "hiển thị ghi chú riêng tư",
            "huyển thị ghi chú riêng tư",
            "đọc ghi chú riêng tư",
        ),
        "has_content": False,
        "requires_secret": True,
    },
)
DEFAULT_POSTPROCESSOR: Final[ASRPostProcessor] = ASRPostProcessor(COMMAND_REGISTRY)

_PHRASE_TO_INTENT: Final[dict[str, str]] = {
    normalize_text(command["phrase"]): command["intent"]
    for command in SYSTEM_COMMANDS
    if "slots" not in command
}
_DISPLAY_BY_NORMALIZED: Final[dict[str, str]] = {
    normalize_text(command["phrase"]): command["phrase"]
    for command in SYSTEM_COMMANDS
    if "slots" not in command
}


def fixed_command_catalog() -> list[dict]:
    return [dict(command) for command in SYSTEM_COMMANDS]


def command_registry() -> list[dict]:
    return [dict(command) for command in COMMAND_REGISTRY]


def postprocess_asr_command(transcript: str) -> tuple[str, str | None]:
    """Snap noisy ASR text to one fixed command when confidence is enough."""

    normalized = normalize_text(transcript)
    hit = fuzzy_match(
        normalized,
        _PHRASE_TO_INTENT,
        threshold=76.0,
        margin=8.0,
        min_words=3,
    )
    if hit is None:
        return transcript, None
    matched_phrase = _DISPLAY_BY_NORMALIZED[normalize_text(hit["matched"])]
    return matched_phrase, matched_phrase
