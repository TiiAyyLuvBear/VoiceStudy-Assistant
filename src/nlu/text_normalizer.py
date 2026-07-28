"""Chuẩn hóa transcript tiếng Việt trước khi phân loại intent."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_AM_PM_RE = re.compile(
    r"(?<!\w)([01]?\d|2[0-3])(?:\s*[:h]\s*([0-5]\d))?"
    r"\s*(a\.?\s*m\.?|p\.?\s*m\.?)(?!\w)",
    flags=re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"(?<![\w/.-])([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?!\d)",
    flags=re.IGNORECASE,
)
_HOUR_RE = re.compile(
    r"(?<![\w/.-])([01]?\d|2[0-3])\s*h(?![\w:])",
    flags=re.IGNORECASE,
)
_HOUR_MINUTE_WORD_RE = re.compile(
    r"(?<!\w)([01]?\d|2[0-3])\s+giờ\s+([0-5]?\d)\s+phút(?!\w)",
    flags=re.IGNORECASE,
)


def _format_hour_minute(hour: str, minute: str | None = None) -> str:
    normalized_hour = str(int(hour))
    if minute is None or int(minute) == 0:
        return f"{normalized_hour} giờ"
    return f"{normalized_hour} giờ {int(minute):02d}"


def _replace_am_pm(match: re.Match[str]) -> str:
    period = re.sub(r"[.\s]", "", match.group(3)).lower()
    vietnamese_period = "sáng" if period == "am" else "chiều"
    return f"{_format_hour_minute(match.group(1), match.group(2))} {vietnamese_period}"


def _replace_clock(match: re.Match[str]) -> str:
    return _format_hour_minute(match.group(1), match.group(2))


def _replace_hour(match: re.Match[str]) -> str:
    return _format_hour_minute(match.group(1))


def _replace_word_minutes(match: re.Match[str]) -> str:
    return _format_hour_minute(match.group(1), match.group(2))


def _remove_unnecessary_punctuation(text: str) -> str:
    """Bỏ dấu câu nhưng giữ ``/`` và ``-`` khi chúng nằm trong ngày số."""

    output: list[str] = []
    for index, char in enumerate(text):
        if char in {"/", "-"}:
            previous_is_digit = index > 0 and text[index - 1].isdigit()
            next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
            output.append(char if previous_is_digit and next_is_digit else " ")
            continue

        category = unicodedata.category(char)
        output.append(" " if category.startswith(("P", "S")) else char)

    return "".join(output)


def normalize_text(text: str) -> str:
    """Chuẩn hóa transcript và vẫn giữ nguyên dấu tiếng Việt.

    Các dạng ``8h``, ``8h30``, ``08:30`` và ``8 PM`` lần lượt được đưa về
    cách viết chứa ``giờ`` để entity extractor chỉ cần xử lý một dạng chính.
    Hàm này không biến đổi chữ tiếng Việt có dấu thành không dấu.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = _AM_PM_RE.sub(_replace_am_pm, normalized)
    normalized = _CLOCK_RE.sub(_replace_clock, normalized)
    normalized = _HOUR_RE.sub(_replace_hour, normalized)
    normalized = _HOUR_MINUTE_WORD_RE.sub(_replace_word_minutes, normalized)
    normalized = _remove_unnecessary_punctuation(normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
