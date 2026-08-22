"""Regex-based entity extraction cho câu lệnh lịch học."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .intent_schema import Entities, Intent
from .text_normalizer import normalize_text


_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b")
_DMY_DATE_RE = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])[/-](0?[1-9]|1[0-2])(?:[/-](\d{4}))?\b"
)
_VI_DATE_RE = re.compile(
    r"\bngày\s+(0?[1-9]|[12]\d|3[01])\s+tháng\s+"
    r"(0?[1-9]|1[0-2])(?:\s+năm\s+(\d{4}))?\b"
)
_DIGIT_TIME_RE = re.compile(
    r"(?<!\w)([01]?\d|2[0-3])\s+giờ(?:\s+([0-5]?\d))?"
    r"(?:\s+(sáng|trưa|chiều|tối))?(?!\w)"
)

_NUMBER_WORDS = {
    "không": 0,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
    "mười một": 11,
    "mười hai": 12,
}
_WORD_HOURS = "|".join(
    re.escape(value) for value in sorted(_NUMBER_WORDS, key=len, reverse=True)
)
_WORD_TIME_RE = re.compile(
    rf"(?<!\w)({_WORD_HOURS})\s+giờ(?:\s+(sáng|trưa|chiều|tối))?(?!\w)"
)

_WEEKDAYS = {
    "thứ hai": 0,
    "thứ 2": 0,
    "thứ ba": 1,
    "thứ 3": 1,
    "thứ tư": 2,
    "thứ 4": 2,
    "thứ năm": 3,
    "thứ 5": 3,
    "thứ sáu": 4,
    "thứ 6": 4,
    "thứ bảy": 5,
    "thứ 7": 5,
    "chủ nhật": 6,
}
_WEEKDAY_RE = re.compile(
    r"\b(" + "|".join(re.escape(value) for value in _WEEKDAYS) + r")\b"
)

_ADD_PREFIX_RE = re.compile(
    r"^(?:xin\s+)?(?:hãy\s+)?"
    r"(?:(?:thêm|tạo|lập|đặt)\s+(?:cho tôi\s+|cho mình\s+)?(?:một\s+)?"
    r"(?:(?:lịch|buổi|cuộc hẹn)\s*)?"
    r"|tạo\s+lời\s+nhắc\s+"
    r"|nhắc\s+(?:tôi|mình)\s+)"
)
_LEADING_DATE_RE = re.compile(
    r"^(?:vào\s+)?(?:sáng\s+|chiều\s+|tối\s+)?"
    r"(?:hôm nay|ngày mai|ngày kia|"
    r"thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật|"
    r"ngày \d{1,2}(?:[/-]\d{1,2}| tháng \d{1,2})(?:[/-]\d{4}| năm \d{4})?)\s+"
)
_TEMPORAL_MARKER_RE = re.compile(
    r"\s+(?:vào\s+)?(?:lúc\s+\d|lúc\s+(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười)|"
    r"hôm nay|ngày mai|ngày kia|thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật|"
    r"ngày \d{1,2}(?:[/-]| tháng ))"
)
_ADD_PRIVATE_NOTE_PREFIX_RE = re.compile(
    r"^(?:xin\s+)?(?:hãy\s+)?"
    r"(?:thêm|tạo|lưu|ghi)\s+"
    r"(?:(?:cho tôi|cho mình)\s+)?(?:một\s+)?"
    r"(?:ghi chú|ghi chủ|ghi chỗ|note)\s+"
    r"(?:(?:riêng tư|riêng từ|cá nhân|bảo mật)\s*)?"
)
_ADD_NOTE_PREFIX_RE = re.compile(
    r"^(?:xin\s+)?(?:hãy\s+)?"
    r"(?:thêm|tạo|lưu|ghi)\s+"
    r"(?:(?:cho tôi|cho mình)\s+)?(?:một\s+)?"
    r"(?:ghi chú|ghi chủ|ghi chỗ|note)\s+"
)
_TRAILING_SECRET_RE = re.compile(
    r"\s+(?:mật khẩu|câu bí mật|lệnh bí mật|khẩu lệnh|secret phrase|passphrase)\b.*$"
)


def _reference_date(value: date | datetime | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("reference_date must use YYYY-MM-DD") from exc
    raise TypeError("reference_date must be date, datetime, ISO string, or None")


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_date(
    text: str,
    reference_date: date | datetime | str | None = None,
) -> str | None:
    """Trích xuất ngày và trả ISO ``YYYY-MM-DD``."""

    normalized = normalize_text(text)
    reference = _reference_date(reference_date)

    match = _ISO_DATE_RE.search(normalized)
    if match:
        parsed = _safe_date(*(int(value) for value in match.groups()))
        return parsed.isoformat() if parsed else None

    match = _DMY_DATE_RE.search(normalized)
    if match:
        day_value, month_value, year_value = match.groups()
        parsed = _safe_date(
            int(year_value) if year_value else reference.year,
            int(month_value),
            int(day_value),
        )
        return parsed.isoformat() if parsed else None

    match = _VI_DATE_RE.search(normalized)
    if match:
        day_value, month_value, year_value = match.groups()
        parsed = _safe_date(
            int(year_value) if year_value else reference.year,
            int(month_value),
            int(day_value),
        )
        return parsed.isoformat() if parsed else None

    if re.search(r"\bngày kia\b", normalized):
        return (reference + timedelta(days=2)).isoformat()
    if re.search(r"\b(?:ngày mai|sáng mai|chiều mai|tối mai)\b", normalized):
        return (reference + timedelta(days=1)).isoformat()
    if re.search(r"\b(?:hôm nay|sáng nay|trưa nay|chiều nay|tối nay)\b", normalized):
        return reference.isoformat()

    match = _WEEKDAY_RE.search(normalized)
    if match:
        target_weekday = _WEEKDAYS[match.group(1)]
        if "tuần sau" in normalized:
            days_ahead = 7 - reference.weekday() + target_weekday
        else:
            days_ahead = (target_weekday - reference.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
        return (reference + timedelta(days=days_ahead)).isoformat()

    return None


def _to_24_hour(hour: int, period: str | None) -> int | None:
    if hour > 23:
        return None
    if period == "sáng":
        return 0 if hour == 12 else hour
    if period in {"chiều", "tối"}:
        return hour + 12 if 1 <= hour <= 11 else hour
    if period == "trưa":
        return hour + 12 if 1 <= hour <= 11 else hour
    return hour


def extract_time(text: str) -> str | None:
    """Trích xuất giờ và trả định dạng 24 giờ ``HH:MM``."""

    normalized = normalize_text(text)
    match = _DIGIT_TIME_RE.search(normalized)
    if match:
        hour_value, minute_value, period = match.groups()
        hour = _to_24_hour(int(hour_value), period)
        if hour is None:
            return None
        return f"{hour:02d}:{int(minute_value or 0):02d}"

    match = _WORD_TIME_RE.search(normalized)
    if match:
        hour = _to_24_hour(_NUMBER_WORDS[match.group(1)], match.group(2))
        return f"{hour:02d}:00" if hour is not None else None

    return None


def extract_title(text: str) -> str | None:
    """Trích title từ câu ADD_SCHEDULE theo cấu trúc lệnh thông dụng."""

    normalized = normalize_text(text)
    body = _ADD_PREFIX_RE.sub("", normalized, count=1)
    body = _LEADING_DATE_RE.sub("", body, count=1)

    if re.match(r"^(?:vào\s+)?lúc\b", body):
        return None

    marker = _TEMPORAL_MARKER_RE.search(body)
    if marker:
        body = body[: marker.start()]

    body = re.sub(r"^(?:môn|việc)\s+", "", body)
    body = re.sub(r"\s+(?:vào|lúc)$", "", body)
    body = body.strip()
    return body or None


def extract_private_note_content(text: str) -> str | None:
    """Trích nội dung cho ADD_PRIVATE_NOTE."""

    normalized = normalize_text(text)
    body = _ADD_PRIVATE_NOTE_PREFIX_RE.sub("", normalized, count=1)
    body = _TRAILING_SECRET_RE.sub("", body, count=1)
    body = body.strip(" .,:;-")
    return body or None


def extract_note_content(text: str) -> str | None:
    """Trích nội dung cho ADD_NOTE, không sửa ngữ nghĩa nội dung."""

    normalized = normalize_text(text)
    body = _ADD_NOTE_PREFIX_RE.sub("", normalized, count=1)
    body = _TRAILING_SECRET_RE.sub("", body, count=1)
    body = body.strip(" .,:;-")
    return body or None


def extract_entities(
    text: str,
    intent: str | Intent,
    reference_date: date | datetime | str | None = None,
) -> Entities:
    """Trích entity đúng theo intent; intent khác không bị rò entity."""

    intent_value = intent.value if isinstance(intent, Intent) else intent
    entities: Entities = {}

    if intent_value == Intent.VIEW_SCHEDULE.value:
        date_value = extract_date(text, reference_date)
        if date_value:
            entities["date"] = date_value
        return entities

    if intent_value == Intent.ADD_PRIVATE_NOTE.value:
        content = extract_private_note_content(text)
        if content:
            entities["content"] = content
        return entities

    if intent_value == Intent.ADD_NOTE.value:
        content = extract_note_content(text)
        if content:
            entities["content"] = content
        return entities

    if intent_value != Intent.ADD_SCHEDULE.value:
        return entities

    title = extract_title(text)
    date_value = extract_date(text, reference_date)
    time_value = extract_time(text)
    if title:
        entities["title"] = title
    if date_value:
        entities["date"] = date_value
    if time_value:
        entities["time"] = time_value
    return entities
