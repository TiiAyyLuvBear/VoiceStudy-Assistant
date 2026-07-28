"""Rule-based intent classifier cho các lệnh chính thức của hệ thống."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .intent_schema import Intent
from .text_normalizer import normalize_text


_NOTE_TERM = re.compile(r"\b(?:ghi chú|note)\b")
_VIEW_ACTION = re.compile(r"\b(?:xem|mở|đọc|hiển thị|cho tôi biết)\b")
_PRIVATE_MARKER = re.compile(r"\b(?:riêng tư|cá nhân|bảo mật|của tôi)\b")

_ADD_PATTERNS = (
    re.compile(r"\b(?:thêm|tạo|lập|đặt)\b.{0,35}\b(?:lịch|buổi(?: học)?|cuộc hẹn)\b"),
    re.compile(r"\b(?:thêm|tạo|lập|đặt)\s+lịch\b"),
    re.compile(r"^(?:hãy\s+)?nhắc\s+(?:tôi|mình)\b"),
)

_VIEW_SCHEDULE_PATTERNS = (
    re.compile(r"\b(?:xem|mở|đọc|hiển thị|kiểm tra)\b.{0,30}\b(?:lịch|thời khóa biểu)\b"),
    re.compile(r"\b(?:lịch|thời khóa biểu)\b.{0,30}\b(?:của tôi|của mình)\b"),
    re.compile(r"\b(?:hôm nay|ngày mai|ngày kia|thứ [a-z0-9]+)\b.{0,20}\bcó\b.{0,20}\b(?:lịch|môn|buổi)\b"),
    re.compile(r"\btôi\s+có\s+(?:lịch|môn|buổi)\s+gì\b"),
)

_GET_TIME_PATTERNS = (
    re.compile(r"\b(?:bây giờ|hiện tại)\b.{0,20}\b(?:mấy giờ|thời gian)\b"),
    re.compile(r"\bmấy giờ\s+rồi\b"),
    re.compile(r"\b(?:cho tôi biết|đọc|báo)\b.{0,20}\b(?:giờ hiện tại|thời gian hiện tại)\b"),
    re.compile(r"^(?:xin\s+)?(?:cho biết\s+)?thời gian hiện tại$"),
)


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_intent(text: str) -> str:
    """Phân loại một câu vào đúng một trong năm intent.

    Rule cụ thể được ưu tiên trước rule tổng quát. Khi không có bằng chứng rõ
    ràng, hàm luôn trả ``OUT_OF_SCOPE`` thay vì đoán một chức năng nhạy cảm.
    """

    normalized = normalize_text(text)
    if not normalized:
        return Intent.OUT_OF_SCOPE.value

    if (
        _NOTE_TERM.search(normalized)
        and _VIEW_ACTION.search(normalized)
        and _PRIVATE_MARKER.search(normalized)
    ):
        return Intent.VIEW_PRIVATE_NOTE.value

    if _matches_any(normalized, _ADD_PATTERNS):
        return Intent.ADD_SCHEDULE.value

    if _matches_any(normalized, _VIEW_SCHEDULE_PATTERNS):
        return Intent.VIEW_SCHEDULE.value

    if _matches_any(normalized, _GET_TIME_PATTERNS):
        return Intent.GET_TIME.value

    return Intent.OUT_OF_SCOPE.value
