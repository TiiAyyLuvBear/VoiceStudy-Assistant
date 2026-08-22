"""Rule-based intent classifier with fuzzy fallback for noisy ASR."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .intent_schema import Intent
from .text_normalizer import normalize_text
from src.utils.fuzzy_match import fuzzy_match

_NOTE_TERM = re.compile(r"\b(?:ghi ch\u00fa|note)\b")
_VIEW_ACTION = re.compile(r"\b(?:xem|m\u1edf|\u0111\u1ecdc|hi\u1ec3n th\u1ecb|huy\u1ec3n th\u1ecb|cho t\u00f4i bi\u1ebft)\b")
_ADD_NOTE_PATTERN = re.compile(
    r"\b(?:th\u00eam|t\u1ea1o|l\u01b0u)\b.{0,30}\b(?:ghi ch\u00fa|note)\b|"
    r"\bghi\s+(?:ghi ch\u00fa|note)\b"
)
_PRIVATE_MARKER = re.compile(r"\b(?:ri\u00eang t\u01b0|c\u00e1 nh\u00e2n|b\u1ea3o m\u1eadt|c\u1ee7a t\u00f4i)\b")

_ADD_PATTERNS = (
    re.compile(r"\b(?:th\u00eam|t\u1ea1o|l\u1eadp|\u0111\u1eb7t)\b.{0,35}\b(?:l\u1ecbch|bu\u1ed5i(?: h\u1ecdc)?|cu\u1ed9c h\u1eb9n)\b"),
    re.compile(r"\b(?:th\u00eam|t\u1ea1o|l\u1eadp|\u0111\u1eb7t)\s+l\u1ecbch\b"),
    re.compile(r"\bt\u1ea1o\s+l\u1eddi\s+nh\u1eafc\b"),
    re.compile(r"^(?:h\u00e3y\s+)?nh\u1eafc\s+(?:t\u00f4i|m\u00ecnh)\b"),
)

_VIEW_SCHEDULE_PATTERNS = (
    re.compile(r"\b(?:xem|m\u1edf|\u0111\u1ecdc|hi\u1ec3n th\u1ecb|ki\u1ec3m tra)\b.{0,30}\b(?:l\u1ecbch|th\u1eddi kh\u00f3a bi\u1ec3u)\b"),
    re.compile(r"\b(?:l\u1ecbch|th\u1eddi kh\u00f3a bi\u1ec3u)\b.{0,30}\b(?:c\u1ee7a t\u00f4i|c\u1ee7a m\u00ecnh)\b"),
    re.compile(r"\b(?:h\u00f4m nay|ng\u00e0y mai|ng\u00e0y kia|th\u1ee9 [a-z0-9]+)\b.{0,20}\bc\u00f3\b.{0,20}\b(?:l\u1ecbch|m\u00f4n|bu\u1ed5i)\b"),
    re.compile(r"\bt\u00f4i\s+c\u00f3\s+(?:l\u1ecbch|m\u00f4n|bu\u1ed5i)\s+g\u00ec\b"),
)

_GET_TIME_PATTERNS = (
    re.compile(r"mấy giờ\s+rồi"),
    re.compile(r"báo.{0,20}giờ hiện tại"),
    re.compile(r"\b(?:b\u00e2y gi\u1edf|hi\u1ec7n t\u1ea1i)\b.{0,20}\b(?:m\u1ea5y gi\u1edf|th\u1eddi gian)\b"),
    re.compile(r"\bm\u1ea5y gi\u1edf\s+r\u1ed3i\b"),
    re.compile(r"^(?:xin\s+)?m\u1ea5y gi\u1edf\s+r\u1ed3i$"),
    re.compile(r"\bb\u00e1o\b.{0,20}\bgi\u1edd hi\u1ec7n t\u1ea1i\b"),
    re.compile(r"\b(?:cho t\u00f4i bi\u1ebft|\u0111\u1ecdc|b\u00e1o)\b.{0,20}\b(?:gi\u1edf hi\u1ec7n t\u1ea1i|th\u1eddi gian hi\u1ec7n t\u1ea1i)\b"),
    re.compile(r"^(?:xin\s+)?(?:cho bi\u1ebft\s+)?th\u1eddi gian hi\u1ec7n t\u1ea1i$"),
)


# ---------------------------------------------------------------------------
# Fuzzy-match candidates: phrase -> intent value
# ---------------------------------------------------------------------------

FUZZY_CANDIDATES: dict[str, str] = {
    # GET_TIME
    "b\u00e2y gi\u1edf m\u1ea5y gi\u1edf": Intent.GET_TIME.value,
    "m\u1ea5y gi\u1edf r\u1ed3i": Intent.GET_TIME.value,
    "cho t\u00f4i bi\u1ebft gi\u1edf hi\u1ec7n t\u1ea1i": Intent.GET_TIME.value,
    "th\u1eddi gian hi\u1ec7n t\u1ea1i": Intent.GET_TIME.value,
    "b\u00e2y gi\u1edf l\u00e0 m\u1ea5y gi\u1edf": Intent.GET_TIME.value,
    "gi\u1edf hi\u1ec7n t\u1ea1i l\u00e0 m\u1ea5y": Intent.GET_TIME.value,
    "b\u00e2y gi\u1edf th\u1eddi gian th\u1ebf n\u00e0o": Intent.GET_TIME.value,
    "hi\u1ec7n t\u1ea1i ch\u00ednh x\u00e1c l\u00e0 m\u1ea5y gi\u1edf": Intent.GET_TIME.value,
    "b\u00e1o gi\u1edf hi\u1ec7n t\u1ea1i": Intent.GET_TIME.value,
    "\u0111\u1ecdc th\u1eddi gian hi\u1ec7n t\u1ea1i": Intent.GET_TIME.value,
    "xin cho t\u00f4i bi\u1ebft m\u1ea5y gi\u1edf": Intent.GET_TIME.value,
    # VIEW_SCHEDULE
    "xem l\u1ecbch h\u00f4m nay": Intent.VIEW_SCHEDULE.value,
    "xem l\u1ecbch ng\u00e0y mai": Intent.VIEW_SCHEDULE.value,
    "cho t\u00f4i xem l\u1ecbch": Intent.VIEW_SCHEDULE.value,
    "l\u1ecbch c\u1ee7a t\u00f4i": Intent.VIEW_SCHEDULE.value,
    "xem th\u1eddi kh\u00f3a bi\u1ec3u": Intent.VIEW_SCHEDULE.value,
    "ki\u1ec3m tra l\u1ecbch": Intent.VIEW_SCHEDULE.value,
    "h\u00f4m nay c\u00f3 l\u1ecbch g\u00ec": Intent.VIEW_SCHEDULE.value,
    "t\u00f4i c\u00f3 l\u1ecbch g\u00ec": Intent.VIEW_SCHEDULE.value,
    "ng\u00e0y mai c\u00f3 bu\u1ed5i g\u00ec": Intent.VIEW_SCHEDULE.value,
    "ng\u00e0y mai m\u00ecnh c\u00f3 bu\u1ed5i g\u00ec": Intent.VIEW_SCHEDULE.value,
    "\u0111\u1ecdc l\u1ecbch h\u1ecdc": Intent.VIEW_SCHEDULE.value,
    # ADD_SCHEDULE
    "th\u00eam l\u1ecbch h\u1ecdc": Intent.ADD_SCHEDULE.value,
    "t\u1ea1o l\u1ecbch h\u1ecdc": Intent.ADD_SCHEDULE.value,
    "th\u00eam l\u1ecbch": Intent.ADD_SCHEDULE.value,
    "t\u1ea1o l\u1ecbch": Intent.ADD_SCHEDULE.value,
    "\u0111\u1eb7t l\u1ecbch h\u1ecdc": Intent.ADD_SCHEDULE.value,
    "l\u1eadp l\u1ecbch": Intent.ADD_SCHEDULE.value,
    "nh\u1eafc t\u00f4i": Intent.ADD_SCHEDULE.value,
    "t\u1ea1o l\u1eddi nh\u1eafc": Intent.ADD_SCHEDULE.value,
    "th\u00eam bu\u1ed5i h\u1ecdc": Intent.ADD_SCHEDULE.value,
    "th\u00eam cu\u1ed9c h\u1eb9n": Intent.ADD_SCHEDULE.value,
    "l\u1eadp cu\u1ed9c h\u1eb9n": Intent.ADD_SCHEDULE.value,
    "\u0111\u1eb7t bu\u1ed5i": Intent.ADD_SCHEDULE.value,
    # ADD_PRIVATE_NOTE
    "th\u00eam ghi ch\u00fa ri\u00eang t\u01b0": Intent.ADD_PRIVATE_NOTE.value,
    "th\u00eam ghi ch\u1ee7 ri\u00eang t\u1eeb": Intent.ADD_PRIVATE_NOTE.value,
    "th\u00eam ghi ch\u1ed7 ri\u00eang t\u01b0": Intent.ADD_PRIVATE_NOTE.value,
    "l\u01b0u ghi ch\u00fa b\u1ea3o m\u1eadt": Intent.ADD_PRIVATE_NOTE.value,
    "t\u1ea1o ghi ch\u00fa c\u00e1 nh\u00e2n": Intent.ADD_PRIVATE_NOTE.value,
    "ghi note ri\u00eang t\u01b0": Intent.ADD_PRIVATE_NOTE.value,
    # ADD_NOTE
    "th\u00eam ghi ch\u00fa": Intent.ADD_NOTE.value,
    "t\u1ea1o ghi ch\u00fa": Intent.ADD_NOTE.value,
    "l\u01b0u ghi ch\u00fa": Intent.ADD_NOTE.value,
    # VIEW_PRIVATE_NOTE
    "m\u1edf ghi ch\u00fa ri\u00eang t\u01b0 c\u1ee7a t\u00f4i": Intent.VIEW_PRIVATE_NOTE.value,
    "xem ghi ch\u00fa c\u00e1 nh\u00e2n": Intent.VIEW_PRIVATE_NOTE.value,
    "xem ghi ch\u00fa ri\u00eang t\u01b0": Intent.VIEW_PRIVATE_NOTE.value,
    "xem ghi ch\u00fa b\u1ea3o m\u1eadt": Intent.VIEW_PRIVATE_NOTE.value,
    "\u0111\u1ecdc ghi ch\u00fa ri\u00eang t\u01b0 c\u1ee7a t\u00f4i": Intent.VIEW_PRIVATE_NOTE.value,
    "m\u1edf note ri\u00eang t\u01b0": Intent.VIEW_PRIVATE_NOTE.value,
    "m\u1edf note b\u1ea3o m\u1eadt": Intent.VIEW_PRIVATE_NOTE.value,
    "hi\u1ec3n th\u1ecb ghi ch\u00fa ri\u00eang t\u01b0 c\u1ee7a t\u00f4i": Intent.VIEW_PRIVATE_NOTE.value,
    "huy\u1ec3n th\u1ecb ghi ch\u00fa ri\u00eang t\u01b0 m\u1edbi nh\u1ea5t": Intent.VIEW_PRIVATE_NOTE.value,
    "cho t\u00f4i xem ghi ch\u00fa c\u00e1 nh\u00e2n": Intent.VIEW_PRIVATE_NOTE.value,
    "mở vì chủ luyện từ trà tai": Intent.VIEW_PRIVATE_NOTE.value,
}

# ---------------------------------------------------------------------------
# Keyword anchors — moi intent yeu cau it nhat 1 tu khoa xuat hien
# trong query de duoc chap nhan. Giup tranh false positive khi cau truc
# giong nhung thieu tu khoa then chot.
# ---------------------------------------------------------------------------

KEYWORD_ANCHORS: dict[str, Sequence[str]] = {
    Intent.GET_TIME.value: [
        "gi\u1edf", "gi\u1ee3", "th\u1eddi gian", "m\u1ea5y", "m\u1ea3y",
        "hi\u1ec7n t\u1ea1i", "b\u00e2y gi\u1edf",
    ],
    Intent.VIEW_SCHEDULE.value: [
        "l\u1ecbch", "l\u1ecbc", "th\u1eddi kh\u00f3a bi\u1ec3u",
        "bu\u1ed5i", "b\u1ed1i", "b\u1ed3i", "m\u00f4n",
    ],
    Intent.ADD_SCHEDULE.value: [
        "th\u00eam", "t\u1ea1o", "l\u1eadp", "\u0111\u1eb7t",
        "nh\u1eafc", "l\u1ecbch", "l\u1ecbc", "bu\u1ed5i",
        "cu\u1ed9c h\u1eb9n",
    ],
    Intent.ADD_PRIVATE_NOTE.value: [
        "th\u00eam", "t\u1ea1o", "l\u01b0u", "ghi", "ghi ch\u00fa",
        "ghi ch\u1ee7", "ghi ch\u1ed7", "note", "ri\u00eang t\u01b0", "ri\u00eang t\u1eeb", "c\u00e1 nh\u00e2n",
        "b\u1ea3o m\u1eadt",
    ],
    Intent.ADD_NOTE.value: [
        "th\u00eam", "t\u1ea1o", "l\u01b0u", "ghi", "ghi ch\u00fa",
        "ghi ch\u1ee7", "ghi ch\u1ed7", "note",
    ],
    Intent.VIEW_PRIVATE_NOTE.value: [
        "ghi ch\u00fa", "ghi ch\u1ee7", "note", "n\u00f3t", "n\u00f4t",
        "vì chủ", "luyện từ", "trà tai",
        "ri\u00eang t\u01b0", "ring t\u01b0", "c\u00e1 nh\u00e2n",
        "c\u1ea3 nh\u00e2n", "b\u1ea3o m\u1eadt", "b\u00e1o m\u1eadt",
    ],
}

FUZZY_THRESHOLD: float = 70.0
FUZZY_MARGIN: float = 10.0


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _classify_regex(normalized: str) -> str:
    """Phan loai bang regex — logic goc, khong thay doi."""

    if not normalized:
        return Intent.OUT_OF_SCOPE.value

    has_note = any(marker in normalized for marker in ("ghi chú", "ghi chủ", "ghi chỗ", "note"))
    has_private_marker = any(
        marker in normalized
        for marker in ("riêng tư", "riêng từ", "cá nhân", "bảo mật", "của tôi")
    )
    if has_note and has_private_marker:
        if any(action in normalized for action in ("thêm", "tạo", "lưu")) or "ghi note" in normalized:
            return Intent.ADD_PRIVATE_NOTE.value
        if any(action in normalized for action in ("xem", "mở", "đọc", "hiển thị", "huyển thị")):
            return Intent.VIEW_PRIVATE_NOTE.value

    if not has_private_marker and has_note and any(
        action in normalized for action in ("thêm", "tạo", "lưu")
    ):
        return Intent.ADD_NOTE.value
    if "ảnh" in normalized and not has_note:
        return Intent.OUT_OF_SCOPE.value

    if (
        _NOTE_TERM.search(normalized)
        and _ADD_NOTE_PATTERN.search(normalized)
        and _PRIVATE_MARKER.search(normalized)
    ):
        return Intent.ADD_PRIVATE_NOTE.value

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


def classify_intent(text: str) -> str:
    """Phan loai mot cau vao dung mot trong nam intent.

    Rule cu the duoc uu tien truoc rule tong quat. Khi regex khong match,
    fuzzy matching duoc dung lam fallback de xu ly ASR output bi nhieu.
    Khi khong co bang chung ro rang, ham luon tra ``OUT_OF_SCOPE``
    thay vi doan mot chuc nang nhay cam.
    """

    normalized = normalize_text(text)
    result = _classify_regex(normalized)
    if result != Intent.OUT_OF_SCOPE.value:
        return result

    hit = fuzzy_match(
        normalized,
        FUZZY_CANDIDATES,
        threshold=FUZZY_THRESHOLD,
        margin=FUZZY_MARGIN,
        min_words=4,
        keywords=KEYWORD_ANCHORS,
    )
    if hit is not None:
        if hit["canonical"] == Intent.ADD_PRIVATE_NOTE.value and not any(
            marker in normalized for marker in ("riêng tư", "riêng từ", "cá nhân", "bảo mật", "của tôi")
        ):
            return Intent.OUT_OF_SCOPE.value
        if hit["canonical"] == Intent.ADD_NOTE.value and any(
            marker in normalized for marker in ("riêng tư", "riêng từ", "cá nhân", "bảo mật", "của tôi")
        ):
            return Intent.OUT_OF_SCOPE.value
        if hit["canonical"] == Intent.VIEW_PRIVATE_NOTE.value and "ảnh" in normalized:
            return Intent.OUT_OF_SCOPE.value
        return hit["canonical"]

    return Intent.OUT_OF_SCOPE.value
