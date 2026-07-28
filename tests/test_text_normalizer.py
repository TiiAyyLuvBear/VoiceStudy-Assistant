"""Unit test cho chuẩn hóa transcript tiếng Việt."""

import pytest

from src.nlu.text_normalizer import normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "  Thêm lịch Học Máy lúc 8h sáng mai! ",
            "thêm lịch học máy lúc 8 giờ sáng mai",
        ),
        ("Họp nhóm lúc 08:30.", "họp nhóm lúc 8 giờ 30"),
        ("Nhắc tôi lúc 8h05", "nhắc tôi lúc 8 giờ 05"),
        ("Học lúc 2 PM", "học lúc 2 giờ chiều"),
        ("Học lúc 8:00 a.m.", "học lúc 8 giờ sáng"),
        ("Lịch ngày 28/07/2026", "lịch ngày 28/07/2026"),
        ("Dữ liệu ngày 2026-07-28", "dữ liệu ngày 2026-07-28"),
        ("MỞ GHI CHÚ RIÊNG TƯ???", "mở ghi chú riêng tư"),
        ("", ""),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_text_keeps_vietnamese_diacritics() -> None:
    assert normalize_text("Đọc ghi chú cá nhân") == "đọc ghi chú cá nhân"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(None)  # type: ignore[arg-type]
