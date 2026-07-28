"""Unit test cho năm intent chính thức."""

import pytest

from src.nlu.intent_classifier import classify_intent
from src.nlu.intent_schema import Intent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bây giờ là mấy giờ?", Intent.GET_TIME.value),
        ("Cho tôi xem lịch ngày mai", Intent.VIEW_SCHEDULE.value),
        ("Thêm lịch học máy lúc 8h sáng mai", Intent.ADD_SCHEDULE.value),
        ("Mở ghi chú riêng tư của tôi", Intent.VIEW_PRIVATE_NOTE.value),
        ("Mở nhạc cho tôi", Intent.OUT_OF_SCOPE.value),
        ("Mở ghi chú", Intent.OUT_OF_SCOPE.value),
        ("Mấy giờ học thì tốt?", Intent.OUT_OF_SCOPE.value),
        ("Tôi thích học máy", Intent.OUT_OF_SCOPE.value),
        ("", Intent.OUT_OF_SCOPE.value),
    ],
)
def test_classify_intent(text: str, expected: str) -> None:
    assert classify_intent(text) == expected
