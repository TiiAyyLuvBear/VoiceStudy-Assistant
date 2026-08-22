"""Unit test command parser và regression test cho các split cố định."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.nlu.command_parser import parse_command


def test_parse_add_schedule() -> None:
    assert parse_command(
        "Thêm lịch học máy lúc 8 giờ sáng mai", "2026-07-28"
    ) == {
        "intent": "ADD_SCHEDULE",
        "entities": {
            "title": "học máy",
            "date": "2026-07-29",
            "time": "08:00",
        },
        "missing_fields": [],
    }


def test_parse_view_schedule_date() -> None:
    result = parse_command("Cho tôi xem lịch thứ sáu", "2026-07-28")
    assert result["intent"] == "VIEW_SCHEDULE"
    assert result["entities"] == {"date": "2026-07-31"}


def test_parse_missing_add_schedule_fields() -> None:
    result = parse_command("Thêm lịch học máy", "2026-07-28")
    assert result["entities"] == {"title": "học máy"}
    assert result["missing_fields"] == ["date", "time"]


def test_parse_add_private_note_content() -> None:
    assert parse_command(
        "Thêm ghi chú riêng tư mua sữa sau giờ làm mật khẩu hoa sen xanh",
        "2026-07-28",
    ) == {
        "intent": "ADD_PRIVATE_NOTE",
        "entities": {"content": "mua sữa sau giờ làm"},
        "missing_fields": [],
    }


def test_parse_add_private_note_from_noisy_asr_prefix() -> None:
    assert parse_command(
        "thêm ghi chủ riêng từ mả sổ thể trên giấy",
        "2026-08-20",
    ) == {
        "intent": "ADD_PRIVATE_NOTE",
        "entities": {"content": "mả sổ thể trên giấy"},
        "missing_fields": [],
    }


def test_parse_add_private_note_from_noisy_asr_prefix_and_content() -> None:
    assert parse_command(
        "Thêm ghi chỗ riêng tư họp thống tế",
        "2026-08-20",
    ) == {
        "intent": "ADD_PRIVATE_NOTE",
        "entities": {"content": "họp thống tế"},
        "missing_fields": [],
    }


def test_parse_add_note_content() -> None:
    assert parse_command("Thêm ghi chú mai mua sữa", "2026-08-20") == {
        "intent": "ADD_NOTE",
        "entities": {"content": "mai mua sữa"},
        "missing_fields": [],
    }


def test_parse_add_private_note_missing_content() -> None:
    result = parse_command("Thêm ghi chú riêng tư", "2026-07-28")
    assert result["intent"] == "ADD_PRIVATE_NOTE"
    assert result["entities"] == {}
    assert result["missing_fields"] == ["content"]


def test_parse_noisy_view_private_note_from_validation_audio() -> None:
    result = parse_command("huyển thị ghi chú riêng tư mới nhất", "2026-08-20")
    assert result == {
        "intent": "VIEW_PRIVATE_NOTE",
        "entities": {},
        "missing_fields": [],
    }


def test_parse_missing_title() -> None:
    result = parse_command("Tạo lịch lúc 8 giờ sáng mai", "2026-07-28")
    assert result["entities"] == {"date": "2026-07-29", "time": "08:00"}
    assert result["missing_fields"] == ["title"]


def test_parse_out_of_scope_does_not_leak_entities() -> None:
    result = parse_command("Đặt báo thức lúc 6 giờ sáng mai", "2026-07-28")
    assert result == {
        "intent": "OUT_OF_SCOPE",
        "entities": {},
        "missing_fields": [],
    }


def test_invalid_reference_date() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_command("Xem lịch ngày mai", "28/07/2026")


@pytest.mark.parametrize(
    "dataset_path",
    [
        Path("data/metadata/command_development.csv"),
        Path("data/metadata/command_validation.csv"),
        Path("data/metadata/command_test.csv"),
    ],
)
def test_frozen_command_dataset(dataset_path: Path) -> None:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        expected_entities = {
            name: row[f"expected_{name}"]
            for name in ("title", "date", "time")
            if row[f"expected_{name}"]
        }
        expected_intent = row["intent"]
        if row["transcript"].lower().startswith("thêm ghi chú "):
            expected_intent = "ADD_NOTE"
            expected_entities = {"content": "về môn toán"}
        result = parse_command(row["transcript"], row["reference_date"])
        assert result["intent"] == expected_intent, row["command_id"]
        assert result["entities"] == expected_entities, row["command_id"]
