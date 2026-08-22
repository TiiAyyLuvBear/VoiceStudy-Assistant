"""Tests for required fields and database-write gates."""

from src.nlu.missing_fields import (
    can_execute_command,
    can_write_database,
    get_missing_fields,
)


def test_add_schedule_requires_title_date_and_time() -> None:
    entities = {"title": "hoc may"}
    missing = get_missing_fields("ADD_SCHEDULE", entities)

    assert missing == ["date", "time"]
    assert not can_execute_command("ADD_SCHEDULE", missing)
    assert not can_write_database("ADD_SCHEDULE", entities)


def test_complete_add_schedule_can_write_database() -> None:
    entities = {"title": "hoc may", "date": "2026-07-29", "time": "08:00"}

    assert get_missing_fields("ADD_SCHEDULE", entities) == []
    assert can_execute_command("ADD_SCHEDULE", [])
    assert can_write_database("ADD_SCHEDULE", entities)


def test_add_private_note_requires_content() -> None:
    assert get_missing_fields("ADD_PRIVATE_NOTE", {}) == ["content"]
    assert not can_write_database("ADD_PRIVATE_NOTE", {})
    assert get_missing_fields("ADD_PRIVATE_NOTE", {"content": "secret"}) == []
    assert can_write_database("ADD_PRIVATE_NOTE", {"content": "secret"})


def test_add_note_requires_content() -> None:
    assert get_missing_fields("ADD_NOTE", {}) == ["content"]
    assert not can_write_database("ADD_NOTE", {})
    assert get_missing_fields("ADD_NOTE", {"content": "milk"}) == []
    assert can_write_database("ADD_NOTE", {"content": "milk"})


def test_out_of_scope_never_executes() -> None:
    assert not can_execute_command("OUT_OF_SCOPE", [])
    assert not can_write_database("OUT_OF_SCOPE", {})
