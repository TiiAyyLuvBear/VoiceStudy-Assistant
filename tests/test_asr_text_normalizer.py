"""Tests for ASR-only Vietnamese text normalization."""

import pytest

from src.asr.text_normalizer import normalize_asr_text


@pytest.mark.parametrize("raw", ["email", "e-mail", "e mail", "E-MAIL"])
def test_normalizes_email_variants(raw: str) -> None:
    assert normalize_asr_text(raw) == "email"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", "kh\u00f4ng"),
        ("12", "m\u01b0\u1eddi hai"),
        ("27", "hai m\u01b0\u01a1i b\u1ea3y"),
        ("2017", "hai ngh\u00ecn kh\u00f4ng tr\u0103m m\u01b0\u1eddi b\u1ea3y"),
        ("8h30", "t\u00e1m gi\u1edd ba m\u01b0\u01a1i"),
        ("8:00 a.m.", "t\u00e1m gi\u1edd s\u00e1ng"),
    ],
)
def test_normalizes_digits_to_vietnamese_words(raw: str, expected: str) -> None:
    assert normalize_asr_text(raw) == expected


def test_does_not_hide_real_recognition_errors() -> None:
    assert normalize_asr_text("m\u1edf \u0111\u00e8n") != normalize_asr_text("m\u1ee1 \u0111\u1ec1")


def test_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_asr_text(None)  # type: ignore[arg-type]
