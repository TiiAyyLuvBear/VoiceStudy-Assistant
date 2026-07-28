"""Unit test WER/CER không cần tải Whisper."""

import pytest

from src.asr.metrics import (
    calculate_corpus_error_rates,
    calculate_error_rates,
    edit_distance,
)


def test_edit_distance() -> None:
    assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert edit_distance([], ["a", "b"]) == 2


def test_identical_vietnamese_after_normalization() -> None:
    result = calculate_error_rates(
        "Thêm lịch học máy lúc 8h sáng mai!",
        "thêm lịch học máy lúc 8 giờ sáng mai",
    )
    assert result["wer"] == 0
    assert result["cer"] == 0


def test_word_substitution_rate() -> None:
    result = calculate_error_rates("tôi học máy", "tôi học toán")
    assert result["word_edits"] == 1
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["cer"] > 0


def test_corpus_rates_are_micro_averaged() -> None:
    result = calculate_corpus_error_rates(
        ["xin chào", "tôi học máy"],
        ["xin chào", "tôi học toán"],
    )
    assert result["word_edits"] == 1
    assert result["reference_words"] == 5
    assert result["wer"] == pytest.approx(0.2)


def test_corpus_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        calculate_corpus_error_rates(["a"], [])
