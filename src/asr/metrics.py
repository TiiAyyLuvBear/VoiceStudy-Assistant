"""WER/CER metrics cho đánh giá ASR tiếng Việt."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from src.asr.text_normalizer import normalize_asr_text


class ErrorCounts(TypedDict):
    word_edits: int
    reference_words: int
    char_edits: int
    reference_chars: int


class ASRMetrics(ErrorCounts):
    wer: float
    cer: float


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance dùng bộ nhớ O(len(hypothesis))."""

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_value in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_value != hyp_value)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Tính edit counts sau khi normalize giống nhau cho reference/hypothesis."""

    normalized_reference = normalize_asr_text(reference)
    normalized_hypothesis = normalize_asr_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    reference_chars = list(normalized_reference.replace(" ", ""))
    hypothesis_chars = list(normalized_hypothesis.replace(" ", ""))
    return {
        "word_edits": edit_distance(reference_words, hypothesis_words),
        "reference_words": len(reference_words),
        "char_edits": edit_distance(reference_chars, hypothesis_chars),
        "reference_chars": len(reference_chars),
    }


def _rate(edits: int, reference_units: int) -> float:
    if reference_units:
        return edits / reference_units
    return 0.0 if edits == 0 else float(edits)


def calculate_error_rates(reference: str, hypothesis: str) -> ASRMetrics:
    counts = error_counts(reference, hypothesis)
    return {
        **counts,
        "wer": _rate(counts["word_edits"], counts["reference_words"]),
        "cer": _rate(counts["char_edits"], counts["reference_chars"]),
    }


def calculate_corpus_error_rates(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> ASRMetrics:
    """Corpus WER/CER bằng tổng edit distance chia tổng reference units."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    aggregate: ErrorCounts = {
        "word_edits": 0,
        "reference_words": 0,
        "char_edits": 0,
        "reference_chars": 0,
    }
    for reference, hypothesis in zip(references, hypotheses):
        counts = error_counts(reference, hypothesis)
        for key in aggregate:
            aggregate[key] += counts[key]
    return {
        **aggregate,
        "wer": _rate(aggregate["word_edits"], aggregate["reference_words"]),
        "cer": _rate(aggregate["char_edits"], aggregate["reference_chars"]),
    }
