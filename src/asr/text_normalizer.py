"""ASR-specific text normalization for fair Vietnamese WER/CER scoring."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"(?<!\w)e\s*(?:-\s*)?mail(?!\w)", re.IGNORECASE)
_AM_PM_RE = re.compile(
    r"(?<!\w)([01]?\d|2[0-3])(?:\s*[:h]\s*([0-5]\d))?"
    r"\s*(a\.?\s*m\.?|p\.?\s*m\.?)(?!\w)",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"(?<![\w/.-])([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?!\d)",
    re.IGNORECASE,
)
_HOUR_RE = re.compile(
    r"(?<![\w/.-])([01]?\d|2[0-3])\s*h(?![\w:])", re.IGNORECASE
)
_HOUR_MINUTE_WORD_RE = re.compile(
    r"(?<!\w)([01]?\d|2[0-3])\s+gi\u1edd\s+([0-5]?\d)\s+ph\u00fat(?!\w)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?!\w)")

_DIGIT_WORDS = (
    "kh\u00f4ng",
    "m\u1ed9t",
    "hai",
    "ba",
    "b\u1ed1n",
    "n\u0103m",
    "s\u00e1u",
    "b\u1ea3y",
    "t\u00e1m",
    "ch\u00edn",
)
_GROUP_UNITS = ("", "ngh\u00ecn", "tri\u1ec7u", "t\u1ef7")


def _format_hour_minute(hour: str, minute: str | None = None) -> str:
    normalized_hour = str(int(hour))
    if minute is None or int(minute) == 0:
        return f"{normalized_hour} gi\u1edd"
    return f"{normalized_hour} gi\u1edd {int(minute):02d}"


def _replace_am_pm(match: re.Match[str]) -> str:
    period = re.sub(r"[.\s]", "", match.group(3)).lower()
    vietnamese_period = "s\u00e1ng" if period == "am" else "chi\u1ec1u"
    return f"{_format_hour_minute(match.group(1), match.group(2))} {vietnamese_period}"


def _read_under_hundred(value: int) -> str:
    if value < 10:
        return _DIGIT_WORDS[value]
    tens, ones = divmod(value, 10)
    words = ["m\u01b0\u1eddi"] if tens == 1 else [_DIGIT_WORDS[tens], "m\u01b0\u01a1i"]
    if ones == 0:
        return " ".join(words)
    if ones == 1 and tens > 1:
        words.append("m\u1ed1t")
    elif ones == 5:
        words.append("l\u0103m")
    else:
        words.append(_DIGIT_WORDS[ones])
    return " ".join(words)


def _read_under_thousand(value: int, *, force_hundreds: bool = False) -> str:
    hundreds, remainder = divmod(value, 100)
    words: list[str] = []
    if hundreds:
        words.extend((_DIGIT_WORDS[hundreds], "tr\u0103m"))
    elif force_hundreds and remainder:
        words.extend(("kh\u00f4ng", "tr\u0103m"))
    if remainder:
        if remainder < 10 and (hundreds or force_hundreds):
            words.append("l\u1ebb")
        words.append(_read_under_hundred(remainder))
    return " ".join(words)


def _integer_to_vietnamese(value: int) -> str:
    if value == 0:
        return _DIGIT_WORDS[0]
    if value >= 1_000_000_000_000:
        return " ".join(_DIGIT_WORDS[int(digit)] for digit in str(value))

    groups: list[int] = []
    remaining = value
    while remaining:
        remaining, group = divmod(remaining, 1000)
        groups.append(group)

    highest_group = len(groups) - 1
    words: list[str] = []
    for group_index in range(highest_group, -1, -1):
        group = groups[group_index]
        if group == 0:
            continue
        force_hundreds = group_index < highest_group and group < 100
        words.append(_read_under_thousand(group, force_hundreds=force_hundreds))
        if group_index:
            words.append(_GROUP_UNITS[group_index])
    return " ".join(words)


def _replace_number(match: re.Match[str]) -> str:
    return _integer_to_vietnamese(int(match.group(0)))


def _remove_unnecessary_punctuation(text: str) -> str:
    output: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        output.append(" " if category.startswith(("P", "S")) else char)
    return "".join(output)


def normalize_asr_text(text: str) -> str:
    """Normalize equivalent ASR spellings without hiding recognition errors."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = _EMAIL_RE.sub("email", normalized)
    normalized = _AM_PM_RE.sub(_replace_am_pm, normalized)
    normalized = _CLOCK_RE.sub(
        lambda match: _format_hour_minute(match.group(1), match.group(2)),
        normalized,
    )
    normalized = _HOUR_RE.sub(
        lambda match: _format_hour_minute(match.group(1)), normalized
    )
    normalized = _HOUR_MINUTE_WORD_RE.sub(
        lambda match: _format_hour_minute(match.group(1), match.group(2)),
        normalized,
    )
    normalized = _remove_unnecessary_punctuation(normalized)
    normalized = _NUMBER_RE.sub(_replace_number, normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()
