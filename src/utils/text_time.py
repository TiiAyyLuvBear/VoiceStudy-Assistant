"""Dependency-light helpers for Vietnamese clock-text normalization."""

from __future__ import annotations

import re


def format_hour_minute(hour: str, minute: str | None = None) -> str:
    """Render numeric clock parts as Vietnamese hour text."""

    normalized_hour = str(int(hour))
    if minute is None or int(minute) == 0:
        return f"{normalized_hour} giờ"
    return f"{normalized_hour} giờ {int(minute):02d}"


def replace_am_pm(match: re.Match[str]) -> str:
    """Render an AM/PM regex match as Vietnamese clock text."""

    period = re.sub(r"[.\s]", "", match.group(3)).lower()
    vietnamese_period = "sáng" if period == "am" else "chiều"
    return f"{format_hour_minute(match.group(1), match.group(2))} {vietnamese_period}"
