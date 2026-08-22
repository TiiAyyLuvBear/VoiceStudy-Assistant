"""Shared, dependency-light utilities for every application module."""

from src.utils.config import (
    load_yaml_mapping,
    resolve_path,
    threshold_from_metrics_document,
)
from src.utils.files import canonical_csv_sha256, sha256_file
from src.utils.fuzzy_match import fuzzy_match, FuzzyResult, normalize_for_matching
from src.utils.text_time import format_hour_minute, replace_am_pm

__all__ = [
    "FuzzyResult",
    "canonical_csv_sha256",
    "format_hour_minute",
    "fuzzy_match",
    "load_yaml_mapping",
    "normalize_for_matching",
    "replace_am_pm",
    "resolve_path",
    "threshold_from_metrics_document",
    "sha256_file",
]
