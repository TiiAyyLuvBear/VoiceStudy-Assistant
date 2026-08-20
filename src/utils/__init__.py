"""Shared, dependency-light utilities for every application module."""

from src.utils.config import (
    load_yaml_mapping,
    resolve_path,
    threshold_from_metrics_document,
)
from src.utils.files import canonical_csv_sha256, sha256_file
from src.utils.fuzzy_match import fuzzy_match, FuzzyResult, normalize_for_matching

__all__ = [
    "FuzzyResult",
    "canonical_csv_sha256",
    "fuzzy_match",
    "load_yaml_mapping",
    "normalize_for_matching",
    "resolve_path",
    "threshold_from_metrics_document",
    "sha256_file",
    "canonical_csv_sha256",
]