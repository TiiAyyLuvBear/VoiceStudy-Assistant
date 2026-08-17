"""Dependency-light file integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return lowercase SHA-256 for one file without loading it all into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
