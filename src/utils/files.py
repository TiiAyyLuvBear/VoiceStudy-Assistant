"""Dependency-light file integrity helpers."""

from __future__ import annotations

<<<<<<< HEAD
import hashlib
=======
import csv
import hashlib
import json
>>>>>>> 960772de09ee6970afefe0e394d43f9dc52624ec
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
<<<<<<< HEAD
    """Return lowercase SHA-256 for one file without loading it all into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
=======
    """Return the lowercase SHA-256 digest of a file without loading it whole."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_csv_sha256(path: str | Path) -> str:
    """Hash logical CSV rows independently of BOM, quoting, and newlines."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            normalized_row = [
                value.replace("\r\n", "\n").replace("\r", "\n")
                for value in row
            ]
            canonical_row = json.dumps(
                normalized_row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest.update(canonical_row.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()
>>>>>>> 960772de09ee6970afefe0e394d43f9dc52624ec
