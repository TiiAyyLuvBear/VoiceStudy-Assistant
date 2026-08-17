"""Tests for shared file integrity helpers."""

import hashlib

import pytest

from src.utils.files import sha256_file


def test_sha256_file_streams_expected_digest(tmp_path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"voice-study" * 100)

    assert sha256_file(source, chunk_size=7) == hashlib.sha256(source.read_bytes()).hexdigest()


def test_sha256_file_rejects_invalid_chunk_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        sha256_file(tmp_path / "unused", chunk_size=0)
