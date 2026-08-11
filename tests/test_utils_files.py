from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils.files import canonical_csv_sha256, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    content = b"frozen speaker artifact\x00" * 100
    source.write_bytes(content)

    assert sha256_file(source) == hashlib.sha256(content).hexdigest()


def test_canonical_csv_sha256_normalizes_multiline_field_newlines(
    tmp_path: Path,
) -> None:
    windows_csv = tmp_path / "windows.csv"
    unix_csv = tmp_path / "unix.csv"
    windows_csv.write_bytes(b'id,text\r\n1,"first\r\nsecond"\r\n')
    unix_csv.write_bytes(b'id,text\n1,"first\nsecond"\n')

    assert canonical_csv_sha256(windows_csv) == canonical_csv_sha256(unix_csv)
