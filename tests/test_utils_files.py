from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils.files import sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    content = b"frozen speaker artifact\x00" * 100
    source.write_bytes(content)

    assert sha256_file(source) == hashlib.sha256(content).hexdigest()
