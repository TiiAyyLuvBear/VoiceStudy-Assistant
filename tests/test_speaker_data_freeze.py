from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_speaker_data import (
    CHECKSUM_ALGORITHM,
    DEFAULT_MANIFEST,
    FreezeVerificationError,
    canonical_csv_sha256,
    verify_manifest,
)


def test_canonical_checksum_ignores_bom_line_endings_and_csv_quoting(
    tmp_path: Path,
) -> None:
    windows_csv = tmp_path / 'windows.csv'
    unix_csv = tmp_path / 'unix.csv'
    windows_csv.write_bytes(
        b'\xef\xbb\xbfid,name\r\n1,Tu\xe1\xba\xa5n Anh\r\n'
    )
    unix_csv.write_bytes(
        b'id,name\n1,\x22Tu\xe1\xba\xa5n Anh\x22\n'
    )

    assert canonical_csv_sha256(windows_csv) == canonical_csv_sha256(unix_csv)


def test_canonical_checksum_normalizes_newlines_inside_quoted_fields(
    tmp_path: Path,
) -> None:
    windows_csv = tmp_path / 'windows_multiline.csv'
    unix_csv = tmp_path / 'unix_multiline.csv'
    windows_csv.write_bytes(
        b'id,transcript\r\n1,"first line\r\nsecond line"\r\n'
    )
    unix_csv.write_bytes(
        b'id,transcript\n1,"first line\nsecond line"\n'
    )

    assert canonical_csv_sha256(windows_csv) == canonical_csv_sha256(unix_csv)


def test_canonical_checksum_detects_logical_data_change(tmp_path: Path) -> None:
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'
    first.write_bytes(b'id,value\n1,A\n')
    second.write_bytes(b'id,value\n1,B\n')

    assert canonical_csv_sha256(first) != canonical_csv_sha256(second)


def test_current_speaker_v1_manifest_verifies() -> None:
    manifest = verify_manifest()

    assert manifest['dataset_version'] == 'v1'
    assert manifest['freeze_status'] == 'FROZEN'
    assert manifest['checksum_algorithm'] == CHECKSUM_ALGORITHM
    assert len(manifest['splits']) == 10


def test_verify_rejects_manifest_checksum_mismatch(tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding='utf-8'))
    first_split = next(iter(manifest['splits'].values()))
    first_split['checksum'] = '0' * 64
    tampered_manifest = tmp_path / 'split_manifest.json'
    tampered_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding='utf-8',
    )

    with pytest.raises(FreezeVerificationError, match='checksum'):
        verify_manifest(tampered_manifest)
