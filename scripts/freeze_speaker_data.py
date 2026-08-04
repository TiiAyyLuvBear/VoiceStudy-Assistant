'''Create or verify the frozen speaker dataset manifest.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET_VERSION = 'v1'
RANDOM_SEED = 42
PREPROCESSING_VERSION = 'v1'
CHECKSUM_ALGORITHM = 'sha256-canonical-csv-v1'
DEFAULT_METADATA_DIR = Path('data/processed/v1/metadata')
DEFAULT_MANIFEST = Path('data/processed/v1/split_manifest.json')
SPLIT_FILES = (
    'svm_closed_set_enrollment.csv',
    'svm_closed_set_train.csv',
    'svm_closed_set_validation.csv',
    'svm_closed_set_test.csv',
    'cosine_validation_enrollment.csv',
    'cosine_validation_query.csv',
    'cosine_validation_unknown.csv',
    'cosine_test_enrollment.csv',
    'cosine_test_query.csv',
    'cosine_test_unknown.csv',
)


class FreezeVerificationError(RuntimeError):
    '''Raised when frozen speaker data no longer matches its manifest.'''


def canonical_csv_sha256(path: Path) -> str:
    '''Hash logical CSV rows independent of BOM, quoting, and line endings.'''

    digest = hashlib.sha256()
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream):
            canonical_row = json.dumps(
                row,
                ensure_ascii=False,
                separators=(',', ':'),
            )
            digest.update(canonical_row.encode('utf-8'))
            digest.update(b'\n')
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen split: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Frozen split is empty: {path}')
    return rows


def _speaker_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row['normalized_speaker_id'] for row in rows)


def validate_split_invariants(
    metadata_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    '''Validate counts, uniqueness, and speaker isolation before hashing.'''

    splits = {
        name: _read_csv(metadata_dir / name)
        for name in SPLIT_FILES
    }
    expected_svm = {
        'svm_closed_set_enrollment.csv': 5,
        'svm_closed_set_train.csv': 10,
        'svm_closed_set_validation.csv': 5,
        'svm_closed_set_test.csv': 5,
    }
    for name, expected_count in expected_svm.items():
        counts = _speaker_counts(splits[name])
        if len(counts) != 10 or set(counts.values()) != {expected_count}:
            raise FreezeVerificationError(
                f'{name} must contain 10 speakers x {expected_count} audio'
            )

    for name in (
        'cosine_validation_enrollment.csv',
        'cosine_validation_query.csv',
        'cosine_test_enrollment.csv',
        'cosine_test_query.csv',
    ):
        counts = _speaker_counts(splits[name])
        if len(counts) != 2 or set(counts.values()) != {5}:
            raise FreezeVerificationError(f'{name} must contain 2 speakers x 5 audio')

    for name in (
        'cosine_validation_unknown.csv',
        'cosine_test_unknown.csv',
    ):
        if len(_speaker_counts(splits[name])) != 2:
            raise FreezeVerificationError(f'{name} must contain exactly 2 speakers')

    all_rows = [
        (name, row)
        for name, rows in splits.items()
        for row in rows
    ]
    for field in ('audio_path', 'checksum'):
        values = [row[field] for _, row in all_rows]
        duplicates = len(values) - len(set(values))
        if duplicates:
            raise FreezeVerificationError(
                f'Found {duplicates} duplicate {field} values across frozen splits'
            )

    svm = set().union(
        *(
            set(_speaker_counts(splits[name]))
            for name in expected_svm
        )
    )
    validation_enrolled = set(_speaker_counts(
        splits['cosine_validation_enrollment.csv']
    ))
    validation_unknown = set(_speaker_counts(
        splits['cosine_validation_unknown.csv']
    ))
    test_enrolled = set(_speaker_counts(
        splits['cosine_test_enrollment.csv']
    ))
    test_unknown = set(_speaker_counts(
        splits['cosine_test_unknown.csv']
    ))
    validation = validation_enrolled | validation_unknown
    test = test_enrolled | test_unknown
    overlaps = {
        'svm_validation': svm & validation,
        'svm_test': svm & test,
        'validation_test': validation & test,
        'validation_enrolled_unknown': validation_enrolled & validation_unknown,
        'test_enrolled_unknown': test_enrolled & test_unknown,
    }
    failed = {name: sorted(values) for name, values in overlaps.items() if values}
    if failed:
        raise FreezeVerificationError(f'Speaker leakage detected: {failed}')
    return splits


def build_manifest(
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    splits = validate_split_invariants(metadata_dir)
    split_summary = {}
    for name, rows in splits.items():
        split_summary[name] = {
            'num_audio': len(rows),
            'num_speaker': len(_speaker_counts(rows)),
            'checksum': canonical_csv_sha256(metadata_dir / name),
        }
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    return {
        'manifest_schema_version': 2,
        'dataset_version': DATASET_VERSION,
        'created_at': created_at or now,
        'verified_at': now,
        'random_seed': RANDOM_SEED,
        'preprocessing_version': PREPROCESSING_VERSION,
        'freeze_status': 'FROZEN',
        'checksum_algorithm': CHECKSUM_ALGORITHM,
        'rule': [
            'Do not change speaker assignment',
            'Do not change split assignment',
            'Create v2 for any data modification',
        ],
        'splits': split_summary,
    }


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing freeze manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_top_level = {
        'dataset_version': DATASET_VERSION,
        'random_seed': RANDOM_SEED,
        'preprocessing_version': PREPROCESSING_VERSION,
        'freeze_status': 'FROZEN',
        'checksum_algorithm': CHECKSUM_ALGORITHM,
    }
    errors = [
        f'{field}: expected {expected!r}, found {manifest.get(field)!r}'
        for field, expected in expected_top_level.items()
        if manifest.get(field) != expected
    ]
    recorded_splits = manifest.get('splits', {})
    if set(recorded_splits) != set(SPLIT_FILES):
        errors.append(
            f'split files differ: expected {sorted(SPLIT_FILES)}, '
            f'found {sorted(recorded_splits)}'
        )

    current = validate_split_invariants(metadata_dir)
    for name in SPLIT_FILES:
        recorded = recorded_splits.get(name, {})
        rows = current[name]
        actual = {
            'num_audio': len(rows),
            'num_speaker': len(_speaker_counts(rows)),
            'checksum': canonical_csv_sha256(metadata_dir / name),
        }
        for field, value in actual.items():
            if recorded.get(field) != value:
                errors.append(
                    f'{name}.{field}: expected {recorded.get(field)!r}, '
                    f'found {value!r}'
                )
    if errors:
        details = '\n - '.join(errors)
        raise FreezeVerificationError(f'Frozen speaker data verification failed:\n - {details}')
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--metadata-dir', type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--confirm-version')
    args = parser.parse_args()

    if args.refresh:
        if args.confirm_version != DATASET_VERSION:
            parser.error(
                f'--refresh requires --confirm-version {DATASET_VERSION}; '
                'create v2 instead when speaker assignments or splits changed'
            )
        created_at = None
        if args.manifest.is_file():
            previous = json.loads(args.manifest.read_text(encoding='utf-8'))
            if previous.get('dataset_version') != DATASET_VERSION:
                raise FreezeVerificationError(
                    'Refusing to overwrite a different dataset version'
                )
            created_at = previous.get('created_at')
        manifest = build_manifest(args.metadata_dir, created_at=created_at)
        _write_manifest(args.manifest, manifest)
        verify_manifest(args.manifest, args.metadata_dir)
        print(f'Refreshed and verified frozen speaker data: {args.manifest}')
        return 0

    manifest = verify_manifest(args.manifest, args.metadata_dir)
    version = manifest['dataset_version']
    split_count = len(manifest['splits'])
    print(f'Frozen speaker data verified: {version} ({split_count} splits)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
