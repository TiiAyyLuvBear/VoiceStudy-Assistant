"""Extract ECAPA embeddings for every frozen speaker protocol row.

The command is resumable: a valid existing ``.npy`` file and matching metadata
row are reused, while each newly extracted row is checkpointed atomically.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor


PROTOCOL_FILES: tuple[str, ...] = (
    "svm_closed_set_enrollment.csv",
    "svm_closed_set_train.csv",
    "svm_closed_set_validation.csv",
    "svm_closed_set_test.csv",
    "cosine_validation_enrollment.csv",
    "cosine_validation_query.csv",
    "cosine_validation_unknown.csv",
    "cosine_test_enrollment.csv",
    "cosine_test_query.csv",
    "cosine_test_unknown.csv",
)

OUTPUT_FIELDS: tuple[str, ...] = (
    "audio_id",
    "speaker_id",
    "protocol",
    "split",
    "role",
    "embedding_path",
    "embedding_dim",
    "l2_norm",
    "latency_ms",
)

INPUT_FIELDS: tuple[str, ...] = (
    "audio_id",
    "audio_path",
    "normalized_speaker_id",
    "protocol",
    "split_name",
    "role",
)


class EmbeddingExtractor(Protocol):
    is_frozen: bool

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[np.ndarray, int, float]: ...


Preprocessor = Callable[[str], tuple[np.ndarray, int]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Protocol metadata does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in INPUT_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return list(reader)


def load_protocol_rows(
    metadata_dir: Path,
    protocol_files: Sequence[str] = PROTOCOL_FILES,
) -> list[dict[str, str]]:
    """Load and validate every protocol row in deterministic file order."""

    rows: list[dict[str, str]] = []
    seen_audio_ids: set[str] = set()
    for filename in protocol_files:
        path = metadata_dir / filename
        for line_number, row in enumerate(_read_csv(path), start=2):
            blank = [field for field in INPUT_FIELDS if not row.get(field, "").strip()]
            if blank:
                raise ValueError(f"{path}:{line_number} has blank fields: {blank}")
            audio_id = row["audio_id"].strip()
            if audio_id in seen_audio_ids:
                raise ValueError(f"Duplicate audio_id across protocols: {audio_id}")
            seen_audio_ids.add(audio_id)
            rows.append(row)
    return rows


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not component:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return component


def _embedding_path(embedding_dir: Path, row: dict[str, str]) -> Path:
    return (
        embedding_dir
        / _safe_component(row["protocol"].lower())
        / _safe_component(row["split_name"].lower())
        / f"{_safe_component(row['audio_id'])}.npy"
    )


def _write_embedding(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_previous(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OUTPUT_FIELDS:
            return {}
        return {row["audio_id"]: row for row in reader if row.get("audio_id")}


def _valid_cached_row(
    cached: dict[str, str] | None,
    *,
    row: dict[str, str],
    embedding_path: Path,
) -> bool:
    if cached is None or not embedding_path.is_file():
        return False
    expected = {
        "speaker_id": row["normalized_speaker_id"],
        "protocol": row["protocol"],
        "split": row["split_name"],
        "role": row["role"],
        "embedding_path": embedding_path.as_posix(),
    }
    if any(cached.get(key) != value for key, value in expected.items()):
        return False
    try:
        embedding = np.load(embedding_path, allow_pickle=False)
        dimension = int(cached["embedding_dim"])
        recorded_norm = float(cached["l2_norm"])
    except (OSError, ValueError, KeyError):
        return False
    actual_norm = float(np.linalg.norm(embedding))
    return (
        embedding.ndim == 1
        and embedding.size == dimension
        and np.isfinite(embedding).all()
        and abs(actual_norm - recorded_norm) <= 1e-5
        and abs(actual_norm - 1.0) <= 1e-5
    )


def extract_all_embeddings(
    *,
    metadata_dir: Path = Path("data/processed/v1/metadata"),
    audio_root: Path = Path("data/processed/v1/audio"),
    embedding_dir: Path = Path("data/embeddings"),
    output_path: Path = Path("data/metadata/embedding_metadata.csv"),
    config_path: Path = Path("config.yaml"),
    protocol_files: Sequence[str] = PROTOCOL_FILES,
    extractor: EmbeddingExtractor | None = None,
    preprocessor: Preprocessor = preprocess_audio,
    resume: bool = True,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Extract all protocol embeddings and return the complete metadata rows."""

    rows = load_protocol_rows(metadata_dir, protocol_files)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        rows = rows[:limit]

    previous = _read_previous(output_path) if resume else {}
    results: list[dict[str, str]] = []
    engine = extractor
    for index, row in enumerate(rows, start=1):
        target = _embedding_path(embedding_dir, row)
        cached = previous.get(row["audio_id"])
        if resume and _valid_cached_row(cached, row=row, embedding_path=target):
            results.append(cached)
            print(f"[{index}/{len(rows)}] resume {row['audio_id']}", flush=True)
            continue

        source = audio_root / Path(row["audio_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Audio does not exist: {source}")
        audio, sample_rate = preprocessor(str(source))
        if engine is None:
            engine = ECAPAEmbeddingExtractor.from_config(config_path)
            if not engine.is_frozen:
                raise RuntimeError("ECAPA model must be frozen and in evaluation mode")

        embedding, dimension, latency_ms = engine.extract(
            audio,
            sample_rate=sample_rate,
        )
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if dimension != embedding.size:
            raise ValueError(
                f"Embedding dimension mismatch for {row['audio_id']}: "
                f"reported {dimension}, actual {embedding.size}"
            )
        if not np.isfinite(embedding).all() or abs(norm - 1.0) > 1e-5:
            raise ValueError(f"Invalid L2-normalized embedding for {row['audio_id']}")

        _write_embedding(target, embedding)
        result = {
            "audio_id": row["audio_id"],
            "speaker_id": row["normalized_speaker_id"],
            "protocol": row["protocol"],
            "split": row["split_name"],
            "role": row["role"],
            "embedding_path": target.as_posix(),
            "embedding_dim": str(dimension),
            "l2_norm": f"{norm:.8f}",
            "latency_ms": f"{latency_ms:.3f}",
        }
        results.append(result)
        _write_metadata(output_path, results)
        print(
            f"[{index}/{len(rows)}] extracted {row['audio_id']} "
            f"dim={dimension} latency_ms={latency_ms:.3f}",
            flush=True,
        )

    _write_metadata(output_path, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/processed/v1/metadata"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/processed/v1/audio"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/embeddings"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/embedding_metadata.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    results = extract_all_embeddings(
        metadata_dir=args.metadata_dir,
        audio_root=args.audio_root,
        embedding_dir=args.embedding_dir,
        output_path=args.output,
        config_path=args.config,
        resume=args.resume,
        limit=args.limit,
    )
    print(f"Created {len(results)} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
