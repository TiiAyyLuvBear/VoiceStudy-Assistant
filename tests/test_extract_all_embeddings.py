"""Tests for resumable full-protocol embedding extraction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.extract_all_embeddings import OUTPUT_FIELDS, extract_all_embeddings


class _FakeExtractor:
    is_frozen = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[np.ndarray, int, float]:
        self.calls += 1
        assert audio.size
        assert sample_rate == 16000
        embedding = np.array([0.6, 0.8], dtype=np.float32)
        return embedding, 2, 12.5


def _write_protocol(path: Path, audio_id: str, audio_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "audio_id",
        "audio_path",
        "normalized_speaker_id",
        "protocol",
        "split_name",
        "role",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "audio_id": audio_id,
                "audio_path": audio_path,
                "normalized_speaker_id": "speaker_001",
                "protocol": "COSINE_VALIDATION",
                "split_name": "cosine_validation_query",
                "role": "ENROLLED",
            }
        )


def test_extracts_exact_metadata_schema_and_resumes(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    audio_root = tmp_path / "audio"
    audio_path = audio_root / "dev" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    waveform = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)
    sf.write(audio_path, waveform, 16000)
    protocol_name = "protocol.csv"
    _write_protocol(metadata_dir / protocol_name, "audio-1", "dev/sample.wav")

    output = tmp_path / "embedding_metadata.csv"
    embedding_dir = tmp_path / "embeddings"
    extractor = _FakeExtractor()
    first = extract_all_embeddings(
        metadata_dir=metadata_dir,
        audio_root=audio_root,
        embedding_dir=embedding_dir,
        output_path=output,
        protocol_files=(protocol_name,),
        extractor=extractor,
    )

    assert extractor.calls == 1
    assert tuple(first[0]) == OUTPUT_FIELDS
    assert first[0]["speaker_id"] == "speaker_001"
    assert first[0]["embedding_dim"] == "2"
    assert float(first[0]["l2_norm"]) == 1.0
    saved = Path(first[0]["embedding_path"])
    assert np.allclose(np.load(saved, allow_pickle=False), [0.6, 0.8])

    second = extract_all_embeddings(
        metadata_dir=metadata_dir,
        audio_root=audio_root,
        embedding_dir=embedding_dir,
        output_path=output,
        protocol_files=(protocol_name,),
        extractor=extractor,
    )
    assert second == first
    assert extractor.calls == 1
