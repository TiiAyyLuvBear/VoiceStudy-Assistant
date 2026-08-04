'''Generate ECAPA embedding and cosine-similarity baseline CSV files.'''

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.speaker.embedding import ECAPAEmbeddingExtractor


MODEL_NAME = 'speechbrain/spkrec-ecapa-voxceleb'


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _select_samples(
    rows: list[dict[str, str]],
    *,
    speakers: int,
    samples_per_speaker: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row['normalized_speaker_id']].append(row)
    selected: list[dict[str, str]] = []
    for speaker_id in sorted(grouped)[:speakers]:
        available = grouped[speaker_id]
        if len(available) < samples_per_speaker:
            raise ValueError(
                f'{speaker_id} has {len(available)} samples; '
                f'{samples_per_speaker} required'
            )
        selected.extend(available[:samples_per_speaker])
    if len({row['normalized_speaker_id'] for row in selected}) < speakers:
        raise ValueError(f'Metadata does not contain {speakers} speakers')
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_baseline(
    metadata_path: Path,
    audio_root: Path,
    output_dir: Path,
    *,
    speakers: int = 2,
    samples_per_speaker: int = 2,
    config_path: Path = Path('config.yaml'),
) -> tuple[Path, Path]:
    selected = _select_samples(
        _read_rows(metadata_path),
        speakers=speakers,
        samples_per_speaker=samples_per_speaker,
    )
    prepared = []
    for row in selected:
        relative_path = Path(row['audio_path'])
        audio_path = audio_root / relative_path
        audio, sample_rate = preprocess_audio(str(audio_path))
        prepared.append((row, audio_path, audio, sample_rate))

    extractor = ECAPAEmbeddingExtractor.from_config(config_path)
    if not extractor.is_frozen:
        raise RuntimeError('ECAPA model must be frozen and in evaluation mode')

    results: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    speakers_by_audio: dict[str, str] = {}
    for row, audio_path, audio, sample_rate in prepared:
        embedding, dimension, latency_ms = extractor.extract(
            audio,
            sample_rate=sample_rate,
        )
        audio_id = row['audio_id']
        speaker_id = row['normalized_speaker_id']
        vectors[audio_id] = embedding
        speakers_by_audio[audio_id] = speaker_id
        results.append(
            {
                'audio_id': audio_id,
                'normalized_speaker_id': speaker_id,
                'audio_path': audio_path.as_posix(),
                'model': MODEL_NAME,
                'embedding_dimension': dimension,
                'l2_norm': f'{np.linalg.norm(embedding):.8f}',
                'has_nan': bool(np.isnan(embedding).any()),
                'latency_ms': f'{latency_ms:.3f}',
                'embedding': json.dumps(
                    embedding.round(8).tolist(),
                    separators=(',', ':'),
                ),
            }
        )

    embedding_path = output_dir / 'embedding_baseline.csv'
    _write_csv(embedding_path, results)

    similarities: list[dict[str, object]] = []
    for first_id, second_id in combinations(vectors, 2):
        same_speaker = speakers_by_audio[first_id] == speakers_by_audio[second_id]
        similarities.append(
            {
                'audio_id_1': first_id,
                'audio_id_2': second_id,
                'speaker_id_1': speakers_by_audio[first_id],
                'speaker_id_2': speakers_by_audio[second_id],
                'pair_type': 'same_speaker' if same_speaker else 'different_speaker',
                'cosine_similarity': f'{np.dot(vectors[first_id], vectors[second_id]):.8f}',
            }
        )
    similarity_path = output_dir / 'similarity_baseline.csv'
    _write_csv(similarity_path, similarities)
    return embedding_path, similarity_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path(
            'data/processed/v1/metadata/cosine_validation_enrollment.csv'
        ),
    )
    parser.add_argument(
        '--audio-root',
        type=Path,
        default=Path('data/processed/v1/audio'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/metadata'),
    )
    parser.add_argument('--speakers', type=int, default=2)
    parser.add_argument('--samples-per-speaker', type=int, default=2)
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    args = parser.parse_args()

    embedding_path, similarity_path = generate_baseline(
        args.metadata,
        args.audio_root,
        args.output_dir,
        speakers=args.speakers,
        samples_per_speaker=args.samples_per_speaker,
        config_path=args.config,
    )
    print(f'Created: {embedding_path}')
    print(f'Created: {similarity_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
