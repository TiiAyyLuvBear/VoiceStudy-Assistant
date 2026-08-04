from __future__ import annotations

import numpy as np
import pytest
import torch

from src.speaker.embedding import (
    ECAPAEmbeddingExtractor,
    EmbeddingError,
    extract_embedding,
)


class FakeEncoderClassifier(torch.nn.Module):
    def __init__(self, *, return_nan: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.return_nan = return_nan

    def encode_batch(self, waveform, normalize=False):
        assert waveform.shape[0] == 1
        assert normalize is False
        values = torch.arange(1, 193, dtype=torch.float32) * self.scale
        if self.return_nan:
            values[0] = torch.nan
        return values.reshape(1, 1, 192)


def _speech() -> np.ndarray:
    time_axis = np.arange(16000, dtype=np.float32) / 16000
    return 0.5 * np.sin(2 * np.pi * 220 * time_axis)


def test_ecapa_is_eval_frozen_and_returns_l2_embedding() -> None:
    classifier = FakeEncoderClassifier()
    extractor = ECAPAEmbeddingExtractor(classifier=classifier)

    embedding, dimension, latency_ms = extract_embedding(
        _speech(),
        extractor=extractor,
    )

    assert extractor.is_frozen
    assert classifier.training is False
    assert all(not parameter.requires_grad for parameter in classifier.parameters())
    assert dimension == 192
    assert embedding.shape == (192,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(embedding).all()
    assert latency_ms >= 0.0


@pytest.mark.parametrize(
    ('audio', 'sample_rate'),
    [
        (np.empty(0, dtype=np.float32), 16000),
        (np.zeros(16000, dtype=np.float32), 16000),
        (np.ones((16000, 2), dtype=np.float32), 16000),
        (np.ones(8000, dtype=np.float32), 8000),
        (np.array([0.1, np.nan], dtype=np.float32), 16000),
    ],
)
def test_embedding_rejects_non_preprocessed_audio(audio, sample_rate) -> None:
    extractor = ECAPAEmbeddingExtractor(classifier=FakeEncoderClassifier())

    with pytest.raises(EmbeddingError):
        extractor.extract(audio, sample_rate=sample_rate)


def test_embedding_rejects_nan_model_output() -> None:
    extractor = ECAPAEmbeddingExtractor(
        classifier=FakeEncoderClassifier(return_nan=True)
    )

    with pytest.raises(EmbeddingError, match='NaN'):
        extractor.extract(_speech())


def test_config_keeps_ecapa_frozen() -> None:
    extractor = ECAPAEmbeddingExtractor.from_config(
        'config.yaml',
        classifier=FakeEncoderClassifier(),
    )

    assert extractor.model_source == 'speechbrain/spkrec-ecapa-voxceleb'
    assert extractor.expected_dimension == 192
    assert extractor.is_frozen
