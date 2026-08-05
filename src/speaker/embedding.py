'''Frozen ECAPA-TDNN speaker embedding extraction.'''

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml


DEFAULT_CONFIG_PATH = Path('config.yaml')
DEFAULT_MODEL_SOURCE = 'speechbrain/spkrec-ecapa-voxceleb'
TARGET_SAMPLE_RATE = 16000
EXPECTED_EMBEDDING_DIMENSION = 192


class EmbeddingError(ValueError):
    '''Raised when input audio or an extracted embedding is invalid.'''


def _validate_preprocessed_audio(
    audio: np.ndarray | torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise EmbeddingError(
            f'ECAPA expects {TARGET_SAMPLE_RATE} Hz preprocessed audio; '
            f'received {sample_rate} Hz'
        )
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    if waveform.ndim != 1:
        raise EmbeddingError(
            'ECAPA input must be a mono 1-D waveform; preprocess stereo audio first'
        )
    if waveform.numel() == 0:
        raise EmbeddingError('ECAPA input audio is empty')
    if not torch.isfinite(waveform).all():
        raise EmbeddingError('ECAPA input audio contains NaN or infinity')
    if torch.max(torch.abs(waveform)).item() == 0.0:
        raise EmbeddingError('ECAPA input audio is silent')
    return waveform


class ECAPAEmbeddingExtractor:
    '''Inference-only wrapper for SpeechBrain pretrained ECAPA-TDNN.'''

    def __init__(
        self,
        *,
        model_source: str = DEFAULT_MODEL_SOURCE,
        device: str = 'cpu',
        cache_dir: str | Path = 'models/cache/ecapa',
        expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
        classifier: Any | None = None,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.expected_dimension = expected_dimension
        self.classifier = classifier if classifier is not None else self._load_classifier()
        self._freeze_for_inference()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        classifier: Any | None = None,
    ) -> ECAPAEmbeddingExtractor:
        path = Path(config_path)
        with path.open('r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
        speaker = config.get('speaker', {})
        if speaker.get('fine_tune', False):
            raise ValueError('speaker.fine_tune must remain false')
        if not speaker.get('freeze_parameters', True):
            raise ValueError('speaker.freeze_parameters must remain true')
        if not speaker.get('evaluation_mode', True):
            raise ValueError('speaker.evaluation_mode must remain true')

        cache_dir = Path(speaker.get('cache_dir', 'models/cache/ecapa'))
        if not cache_dir.is_absolute():
            cache_dir = path.resolve().parent / cache_dir
        return cls(
            model_source=speaker.get('embedding_model', DEFAULT_MODEL_SOURCE),
            device=speaker.get('device', 'cpu'),
            cache_dir=cache_dir,
            expected_dimension=int(
                speaker.get('embedding_dimension', EXPECTED_EMBEDDING_DIMENSION)
            ),
            classifier=classifier,
        )

    def _load_classifier(self) -> Any:
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:
            raise RuntimeError(
                'SpeechBrain is required; run: pip install -r requirements.txt'
            ) from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=str(self.cache_dir),
            run_opts={'device': self.device},
            freeze_params=True,
            local_strategy=LocalStrategy.COPY,
        )

    def _freeze_for_inference(self) -> None:
        self.classifier.eval()
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)

    @property
    def is_frozen(self) -> bool:
        return not self.classifier.training and all(
            not parameter.requires_grad for parameter in self.classifier.parameters()
        )

    def extract(
        self,
        audio: np.ndarray | torch.Tensor,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> tuple[np.ndarray, int, float]:
        '''Return L2-normalized embedding, dimension, and latency in ms.'''

        waveform = _validate_preprocessed_audio(audio, sample_rate)
        started_at = time.perf_counter()
        with torch.inference_mode():
            encoded = self.classifier.encode_batch(
                waveform.unsqueeze(0),
                normalize=False,
            )
            embedding = torch.as_tensor(encoded).detach().float().reshape(-1)
            if not torch.isfinite(embedding).all():
                raise EmbeddingError('ECAPA embedding contains NaN or infinity')
            norm = torch.linalg.vector_norm(embedding)
            if not torch.isfinite(norm) or norm.item() <= torch.finfo(torch.float32).eps:
                raise EmbeddingError('ECAPA returned a zero-norm embedding')
            embedding = functional.normalize(embedding, p=2, dim=0)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        dimension = int(embedding.numel())
        if self.expected_dimension and dimension != self.expected_dimension:
            raise EmbeddingError(
                f'Expected {self.expected_dimension}-D; received {dimension}-D'
            )
        return embedding.cpu().numpy(), dimension, latency_ms


def extract_embedding(
    audio: np.ndarray | torch.Tensor,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    extractor: ECAPAEmbeddingExtractor | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[np.ndarray, int, float]:
    '''Extract one frozen, L2-normalized speaker embedding.'''

    engine = extractor or ECAPAEmbeddingExtractor.from_config(config_path)
    return engine.extract(audio, sample_rate=sample_rate)
