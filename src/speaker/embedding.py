'''Frozen ECAPA-TDNN speaker embedding extraction.'''

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from src.utils.config import load_yaml_mapping, resolve_path


DEFAULT_CONFIG_PATH = Path('config.yaml')
DEFAULT_MODEL_SOURCE = 'speechbrain/spkrec-ecapa-voxceleb'
TARGET_SAMPLE_RATE = 16000
EXPECTED_EMBEDDING_DIMENSION = 192


class EmbeddingError(ValueError):
    '''Raised when input audio or an extracted embedding is invalid.'''


class CheckpointValidationError(EmbeddingError):
    '''Raised when a configured fine-tuned checkpoint cannot be trusted.'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_document(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CheckpointValidationError(f'ECAPA checkpoint does not exist: {path}')
    actual_sha256 = _sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise CheckpointValidationError(
            f'ECAPA checkpoint SHA-256 mismatch: expected {expected_sha256}, '
            f'received {actual_sha256}'
        )
    try:
        document = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise CheckpointValidationError(
            f'Unable to load ECAPA checkpoint: {path}'
        ) from error
    if not isinstance(document, Mapping):
        raise CheckpointValidationError('ECAPA checkpoint must contain a mapping')
    epoch = document.get('epoch')
    if not isinstance(epoch, int) or epoch <= 0:
        raise CheckpointValidationError('ECAPA checkpoint epoch is missing or invalid')
    return document


def _configured_checkpoint(
    speaker: Mapping[str, Any],
    config_root: Path,
) -> tuple[Path | None, str, str | None, bool, str]:
    path_value = speaker.get('checkpoint_path')
    checkpoint_path = (
        resolve_path(path_value, config_root) if path_value else None
    )
    return (
        checkpoint_path,
        str(speaker.get('checkpoint_encoder_key', 'encoder')),
        str(speaker['checkpoint_sha256']) if speaker.get('checkpoint_sha256') else None,
        bool(speaker.get('checkpoint_strict', True)),
        str(speaker.get('model_version', 'speechbrain-ecapa-voxceleb')),
    )


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
        checkpoint_path: str | Path | None = None,
        checkpoint_key: str = 'encoder',
        checkpoint_sha256: str | None = None,
        checkpoint_strict: bool = True,
        model_version: str = 'speechbrain-ecapa-voxceleb',
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.expected_dimension = expected_dimension
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.checkpoint_key = checkpoint_key
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_strict = checkpoint_strict
        self.model_version = model_version
        self.checkpoint_metadata: dict[str, Any] = {}
        self.classifier = classifier if classifier is not None else self._load_classifier()
        self._load_finetuned_checkpoint()
        self._freeze_for_inference()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        classifier: Any | None = None,
    ) -> ECAPAEmbeddingExtractor:
        config, config_root = load_yaml_mapping(config_path)
        speaker = config.get('speaker', {})
        if speaker.get('fine_tune', False):
            raise ValueError('speaker.fine_tune must remain false')
        if not speaker.get('freeze_parameters', True):
            raise ValueError('speaker.freeze_parameters must remain true')
        if not speaker.get('evaluation_mode', True):
            raise ValueError('speaker.evaluation_mode must remain true')

        cache_dir = Path(speaker.get('cache_dir', 'models/cache/ecapa'))
        cache_dir = resolve_path(cache_dir, config_root)
        checkpoint_path, checkpoint_key, checkpoint_sha256, checkpoint_strict, model_version = (
            _configured_checkpoint(speaker, config_root)
        )
        return cls(
            model_source=speaker.get('embedding_model', DEFAULT_MODEL_SOURCE),
            device=speaker.get('device', 'cpu'),
            cache_dir=cache_dir,
            expected_dimension=int(
                speaker.get('embedding_dimension', EXPECTED_EMBEDDING_DIMENSION)
            ),
            classifier=classifier,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_strict=checkpoint_strict,
            model_version=model_version,
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

    def _load_finetuned_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        if not self.checkpoint_sha256:
            raise CheckpointValidationError(
                'checkpoint_sha256 is required when checkpoint_path is configured'
            )
        document = _checkpoint_document(self.checkpoint_path, self.checkpoint_sha256)
        state_dict = document.get(self.checkpoint_key)
        if not isinstance(state_dict, Mapping):
            raise CheckpointValidationError(
                f'ECAPA checkpoint key is missing or invalid: {self.checkpoint_key}'
            )
        mods = getattr(self.classifier, 'mods', None)
        embedding_model = getattr(mods, 'embedding_model', None)
        if embedding_model is None:
            raise CheckpointValidationError(
                'SpeechBrain classifier has no mods.embedding_model'
            )
        try:
            incompatible = embedding_model.load_state_dict(
                state_dict,
                strict=self.checkpoint_strict,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CheckpointValidationError(
                'Fine-tuned encoder state is incompatible with SpeechBrain ECAPA'
            ) from error
        if self.checkpoint_strict and (
            incompatible.missing_keys or incompatible.unexpected_keys
        ):
            raise CheckpointValidationError(
                'Fine-tuned encoder state did not load strictly'
            )
        validation = document.get('validation', document.get('validation_metrics', {}))
        self.checkpoint_metadata = {
            'path': str(self.checkpoint_path),
            'sha256': self.checkpoint_sha256.lower(),
            'epoch': int(document['epoch']),
            'validation': dict(validation) if isinstance(validation, Mapping) else {},
        }

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

    engine = extractor or get_embedding_extractor(config_path)
    return engine.extract(audio, sample_rate=sample_rate)


@lru_cache(maxsize=4)
def _cached_embedding_extractor(config_path: str) -> ECAPAEmbeddingExtractor:
    return ECAPAEmbeddingExtractor.from_config(config_path)


def get_embedding_extractor(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ECAPAEmbeddingExtractor:
    '''Return one cached extractor for each resolved configuration path.'''

    return _cached_embedding_extractor(str(Path(config_path).resolve()))


def clear_embedding_extractor_cache() -> None:
    '''Clear cached runtime models for tests or explicit model reloads.'''

    _cached_embedding_extractor.cache_clear()
