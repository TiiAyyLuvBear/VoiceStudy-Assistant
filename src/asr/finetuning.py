"""Training-only helpers for local Whisper fine-tuning.

This module is intentionally ASR-specific. Production inference remains in
``src.asr.whisper_model`` and does not acquire training dependencies.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


def load_finetune_rows(path: str | Path) -> list[dict[str, str]]:
    """Load and validate an ASR fine-tune CSV."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ASR fine-tune CSV not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = {"audio_id", "audio_path", "transcript"} - fields
        if missing:
            raise ValueError(
                f"{source} is missing required fields: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"ASR fine-tune CSV is empty: {source}")
    for row in rows:
        if not row["audio_id"].strip() or not row["transcript"].strip():
            raise ValueError(f"Blank audio_id/transcript in {source}")
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(f"Audio file not found: {row['audio_path']}")
    return rows


def epoch_rows(
    rows: Sequence[dict[str, str]], *, seed: int, epoch: int
) -> list[dict[str, str]]:
    """Return a deterministic per-epoch shuffle without mutating source rows."""

    shuffled = list(rows)
    random.Random(seed + epoch).shuffle(shuffled)
    return shuffled


def audio_bucket_seconds(
    sample_count: int,
    sample_rate: int,
    *,
    maximum_seconds: int = 30,
) -> int:
    """Choose an integer-second Whisper window that covers an audio sample."""

    if sample_count < 1 or sample_rate < 1:
        raise ValueError("sample_count and sample_rate must be positive")
    if maximum_seconds < 1 or maximum_seconds > 30:
        raise ValueError("maximum_seconds must be between 1 and 30")
    return min(maximum_seconds, max(1, math.ceil(sample_count / sample_rate)))


def batched(rows: Sequence[dict[str, str]], batch_size: int) -> Iterator[list[dict[str, str]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for index in range(0, len(rows), batch_size):
        yield list(rows[index : index + batch_size])


class WhisperBatchBuilder:
    """Load/resample audio and temporarily shorten Whisper's encoder window.

    Whisper's reference implementation normally pads every clip to 30 seconds.
    The v3 clips are much shorter. For local CPU training we crop the frozen
    positional embedding to an integer-second bucket, then restore the full
    30-second embedding before exporting the model.
    """

    def __init__(
        self,
        processor: Any,
        model: Any,
        *,
        sample_rate: int = 16_000,
        maximum_seconds: int = 16,
        device: Any = "cpu",
    ) -> None:
        import torch

        self.processor = processor
        self.model = model
        self.sample_rate = sample_rate
        self.maximum_seconds = maximum_seconds
        self.device = torch.device(device)
        self.encoder = model.get_encoder()
        self._full_positions = self.encoder.embed_positions.weight.detach().clone()
        self._full_position_count = self.encoder.embed_positions.num_embeddings
        self._embedding_cache: dict[int, Any] = {}
        self._torch = torch

    def _set_window(self, seconds: int) -> None:
        torch = self._torch
        # Whisper produces 100 mel frames/s and downsamples them by two.
        position_count = seconds * 50
        if self.encoder.embed_positions.num_embeddings == position_count:
            self.encoder.config.max_source_positions = position_count
            return
        embedding = self._embedding_cache.get(position_count)
        if embedding is None:
            source_weight = self._full_positions[:position_count]
            embedding = torch.nn.Embedding(
                position_count,
                source_weight.shape[1],
                device=source_weight.device,
                dtype=source_weight.dtype,
            )
            with torch.no_grad():
                embedding.weight.copy_(source_weight)
            embedding.weight.requires_grad_(False)
            self._embedding_cache[position_count] = embedding
        self.encoder.embed_positions = embedding
        self.encoder.config.max_source_positions = position_count

    def restore_full_window(self) -> None:
        """Restore the original 30-second positional embedding before export."""

        torch = self._torch
        weight = self._full_positions
        embedding = torch.nn.Embedding(
            self._full_position_count,
            weight.shape[1],
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            embedding.weight.copy_(weight)
        embedding.weight.requires_grad_(False)
        self.encoder.embed_positions = embedding
        self.encoder.config.max_source_positions = self._full_position_count

    def __call__(self, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
        import librosa
        import soundfile as sf
        import torch

        audio_arrays: list[np.ndarray] = []
        transcripts: list[str] = []
        maximum_bucket = 1
        for row in rows:
            audio, source_rate = sf.read(row["audio_path"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if source_rate != self.sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=source_rate,
                    target_sr=self.sample_rate,
                )
            audio = np.asarray(audio, dtype=np.float32)
            bucket = audio_bucket_seconds(
                len(audio),
                self.sample_rate,
                maximum_seconds=self.maximum_seconds,
            )
            maximum_bucket = max(maximum_bucket, bucket)
            audio_arrays.append(audio)
            transcripts.append(row["transcript"])

        self._set_window(maximum_bucket)
        features = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding="max_length",
            max_length=maximum_bucket * self.sample_rate,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokenized = self.processor.tokenizer(
            transcripts,
            padding=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        decoder_start = self.model.config.decoder_start_token_id
        if labels.shape[1] and torch.all(labels[:, 0] == decoder_start):
            labels = labels[:, 1:]
        return {
            "input_features": features.input_features.to(self.device),
            "attention_mask": features.attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }
