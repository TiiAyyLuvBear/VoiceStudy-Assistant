"""Whisper ASR backend hỗ trợ faster-whisper và Transformers/PhoWhisper."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, TypedDict

from src.utils.config import load_yaml_mapping, resolve_path
from src.utils.speechbrain_lazy import remove_speechbrain_lazy_modules, restore_modules


class ASRResult(TypedDict):
    """Hợp đồng output ASR dùng bởi Orchestrator."""

    transcript: str
    model: str
    language: str
    latency_ms: float
    success: bool
    error: str | None


class _Segment(Protocol):
    text: str


class _WhisperBackend(Protocol):
    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, Any]: ...


@dataclass(frozen=True)
class WhisperConfig:
    """Cấu hình inference cần thiết, được đọc từ ``config.yaml``."""

    backend: str = "faster-whisper"
    model_size: str = "small"
    model_name: str = "whisper-small"
    model_path: Path | None = None
    language: str = "vi"
    task: str = "transcribe"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    word_timestamps: bool = False
    download_root: Path | None = None
    local_files_only: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "WhisperConfig":
        backend = str(values.get("backend", "faster-whisper"))
        if backend not in {"faster-whisper", "transformers"}:
            raise ValueError(f"Unsupported ASR backend: {backend}")

        model_value = values.get("model_path")
        model_path = None
        if model_value:
            model_path = resolve_path(str(model_value), base_dir).resolve()

        download_value = values.get("download_root")
        download_root = None
        if download_value:
            download_root = resolve_path(str(download_value), base_dir).resolve()

        config = cls(
            backend=backend,
            model_size=str(values.get("model_size", "small")),
            model_name=str(values.get("model_name", "whisper-small")),
            model_path=model_path,
            language=str(values.get("language", "vi")),
            task=str(values.get("task", "transcribe")),
            device=str(values.get("device", "cpu")),
            compute_type=str(values.get("compute_type", "int8")),
            cpu_threads=int(values.get("cpu_threads", 0)),
            num_workers=int(values.get("num_workers", 1)),
            beam_size=int(values.get("beam_size", 5)),
            vad_filter=bool(values.get("vad_filter", True)),
            condition_on_previous_text=bool(
                values.get("condition_on_previous_text", False)
            ),
            word_timestamps=bool(values.get("word_timestamps", False)),
            download_root=download_root,
            local_files_only=bool(values.get("local_files_only", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.backend not in {"faster-whisper", "transformers"}:
            raise ValueError(f"Unsupported ASR backend: {self.backend}")
        if self.model_size != "small":
            raise ValueError("ASR must use the small model class in this profile")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("ASR device must be one of: cpu, cuda, auto")
        if self.language != "vi":
            raise ValueError("ASR language must be Vietnamese ('vi')")
        if self.task != "transcribe":
            raise ValueError("ASR task must be 'transcribe', not translation")
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        if self.model_path is not None and self.backend == "faster-whisper":
            if not self.model_path.is_dir():
                raise ValueError(
                    f"ASR model_path must be an existing directory: {self.model_path}"
                )
            required_files = ("config.json", "model.bin", "tokenizer.json")
            missing = [
                filename
                for filename in required_files
                if not (self.model_path / filename).is_file()
            ]
            if missing:
                raise ValueError(
                    "ASR model_path is missing required CTranslate2 files: "
                    + ", ".join(missing)
                )

    @property
    def model_source(self) -> str:
        """Nguồn model thật truyền cho backend ASR."""

        if self.model_path is not None:
            return str(self.model_path)
        if self.backend == "transformers":
            return self.model_name
        return self.model_size


class TransformersWhisperBackend:
    """Transformers ASR adapter compatible with the faster-whisper boundary."""

    def __init__(self, config: WhisperConfig) -> None:
        torch, AutoModelForSpeechSeq2Seq, AutoProcessor = import_transformers_asr()

        self.config = config
        self._torch = torch
        self._sample_rate = 16000
        self.device = self._resolve_device(config.device)
        kwargs: dict[str, Any] = {"local_files_only": config.local_files_only}
        if config.download_root is not None:
            config.download_root.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(config.download_root)
        self.processor = AutoProcessor.from_pretrained(config.model_source, **kwargs)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.model_source,
            **kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str):
        torch = self._torch
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ASR device cuda requested but CUDA is not available")
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _read_audio(self, audio: str):
        try:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "soundfile, scipy, and numpy are required for Transformers ASR"
            ) from exc

        waveform, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sample_rate != self._sample_rate:
            waveform = resample_poly(
                waveform,
                self._sample_rate,
                int(sample_rate),
            ).astype(np.float32)
        return waveform

    def _forced_decoder_ids(self, language: str, task: str):
        getter = getattr(self.processor, "get_decoder_prompt_ids", None)
        if getter is None:
            return None
        try:
            return getter(language=language, task=task)
        except Exception:
            return None

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, Any]:
        waveform = self._read_audio(audio)
        inputs = self.processor(
            waveform,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        generate_kwargs: dict[str, Any] = {
            "num_beams": int(kwargs.get("beam_size", self.config.beam_size)),
        }
        forced_decoder_ids = self._forced_decoder_ids(
            str(kwargs.get("language", self.config.language)),
            str(kwargs.get("task", self.config.task)),
        )
        if forced_decoder_ids is not None:
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
        with self._torch.inference_mode():
            generated_ids = self.model.generate(input_features, **generate_kwargs)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return (iter([SimpleNamespace(text=text)]), object())


def import_transformers_asr() -> tuple[Any, Any, Any]:
    """Import Transformers ASR dependencies before SpeechBrain lazy modules."""

    removed_lazy_modules = remove_speechbrain_lazy_modules()
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers and torch are required for PhoWhisper; "
            "run: pip install -r requirements.txt"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to import Transformers/PyTorch for PhoWhisper: {exc}"
        ) from exc
    finally:
        restore_modules(removed_lazy_modules)
    return torch, AutoModelForSpeechSeq2Seq, AutoProcessor


def load_whisper_config(config_path: str | Path = "config.yaml") -> WhisperConfig:
    """Đọc và kiểm tra mục ``asr`` trong YAML config."""

    document, base_dir = load_yaml_mapping(config_path)

    asr_values = document.get("asr")
    if not isinstance(asr_values, Mapping):
        raise ValueError("config.yaml must contain an 'asr' mapping")
    return WhisperConfig.from_mapping(asr_values, base_dir=base_dir)


class WhisperASR:
    """Lazy-loaded, thread-safe Whisper Small inference service."""

    def __init__(
        self,
        config: WhisperConfig,
        *,
        model: _WhisperBackend | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._model = model
        self._model_lock = threading.Lock()

    def _create_model(self) -> _WhisperBackend:
        if self.config.backend == "transformers":
            return TransformersWhisperBackend(self.config)

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed; run: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "cpu_threads": self.config.cpu_threads,
            "num_workers": self.config.num_workers,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.download_root is not None:
            self.config.download_root.mkdir(parents=True, exist_ok=True)
            kwargs["download_root"] = str(self.config.download_root)

        return WhisperModel(self.config.model_source, **kwargs)

    def load_model(self) -> _WhisperBackend:
        """Nạp model đúng một lần và tái sử dụng cho các request tiếp theo."""

        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._create_model()
        return self._model

    def _result(
        self,
        *,
        transcript: str,
        started_at: float,
        success: bool,
        error: str | None,
    ) -> ASRResult:
        return {
            "transcript": transcript,
            "model": self.config.model_name,
            "language": self.config.language,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "success": success,
            "error": error,
        }

    def transcribe(self, audio_path: str | Path) -> ASRResult:
        """Chuyển một file audio tiếng Việt thành transcript.

        Lỗi file và lỗi inference được chuyển thành output ``success=false`` để
        Orchestrator có thể dừng pipeline an toàn.
        """

        started_at = time.perf_counter()
        path = Path(audio_path).expanduser()

        if not path.exists():
            return self._result(
                transcript="",
                started_at=started_at,
                success=False,
                error=f"Audio file does not exist: {path}",
            )
        if not path.is_file():
            return self._result(
                transcript="",
                started_at=started_at,
                success=False,
                error=f"Audio path is not a file: {path}",
            )

        try:
            model = self.load_model()
            segments, _ = model.transcribe(
                str(path.resolve()),
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                condition_on_previous_text=self.config.condition_on_previous_text,
                word_timestamps=self.config.word_timestamps,
            )
            # faster-whisper bắt đầu inference khi generator segments được duyệt.
            transcript = " ".join(
                segment.text.strip()
                for segment in segments
                if getattr(segment, "text", "").strip()
            )
            transcript = " ".join(transcript.split())
            if not transcript:
                return self._result(
                    transcript="",
                    started_at=started_at,
                    success=False,
                    error="Whisper did not detect speech in the audio file",
                )
            return self._result(
                transcript=transcript,
                started_at=started_at,
                success=True,
                error=None,
            )
        except Exception as exc:  # Inference boundary for the application layer.
            return self._result(
                transcript="",
                started_at=started_at,
                success=False,
                error=f"ASR inference failed: {exc}",
            )


@lru_cache(maxsize=4)
def get_asr_model(config_path: str | Path = "config.yaml") -> WhisperASR:
    """Lấy ASR service đã cache theo đường dẫn config."""

    normalized_path = str(Path(config_path).expanduser().resolve())
    return WhisperASR(load_whisper_config(normalized_path))


def transcribe_audio(
    audio_path: str | Path,
    config_path: str | Path = "config.yaml",
) -> ASRResult:
    """Convenience function dùng trực tiếp bởi Streamlit/Orchestrator."""

    return get_asr_model(config_path).transcribe(audio_path)


def _main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Transcribe Vietnamese audio")
    parser.add_argument("audio_path", help="Path to WAV/MP3/FLAC audio")
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    args = parser.parse_args()

    result = transcribe_audio(args.audio_path, args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
