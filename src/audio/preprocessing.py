from dataclasses import dataclass
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.audio.source import resolve_audio_path

TARGET_SAMPLE_RATE = 16000
TOP_DB = 30
TARGET_PEAK = 0.99
VAD_PEAK_RATIO = 0.03
MIN_SPEECH_SEGMENT_MS = 80


@dataclass(frozen=True)
class SpeakerPreprocessResult:
    audio: np.ndarray
    sample_rate: int
    metrics: dict

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.

    Args:
        audio: Audio waveform.

    Returns:
        Mono waveform.
    """

    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1)


def _count_channels(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])

def resample_audio(
    audio: np.ndarray,
    sample_rate: int
) -> np.ndarray:
    """
    Resample audio.

    Args:
        audio: Audio waveform.
        sample_rate: Original sample rate.

    Returns:
        Audio at TARGET_SAMPLE_RATE.
    """

    if sample_rate == TARGET_SAMPLE_RATE:
        return audio

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    common_divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
    return resample_poly(
        audio,
        up=TARGET_SAMPLE_RATE // common_divisor,
        down=sample_rate // common_divisor,
    )

def trim_silence(audio: np.ndarray) -> np.ndarray:
    """
    Remove leading and trailing silence.

    Args:
        audio: Audio waveform.

    Returns:
        Trimmed waveform.
    """

    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    threshold = peak * (10.0 ** (-TOP_DB / 20.0))
    audible = np.flatnonzero(np.abs(audio) >= threshold)
    if audible.size == 0:
        return audio[:0]
    return audio[audible[0]:audible[-1] + 1]


def speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    peak_ratio: float = VAD_PEAK_RATIO,
    min_segment_ms: int = MIN_SPEECH_SEGMENT_MS,
) -> list[tuple[int, int]]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or sample_rate <= 0:
        return []
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        return []
    threshold = peak * float(peak_ratio)
    frame_size = max(1, int(sample_rate * 0.03))
    hop_size = max(1, int(sample_rate * 0.01))
    mask = np.zeros(values.size, dtype=bool)
    for start in range(0, values.size, hop_size):
        end = min(values.size, start + frame_size)
        if float(np.max(np.abs(values[start:end]))) >= threshold:
            mask[start:end] = True
        if end == values.size:
            break
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    min_samples = max(1, int(sample_rate * min_segment_ms / 1000))
    segments: list[tuple[int, int]] = []
    start = int(indexes[0])
    previous = start
    for value in indexes[1:]:
        current = int(value)
        if current != previous + 1:
            if previous + 1 - start >= min_samples:
                segments.append((start, previous + 1))
            start = current
        previous = current
    if previous + 1 - start >= min_samples:
        segments.append((start, previous + 1))
    return segments


def extract_speech(audio: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not segments:
        return values[:0]
    return np.concatenate([values[start:end] for start, end in segments]).astype(np.float32)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize peak amplitude.

    Args:
        audio: Audio waveform.

    Returns:
        Normalized waveform.
    """

    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    return audio / peak * TARGET_PEAK


def preprocess_audio_with_metrics(
    input_path: str | Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    vad_peak_ratio: float = VAD_PEAK_RATIO,
) -> SpeakerPreprocessResult:
    source_path = resolve_audio_path(input_path)
    raw_audio, source_sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
    num_channels = _count_channels(raw_audio)
    mono = convert_to_mono(raw_audio).astype(np.float32)
    if target_sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError("Only 16000 Hz speaker preprocessing is currently supported")
    resampled = resample_audio(mono, source_sample_rate).astype(np.float32)
    segments = speech_segments(
        resampled,
        TARGET_SAMPLE_RATE,
        peak_ratio=vad_peak_ratio,
    )
    speech = extract_speech(resampled, segments)
    duration = float(resampled.size / TARGET_SAMPLE_RATE) if TARGET_SAMPLE_RATE > 0 else 0.0
    speech_duration = float(speech.size / TARGET_SAMPLE_RATE) if TARGET_SAMPLE_RATE > 0 else 0.0
    clipping_ratio = float(np.mean(np.abs(resampled) >= 0.99)) if resampled.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(speech)))) if speech.size else 0.0
    normalized = normalize_audio(speech)
    return SpeakerPreprocessResult(
        audio=normalized.astype(np.float32),
        sample_rate=TARGET_SAMPLE_RATE,
        metrics={
            "duration_seconds": duration,
            "speech_duration_seconds": speech_duration,
            "speech_ratio": float(speech_duration / duration) if duration > 0.0 else 0.0,
            "rms": rms,
            "clipping_ratio": clipping_ratio,
            "estimated_snr_db": None,
            "sample_rate": TARGET_SAMPLE_RATE,
            "source_sample_rate": int(source_sample_rate),
            "num_channels": num_channels,
            "speech_segments": [
                {"start_sample": start, "end_sample": end}
                for start, end in segments
            ],
        },
    )

def preprocess_audio(input_path: str):
    """
    Preprocess one audio file.

    Args:
        input_path: WAV path.

    Returns:
        audio,
        sample_rate
    """
    result = preprocess_audio_with_metrics(input_path)
    return result.audio, result.sample_rate

def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str
):
    """
    Save processed audio.

    Args:
        audio: Waveform.
        sample_rate: Sample rate.
        output_path: Output path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        audio,
        sample_rate
    )

def process_file(
    input_path: str,
    output_path: str
):
    """
    Process one file.

    Args:
        input_path: Input WAV.
        output_path: Output WAV.
    """

    audio, sr = preprocess_audio(
        input_path
    )

    save_audio(
        audio,
        sr,
        output_path
    )

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    input_file = (
        project_root
        / "data"
        / "audio"
        / "train-115"
        / "4dcf89cc7708ffe6339d97afd4da24f5.wav"
    )
    output_file = (
        project_root
        / "data"
        / "processed"
        / "v1"
        / "audio"
        / "train-115"
        / "4dcf89cc7708ffe6339d97afd4da24f5.wav"
    )

    process_file(
        str(input_file),
        str(output_file)
    )
    print("Done")
