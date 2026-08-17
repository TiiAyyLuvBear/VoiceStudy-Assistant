from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16000
TOP_DB = 30
TARGET_PEAK = 0.99

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

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize peak amplitude.

    Args:
        audio: Audio waveform.

    Returns:
        Normalized waveform.
    """

    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    return audio / peak * TARGET_PEAK

def preprocess_audio(input_path: str):
    """
    Preprocess one audio file.

    Args:
        input_path: WAV path.

    Returns:
        audio,
        sample_rate
    """
    audio, sr = sf.read(input_path)
    audio = convert_to_mono(audio)
    audio = resample_audio(audio, sr)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)

    return audio, TARGET_SAMPLE_RATE

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
