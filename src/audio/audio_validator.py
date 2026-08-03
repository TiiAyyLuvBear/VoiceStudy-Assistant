import os
import numpy as np
import soundfile as sf

MIN_DURATION = 0.5
MAX_DURATION = 30.0
EXPECTED_SAMPLE_RATE = 48000
EXPECTED_CHANNELS = 1
LOW_VOLUME_THRESHOLD = 0.005
SILENCE_THRESHOLD = 0.001
LONG_SILENCE_THRESHOLD = 2.0


def compute_rms(audio):
    """Compute Root Mean Square (RMS) of an audio signal.

    Args:
        audio (np.ndarray): Audio waveform.

    Returns:
        float: RMS value.
    """
    return np.sqrt(np.mean(audio.astype(np.float64) ** 2))


def longest_silence_duration(audio, sample_rate):
    """Estimate the longest continuous silent segment.

    Args:
        audio (np.ndarray): Audio waveform.
        sample_rate (int): Sampling rate.

    Returns:
        float: Longest silence duration in seconds.
    """
    silent = np.abs(audio) < SILENCE_THRESHOLD

    longest = 0
    current = 0

    for value in silent:
        if value:
            current += 1
        else:
            longest = max(longest, current)
            current = 0

    longest = max(longest, current)

    return longest / sample_rate


def check_audio(file_path):
    """Validate a single audio file.

    Checks:
        - File exists
        - Readable
        - Duration
        - Sample rate
        - Number of channels
        - Empty signal
        - Low volume
        - Long silence

    Args:
        file_path (str): Path to audio file.

    Returns:
        dict: Validation results.
    """

    result = {
        "exists": False,
        "readable": False,
        "duration": np.nan,
        "sample_rate": np.nan,
        "channels": np.nan,
        "is_empty": True,
        "low_volume": False,
        "long_silence": False,
        "reason": ""
    }

    # Check file existence
    if not os.path.exists(file_path):
        result["reason"] = "file_not_found"
        return result

    result["exists"] = True

    # Read metadata only
    try:
        info = sf.info(file_path)
    except Exception:
        result["reason"] = "cannot_read"
        return result

    result["readable"] = True
    result["duration"] = info.duration
    result["sample_rate"] = info.samplerate
    result["channels"] = info.channels

    # Validate metadata
    if info.duration < MIN_DURATION:
        result["reason"] = "too_short"
        return result

    if info.duration > MAX_DURATION:
        result["reason"] = "too_long"
        return result

    if info.samplerate != EXPECTED_SAMPLE_RATE:
        result["reason"] = "wrong_sample_rate"
        return result

    if info.channels != EXPECTED_CHANNELS:
        result["reason"] = "wrong_channels"
        return result

    # Read waveform only if metadata is valid
    try:
        audio, _ = sf.read(file_path, dtype="float32")
    except Exception:
        result["reason"] = "cannot_read"
        return result

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Check empty signal
    if len(audio) == 0 or np.all(audio == 0):
        result["reason"] = "empty_signal"
        return result

    result["is_empty"] = False

    # Check signal energy
    rms = compute_rms(audio)

    if rms < LOW_VOLUME_THRESHOLD:
        result["low_volume"] = True
        result["reason"] = "low_volume"
        return result

    # Check long silence
    silence = longest_silence_duration(audio, info.samplerate)

    if silence > LONG_SILENCE_THRESHOLD:
        result["long_silence"] = True
        result["reason"] = "long_silence"
        return result

    result["reason"] = "valid"

    return result


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]

    example = (
        project_root
        / "data"
        / "audio"
        / "train-115"
        / "4dcf89cc7708ffe6339d97afd4da24f5.wav"
    )

    print(check_audio(str(example)))

# if __name__ == "__main__":
#     example = "../../data/audio/train-115/4dcf89cc7708ffe6339d97afd4da24f5.wav"
#     print(check_audio(example))