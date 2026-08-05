from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.audio.preprocessing import preprocess_audio


def test_preprocessing_converts_stereo_8khz_to_mono_16khz(tmp_path) -> None:
    sample_rate = 8000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = 0.4 * np.sin(2 * np.pi * 220 * time_axis)
    right = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
    stereo = np.column_stack([left, right])
    audio_path = tmp_path / 'stereo_8khz.wav'
    sf.write(audio_path, stereo, sample_rate)

    audio, output_rate = preprocess_audio(str(audio_path))

    assert output_rate == 16000
    assert audio.ndim == 1
    assert 15900 <= audio.size <= 16100
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) == pytest.approx(0.99, abs=1e-3)


def test_preprocessing_rejects_missing_audio(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, RuntimeError)):
        preprocess_audio(str(tmp_path / 'missing.wav'))


def test_preprocessing_rejects_corrupt_audio(tmp_path) -> None:
    audio_path = tmp_path / 'broken.wav'
    audio_path.write_bytes(b'not a wav file')

    with pytest.raises((RuntimeError, ValueError)):
        preprocess_audio(str(audio_path))
