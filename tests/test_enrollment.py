"""Unit tests for application enrollment requirements."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.speaker import enrollment


class _FakeExtractor:
    def extract(self, audio, *, sample_rate):
        vector = np.asarray([0.6, 0.8], dtype=np.float32)
        return vector, 2, 1.0


def _five_audio(tmp_path: Path) -> list[str]:
    paths = []
    for index in range(5):
        path = tmp_path / f"enroll-{index}.wav"
        path.write_bytes(b"audio")
        paths.append(str(path))
    return paths


def test_enrollment_writes_centroid_from_exactly_five_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(enrollment, "APPLICATION_EMBEDDING_DIR", tmp_path / "users")
    monkeypatch.setattr(
        enrollment,
        "preprocess_audio",
        lambda path: (np.ones(20, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(enrollment, "_get_extractor", lambda: _FakeExtractor())

    result = enrollment.enroll_user("user_003", _five_audio(tmp_path))

    assert result["success"] is True
    assert result["audio_count"] == 5
    assert result["embedding_dim"] == 2
    assert np.isclose(result["l2_norm"], 1.0)
    assert Path(result["embedding_path"]).is_file()


def test_enrollment_rejects_wrong_count_duplicate_and_missing_audio(
    tmp_path: Path,
) -> None:
    paths = _five_audio(tmp_path)
    assert enrollment.enroll_user("user_003", paths[:4])["success"] is False
    assert enrollment.enroll_user(
        "user_003", paths[:4] + [paths[0]]
    )["success"] is False
    paths[-1] = str(tmp_path / "missing.wav")
    result = enrollment.enroll_user("user_003", paths)
    assert result["success"] is False
    assert "does not exist" in (result["error"] or "")
