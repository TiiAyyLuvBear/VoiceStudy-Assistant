from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.audio.source import resolve_audio_path


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "huggingface_audio.json"
    path.write_text(
        json.dumps(
            {
                "repo_id": "owner/audio",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "cache_dir": str(tmp_path / "cache"),
                "path_prefixes": ["data/audio/dev/"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolver_prefers_existing_local_file(tmp_path: Path) -> None:
    audio = tmp_path / "local.wav"
    audio.write_bytes(b"audio")

    assert resolve_audio_path(audio) == audio.resolve()


def test_resolver_downloads_configured_project_audio(tmp_path: Path) -> None:
    config = _config(tmp_path)
    downloaded = tmp_path / "downloaded.wav"
    downloaded.write_bytes(b"audio")
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(downloaded)

    result = resolve_audio_path(
        "data/audio/dev/missing.wav",
        config_path=config,
        downloader=fake_download,
    )

    assert result == downloaded.resolve()
    assert calls == [
        {
            "repo_id": "owner/audio",
            "filename": "data/audio/dev/missing.wav",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]


def test_resolver_does_not_download_unconfigured_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_audio_path(
            "unrelated/missing.wav",
            config_path=config,
            downloader=lambda **kwargs: pytest.fail(str(kwargs)),
        )
