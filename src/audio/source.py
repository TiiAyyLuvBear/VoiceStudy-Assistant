"""Resolve project audio locally or from the frozen Hugging Face dataset."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "huggingface_audio.json"
DownloadFile = Callable[..., str]


@lru_cache(maxsize=4)
def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face audio config does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    required = ("repo_id", "repo_type", "revision", "path_prefixes")
    missing = [key for key in required if not document.get(key)]
    if missing:
        raise ValueError(
            "Hugging Face audio config is missing: " + ", ".join(missing)
        )
    if document["repo_type"] != "dataset":
        raise ValueError("Hugging Face audio repo_type must be 'dataset'")
    return document


def _repo_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.resolve(strict=False).relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise FileNotFoundError(f"Audio file does not exist: {path}") from exc
    if ".." in path.parts:
        raise FileNotFoundError(f"Audio path escapes the project root: {path}")
    return path.as_posix().lstrip("./")


def resolve_audio_path(
    value: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    downloader: DownloadFile | None = None,
) -> Path:
    """Return a local audio path, downloading a configured Hub asset if absent."""

    path = Path(value).expanduser()
    local_candidates = [path] if path.is_absolute() else [path, PROJECT_ROOT / path]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.resolve()

    repo_path = _repo_path(value)
    config_file = Path(config_path).expanduser().resolve()
    config = _load_config(str(config_file))
    prefixes = tuple(str(prefix) for prefix in config["path_prefixes"])
    if not repo_path.startswith(prefixes):
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download project audio"
            ) from exc
        downloader = hf_hub_download

    cache_value = config.get("cache_dir")
    cache_dir = None
    if cache_value:
        cache_path = Path(str(cache_value))
        cache_dir = (
            cache_path
            if cache_path.is_absolute()
            else PROJECT_ROOT / cache_path
        )

    try:
        downloaded = downloader(
            repo_id=str(config["repo_id"]),
            filename=repo_path,
            repo_type=str(config["repo_type"]),
            revision=str(config["revision"]),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Audio is unavailable locally and on Hugging Face: {repo_path}"
        ) from exc

    resolved = Path(downloaded)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Hugging Face download did not produce a file: {repo_path}"
        )
    return resolved.resolve()
