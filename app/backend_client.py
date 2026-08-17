"""HTTP client used by Streamlit to call the separate FastAPI backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

from src.utils.config import load_yaml_mapping


DEFAULT_CONFIG_PATH = Path("config.yaml")


class BackendRequestError(RuntimeError):
    """Raised when Streamlit cannot complete a backend request."""


class BackendClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        session: Any | None = None,
    ) -> "BackendClient":
        config, _ = load_yaml_mapping(config_path)
        backend = config.get("backend", {})
        return cls(
            str(backend.get("base_url", "http://127.0.0.1:8000")),
            timeout_seconds=float(backend.get("request_timeout_seconds", 300)),
            session=session,
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise BackendRequestError(f"Backend request failed: {error}") from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def process_audio(
        self,
        filename: str,
        content: bytes,
        *,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        data = {"reference_date": reference_date} if reference_date else {}
        return self._request(
            "POST",
            "/api/v1/process",
            data=data,
            files={"audio": (filename, content, "audio/wav")},
        )

    def enroll(
        self,
        user_id: str,
        name: str,
        audio_files: Iterable[tuple[str, bytes]],
    ) -> dict[str, Any]:
        files = [
            ("audio_files", (filename, content, "audio/wav"))
            for filename, content in audio_files
        ]
        return self._request(
            "POST",
            "/api/v1/enroll",
            data={"user_id": user_id, "name": name},
            files=files,
        )

    def list_users(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/users")

    def delete_user(self, user_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/users/{quote(user_id, safe='')}")


@lru_cache(maxsize=1)
def get_backend_client(config_path: str | Path = DEFAULT_CONFIG_PATH) -> BackendClient:
    return BackendClient.from_config(config_path)
