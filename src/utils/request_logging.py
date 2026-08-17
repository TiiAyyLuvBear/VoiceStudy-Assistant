"""Structured request logging for the application pipeline."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping

from src.utils.config import load_yaml_mapping, resolve_path


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_LOGGER_NAME = "voicestudy.requests"


class ConsoleFieldsFormatter(logging.Formatter):
    """Render structured JSON with exactly one field on each terminal line."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            document = json.loads(record.getMessage())
        except (TypeError, json.JSONDecodeError):
            return record.getMessage()
        return "\n".join(
            f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}"
            if not isinstance(value, str)
            else f"{key}: {value}"
            for key, value in document.items()
        )


class RequestLogger:
    """Write privacy-aware JSON request events to terminal and rotating file."""

    _instances: dict[str, "RequestLogger"] = {}
    _instances_lock = Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        level: str,
        console: bool,
        file_path: Path | None,
        max_bytes: int,
        backup_count: int,
        include_transcript: bool,
        instance_key: str,
    ) -> None:
        self.enabled = enabled
        self.include_transcript = include_transcript
        self._started: dict[str, float] = {}
        self._logger = logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{abs(hash(instance_key))}")
        self._logger.propagate = False
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
        if not enabled:
            return
        formatter = logging.Formatter("%(message)s")
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFieldsFormatter())
            self._logger.addHandler(console_handler)
        if file_path is not None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "RequestLogger":
        resolved_config = Path(config_path).resolve()
        instance_key = str(resolved_config)
        with cls._instances_lock:
            cached = cls._instances.get(instance_key)
            if cached is not None:
                return cached
            config, root = load_yaml_mapping(resolved_config)
            settings = config.get("logging", {}).get("requests", {})
            file_value = settings.get("file_path", "logs/requests.log")
            file_path = resolve_path(file_value, root) if file_value else None
            instance = cls(
                enabled=bool(settings.get("enabled", True)),
                level=str(settings.get("level", "INFO")),
                console=bool(settings.get("console", True)),
                file_path=file_path,
                max_bytes=int(settings.get("max_bytes", 5_242_880)),
                backup_count=int(settings.get("backup_count", 3)),
                include_transcript=bool(settings.get("include_transcript", False)),
                instance_key=instance_key,
            )
            cls._instances[instance_key] = instance
            return instance

    @classmethod
    def clear_instances(cls) -> None:
        with cls._instances_lock:
            for instance in cls._instances.values():
                instance.close()
            cls._instances.clear()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _write(self, record: Mapping[str, Any]) -> None:
        if self.enabled:
            self._logger.info(json.dumps(record, ensure_ascii=False, default=str))

    def start(self, audio_path: str | Path) -> str:
        request_id = uuid.uuid4().hex
        self._started[request_id] = time.perf_counter()
        self._write(
            {
                "timestamp": self._timestamp(),
                "event": "request_started",
                "request_id": request_id,
                "audio_name": Path(audio_path).name,
            }
        )
        return request_id

    def finish(self, request_id: str, result: Mapping[str, Any]) -> None:
        started = self._started.pop(request_id, time.perf_counter())
        speaker = result.get("speaker")
        speaker = speaker if isinstance(speaker, Mapping) else {}
        verification = speaker.get("verification")
        verification = verification if isinstance(verification, Mapping) else {}
        record: dict[str, Any] = {
            "timestamp": self._timestamp(),
            "event": "request_finished",
            "request_id": request_id,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "success": result.get("success"),
            "intent": result.get("intent"),
            "policy": result.get("policy"),
            "candidate_user_id": speaker.get("candidate_user_id"),
            "sid_similarity": speaker.get("similarity"),
            "identified": speaker.get("identified"),
            "verification_similarity": verification.get("similarity"),
            "verified": speaker.get("verified"),
            "error": result.get("error"),
        }
        if self.include_transcript:
            record["transcript"] = result.get("transcript")
        self._write(record)

    def fail(self, request_id: str, error: BaseException) -> None:
        started = self._started.pop(request_id, time.perf_counter())
        self._write(
            {
                "timestamp": self._timestamp(),
                "event": "request_failed",
                "request_id": request_id,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            try:
                handler.flush()
            except ValueError:
                pass
            handler.close()
            self._logger.removeHandler(handler)


def log_audio_request(function: Callable[..., Mapping[str, Any]]) -> Callable[..., Mapping[str, Any]]:
    """Log one pipeline start/end pair without changing its public result."""

    @wraps(function)
    def wrapper(audio_path, *args, **kwargs):
        config_path = kwargs.get("config_path", DEFAULT_CONFIG_PATH)
        request_logger = RequestLogger.from_config(config_path)
        request_id = request_logger.start(audio_path)
        try:
            result = function(audio_path, *args, **kwargs)
        except Exception as error:
            request_logger.fail(request_id, error)
            raise
        request_logger.finish(request_id, result)
        return result

    return wrapper
