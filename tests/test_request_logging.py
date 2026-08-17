from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.utils.request_logging import RequestLogger


@pytest.fixture(autouse=True)
def _clear_request_loggers():
    RequestLogger.clear_instances()
    yield
    RequestLogger.clear_instances()


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "logging": {
                    "requests": {
                        "enabled": True,
                        "level": "INFO",
                        "console": False,
                        "file_path": "logs/requests.log",
                        "max_bytes": 4096,
                        "backup_count": 2,
                        "include_transcript": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_request_logger_writes_privacy_safe_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    logger = RequestLogger.from_config(config)
    request_id = logger.start(tmp_path / "private-command.wav")
    logger.finish(
        request_id,
        {
            "transcript": "đọc ghi chú bí mật",
            "intent": "VIEW_PRIVATE_NOTE",
            "policy": "SID_AND_SV",
            "speaker": {
                "candidate_user_id": "user_001",
                "similarity": 0.81,
                "identified": True,
                "verified": True,
            },
            "error": None,
            "success": True,
        },
    )

    records = _records(tmp_path / "logs" / "requests.log")
    assert [record["event"] for record in records] == ["request_started", "request_finished"]
    assert records[0]["request_id"] == records[1]["request_id"]
    assert records[1]["intent"] == "VIEW_PRIVATE_NOTE"
    assert records[1]["verified"] is True
    assert records[1]["duration_ms"] >= 0
    assert "transcript" not in records[1]
    assert "ghi chú bí mật" not in (tmp_path / "logs" / "requests.log").read_text(encoding="utf-8")


def test_logger_can_include_transcript_and_does_not_duplicate_handlers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["logging"]["requests"]["include_transcript"] = True
    config.write_text(yaml.safe_dump(document), encoding="utf-8")

    first = RequestLogger.from_config(config)
    second = RequestLogger.from_config(config)
    request_id = second.start("query.wav")
    second.finish(request_id, {"transcript": "xin chào", "speaker": {}})

    records = _records(tmp_path / "logs" / "requests.log")
    assert len(records) == 2
    assert records[-1]["transcript"] == "xin chào"
    first.close()
    second.close()


def test_disabled_request_logger_creates_no_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["logging"]["requests"]["enabled"] = False
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    logger = RequestLogger.from_config(config)

    request_id = logger.start("query.wav")
    logger.finish(request_id, {"success": True, "speaker": {}})

    assert not (tmp_path / "logs" / "requests.log").exists()


def test_console_prints_one_request_field_per_line(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["logging"]["requests"]["console"] = True
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    logger = RequestLogger.from_config(config)

    request_id = logger.start("command.wav")
    logger.finish(
        request_id,
        {
            "success": True,
            "intent": "GET_TIME",
            "policy": "PUBLIC",
            "speaker": {},
            "error": None,
        },
    )

    output = capsys.readouterr().out
    assert "event: request_started" in output
    assert "audio_name: command.wav" in output
    assert "event: request_finished" in output
    assert "intent: GET_TIME" in output
    assert "success: true" in output
    assert '{"event"' not in output
