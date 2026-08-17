from __future__ import annotations

from pathlib import Path

import yaml

from app.backend_client import BackendClient


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self):
        return self.payload


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _Response({"success": True})


def test_client_reads_yaml_and_sends_pipeline_audio(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"backend": {"base_url": "http://127.0.0.1:9000", "request_timeout_seconds": 30}}
        ),
        encoding="utf-8",
    )
    session = _Session()
    client = BackendClient.from_config(config, session=session)

    result = client.process_audio("command.wav", b"wav-data", reference_date="2026-08-17")

    method, url, kwargs = session.calls[0]
    assert result == {"success": True}
    assert method == "POST"
    assert url == "http://127.0.0.1:9000/api/v1/process"
    assert kwargs["files"]["audio"][0] == "command.wav"
    assert kwargs["data"] == {"reference_date": "2026-08-17"}
    assert kwargs["timeout"] == 30


def test_client_supports_enrollment_users_and_delete() -> None:
    session = _Session()
    client = BackendClient("http://localhost:8000", timeout_seconds=10, session=session)

    client.enroll("user_001", "Student", [("1.wav", b"a")] * 5)
    client.list_users()
    client.delete_user("user_001")

    assert [call[0] for call in session.calls] == ["POST", "GET", "DELETE"]
    assert session.calls[0][1].endswith("/api/v1/enroll")
    assert session.calls[1][1].endswith("/api/v1/users")
    assert session.calls[2][1].endswith("/api/v1/users/user_001")


def test_streamlit_pages_do_not_import_pipeline_speaker_or_database_services() -> None:
    pages = [
        Path("app/pages/assistant_page.py"),
        Path("app/pages/enrollment_page.py"),
        Path("app/pages/user_management_page.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in pages)

    assert "src.pipeline" not in source
    assert "src.speaker" not in source
    assert "src.database" not in source
    assert "get_backend_client" in source
