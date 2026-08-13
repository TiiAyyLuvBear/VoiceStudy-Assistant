from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path("app/main.py")


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_voice_assistant_page_smoke() -> None:
    app = _app()
    assert app.sidebar.radio[0].value == "Voice Assistant"
    assert app.title[0].value == "Voice Assistant"
    assert app.button


def test_enrollment_page_smoke() -> None:
    app = _app()
    app.sidebar.radio[0].set_value("Speaker Enrollment").run()
    assert not app.exception
    assert app.title[0].value == "Speaker Enrollment"
    assert len(app.text_input) == 2


def test_user_management_page_smoke() -> None:
    app = _app()
    app.sidebar.radio[0].set_value("User Management").run()
    assert not app.exception
    assert app.title[0].value == "User Management"

