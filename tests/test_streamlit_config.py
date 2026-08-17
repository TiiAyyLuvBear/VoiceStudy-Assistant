from pathlib import Path


def test_streamlit_disables_module_file_watcher() -> None:
    path = Path(".streamlit/config.toml")
    assert path.is_file()
    config = path.read_text(encoding="utf-8")
    assert '[server]' in config
    assert 'fileWatcherType = "none"' in config
    assert 'runOnSave = false' in config
