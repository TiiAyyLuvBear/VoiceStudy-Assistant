"""Streamlit entry point for VoiceStudy-Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.pages.assistant_page import render_assistant_page
from app.pages.enrollment_page import render_enrollment_page
from app.pages.user_management_page import render_user_management_page
from src.utils.config import load_yaml_mapping
from src.utils.speechbrain_lazy import remove_speechbrain_lazy_modules

def load_config(config_path: Path) -> dict:
    """Load configuration from a YAML file."""
    config, _ = load_yaml_mapping(config_path)
    return config

def main() -> None:
    remove_speechbrain_lazy_modules()
    st.set_page_config(page_title="VoiceStudy Assistant", page_icon="🎙️", layout="wide")
    load_config(CONFIG_PATH)
    st.sidebar.title("VoiceStudy Assistant")
    page = st.sidebar.radio("Trang", ("Voice Assistant", "Speaker Enrollment", "User Management"))
    if page == "Voice Assistant":
        render_assistant_page()
    elif page == "Speaker Enrollment":
        render_enrollment_page()
    else:
        render_user_management_page()


if __name__ == "__main__":
    main()
