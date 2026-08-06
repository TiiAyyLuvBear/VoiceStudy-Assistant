"""Voice Assistant end-to-end page."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from src.pipeline.orchestrator import process_audio_request
from src.tts.text_to_speech import synthesize_vietnamese


def render_assistant_page() -> None:
    st.title("Voice Assistant")
    st.caption("WAV → ASR → NLU → Application SID/SV → database → TTS")
    recorded = st.audio_input("Thu âm WAV")
    uploaded = st.file_uploader("Hoặc tải WAV", type=["wav"])
    audio = recorded or uploaded
    if audio:
        st.audio(audio)
    if st.button("Xử lý", type="primary"):
        if audio is None:
            st.error("Hãy thu âm hoặc tải file WAV.")
            return
        with TemporaryDirectory(prefix="voicestudy-command-") as directory:
            path = Path(directory) / "command.wav"
            path.write_bytes(audio.getvalue())
            with st.spinner("Đang xử lý audio..."):
                result = process_audio_request(path)
        st.json({
            key: result[key] for key in (
                "transcript", "intent", "entities", "missing_fields", "speaker",
                "policy", "response", "error",
            )
        })
        tts_audio = synthesize_vietnamese(result["response"])
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3")
        else:
            st.info("TTS chưa khả dụng; response text vẫn hiển thị.")
