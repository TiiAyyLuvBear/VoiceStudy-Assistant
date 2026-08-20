"""Voice Assistant end-to-end page."""

from __future__ import annotations

import streamlit as st

from app.backend_client import BackendRequestError, get_backend_client
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
        try:
            with st.spinner("Đang xử lý audio..."):
                result = get_backend_client().process_audio(
                    audio.name or "command.wav",
                    audio.getvalue(),
                )
        except BackendRequestError as error:
            st.error(str(error))
            return
        st.json({
            key: result.get(key) for key in (
                "transcript", "intent", "entities", "missing_fields", "speaker",
                "policy", "stage_latency_ms", "latency_ms", "response", "error",
            )
        })
        response = result.get("response") or ""
        if result.get("error"):
            st.warning(response or "Backend returned an error without a response.")
        else:
            st.success(response)
        tts_audio = synthesize_vietnamese(response) if response else None
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3")
        else:
            st.info("TTS chưa khả dụng; response text vẫn hiển thị.")
