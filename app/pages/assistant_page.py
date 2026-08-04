"""Voice Assistant Week 1 skeleton."""

from __future__ import annotations

import streamlit as st

from src.database.user_repository import list_users
from src.pipeline.orchestrator import process_request
from src.tts.text_to_speech import synthesize_vietnamese


def render_assistant_page() -> None:
    st.title("Voice Assistant")
    st.caption("Week 1: audio input and deterministic ASR/NLU/Speaker mocks.")
    recorded = st.audio_input("Thu âm WAV")
    uploaded = st.file_uploader("Hoặc tải WAV", type=["wav"])
    audio = recorded or uploaded
    if audio:
        st.audio(audio)

    users = list_users()
    user_ids = [user["user_id"] for user in users]
    candidate = st.selectbox("Mock candidate user", user_ids or ["demo-anh"])
    transcript = st.text_input("Mock transcript", placeholder="Ví dụ: Hôm nay tôi có lịch gì?")
    verification = st.checkbox("Mock verification pass", value=True)
    if st.button("Xử lý", type="primary"):
        result = process_request(transcript=transcript, candidate_user_id=candidate, verification_passed=verification)
        st.json({key: result[key] for key in ("transcript", "intent", "speaker", "similarity", "verification", "policy", "response")})
        tts_audio = synthesize_vietnamese(result["response"])
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3")
        else:
            st.info("TTS chưa khả dụng; response text vẫn hiển thị.")
