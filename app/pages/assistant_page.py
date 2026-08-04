"""Voice Assistant Week 1 skeleton."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from src.database.user_repository import list_users
from src.pipeline.asr_nlu import run_asr_nlu_pipeline
from src.pipeline.orchestrator import process_request
from src.tts.text_to_speech import synthesize_vietnamese


def render_assistant_page() -> None:
    st.title("Voice Assistant")
    st.caption("WAV input → Whisper Small CPU → NLU → policy → speaker mock → database → TTS")
    recorded = st.audio_input("Thu âm WAV")
    uploaded = st.file_uploader("Hoặc tải WAV", type=["wav"])
    audio = recorded or uploaded
    if audio:
        st.audio(audio)

    users = list_users()
    user_ids = [user["user_id"] for user in users]
    candidate = st.selectbox("Mock candidate user", user_ids or ["demo-anh"])
    verification = st.checkbox("Mock verification pass", value=True)
    if st.button("Xử lý", type="primary"):
        if audio is None:
            st.error("Hãy thu âm hoặc tải file WAV.")
            return

        with TemporaryDirectory(prefix="voicestudy-") as directory:
            audio_path = Path(directory) / "command.wav"
            audio_path.write_bytes(audio.getvalue())
            pipeline = run_asr_nlu_pipeline(audio_path)

        if not pipeline["success"]:
            st.error(f"ASR thất bại: {pipeline['error']}")
            return

        result = process_request(
            transcript=pipeline["transcript"],
            candidate_user_id=candidate,
            verification_passed=verification,
            intent=pipeline["intent"],
            entities=pipeline["entities"],
            missing_fields=pipeline["missing_fields"],
        )
        st.json({
            key: result[key]
            for key in ("transcript", "intent", "entities", "speaker", "similarity", "verification", "policy", "response")
        })
        tts_audio = synthesize_vietnamese(result["response"])
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3")
        else:
            st.info("TTS chưa khả dụng; response text vẫn hiển thị.")
