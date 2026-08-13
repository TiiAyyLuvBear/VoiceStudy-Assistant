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
        speaker = result.get("speaker", {})
        columns = st.columns(4)
        columns[0].metric("Intent", result.get("intent") or "-")
        columns[1].metric(
            "Speaker",
            speaker.get("candidate_user_id") or speaker.get("status") or "-",
        )
        similarity = speaker.get("cosine_similarity")
        columns[2].metric(
            "Cosine",
            "-" if similarity is None else f"{float(similarity):.4f}",
        )
        columns[3].metric(
            "Tổng latency",
            f"{float(result.get('latency_ms') or 0):.1f} ms",
        )
        st.json({
            key: result[key] for key in (
                "transcript", "intent", "entities", "missing_fields", "speaker",
                "policy", "stage_latency_ms", "latency_ms", "response", "error",
            )
        })
        if result.get("error"):
            st.warning(result["response"])
        else:
            st.success(result["response"])
        tts_audio = synthesize_vietnamese(result["response"]) if result["response"] else None
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3")
        else:
            st.info("TTS chưa khả dụng; response text vẫn hiển thị.")
