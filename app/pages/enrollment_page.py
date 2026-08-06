"""Application-user voice enrollment page."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from src.speaker.application import enroll_user


def render_file_status(result: dict) -> None:
    if result.get("success"):
        st.success(f"Đăng ký thành công: {result['user_id']}.")
        st.caption(f"Centroid: {result['centroid_path']}")
    else:
        st.error(f"Đăng ký thất bại: {result.get('error', 'UNKNOWN_ERROR')}")


def render_enrollment_page() -> None:
    st.title("Speaker Enrollment")
    st.caption("Nhập đúng 5 WAV. Mỗi file được dùng tạo một embedding và centroid chung.")
    user_id = st.text_input("User ID", placeholder="user_001")
    name = st.text_input("Tên người dùng", placeholder="Anh")
    audio_files = st.file_uploader(
        "Chọn đúng 5 WAV", type=["wav"], accept_multiple_files=True,
    )
    for index, audio_file in enumerate(audio_files, start=1):
        st.caption(f"{index}. {audio_file.name} ({audio_file.size} bytes)")

    if st.button("Đăng ký", type="primary"):
        if not user_id.strip() or not name.strip():
            st.error("User ID và tên bắt buộc.")
        elif len(audio_files) != 5:
            st.error("Cần đúng 5 file WAV.")
        else:
            with TemporaryDirectory(prefix="voicestudy-enrollment-") as directory:
                paths: list[Path] = []
                for index, audio_file in enumerate(audio_files, start=1):
                    path = Path(directory) / f"{index}.wav"
                    path.write_bytes(audio_file.getvalue())
                    paths.append(path)
                with st.spinner("Đang tạo embedding và centroid..."):
                    result = enroll_user(user_id.strip(), name.strip(), paths)
            render_file_status(result)
