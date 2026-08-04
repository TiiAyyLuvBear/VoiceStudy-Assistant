"""Speaker Enrollment Week 1 skeleton."""

from __future__ import annotations

import streamlit as st

from src.database.user_repository import create_user, get_user, update_embedding_path


def render_enrollment_page() -> None:
    st.title("Speaker Enrollment")
    st.caption("Week 1 stores mock embedding path. Real ECAPA enrollment arrives in Week 2.")
    user_id = st.text_input("User ID", placeholder="demo-anh")
    name = st.text_input("Tên người dùng", placeholder="Anh")
    audio_files = st.file_uploader("Chọn đúng 5 WAV files", type=["wav"], accept_multiple_files=True)
    if st.button("Đăng ký"):
        if not user_id.strip() or not name.strip():
            st.error("User ID và tên bắt buộc.")
        elif len(audio_files) != 5:
            st.error("Cần đúng 5 file WAV.")
        else:
            embedding_path = f"models/application/user_embeddings/{user_id.strip()}.npy"
            try:
                if get_user(user_id.strip()):
                    update_embedding_path(user_id.strip(), embedding_path)
                else:
                    create_user(user_id.strip(), name.strip(), embedding_path)
                st.success(f"Đã lưu enrollment mock cho {name.strip()}.")
            except Exception as error:
                st.error(f"Không thể lưu user: {error}")
