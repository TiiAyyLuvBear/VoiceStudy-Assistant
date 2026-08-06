"""Manage application users and refresh their enrollment."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from app.pages.enrollment_page import render_file_status
from src.database.user_repository import delete_user, list_users
from src.speaker.application import enroll_user


def render_user_management_page() -> None:
    st.title("User Management")
    users = list_users()
    if not users:
        st.info("Chưa có user. Đăng ký ở Speaker Enrollment.")
        return
    st.dataframe(users, use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Chọn user", [user["user_id"] for user in users])
    selected = next(user for user in users if user["user_id"] == selected_id)
    refresh_files = st.file_uploader(
        "Cập nhật enrollment: chọn đúng 5 WAV", type=["wav"],
        accept_multiple_files=True, key="refresh_enrollment",
    )
    if st.button("Cập nhật enrollment"):
        if len(refresh_files) != 5:
            st.error("Cần đúng 5 file WAV.")
        else:
            with TemporaryDirectory(prefix="voicestudy-refresh-") as directory:
                paths = []
                for index, audio_file in enumerate(refresh_files, start=1):
                    path = Path(directory) / f"{index}.wav"
                    path.write_bytes(audio_file.getvalue())
                    paths.append(path)
                render_file_status(enroll_user(selected_id, selected["name"], paths))
    confirm_delete = st.checkbox(f"Xác nhận xóa {selected_id} cùng lịch và ghi chú")
    if st.button("Xóa user", disabled=not confirm_delete):
        delete_user(selected_id)
        st.success("Đã xóa user.")
        st.rerun()
