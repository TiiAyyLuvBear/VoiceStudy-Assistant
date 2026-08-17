"""Manage application users and refresh their enrollment."""

from __future__ import annotations

import streamlit as st

from app.backend_client import BackendRequestError, get_backend_client
from app.pages.enrollment_page import render_file_status


def render_user_management_page() -> None:
    st.title("User Management")
    try:
        users = get_backend_client().list_users()
    except BackendRequestError as error:
        st.error(str(error))
        return
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
            files = [
                (audio_file.name or f"{index}.wav", audio_file.getvalue())
                for index, audio_file in enumerate(refresh_files, start=1)
            ]
            try:
                result = get_backend_client().enroll(selected_id, selected["name"], files)
            except BackendRequestError as error:
                st.error(str(error))
            else:
                render_file_status(result)
    confirm_delete = st.checkbox(f"Xác nhận xóa {selected_id} cùng lịch và ghi chú")
    if st.button("Xóa user", disabled=not confirm_delete):
        try:
            get_backend_client().delete_user(selected_id)
        except BackendRequestError as error:
            st.error(str(error))
        else:
            st.success("Đã xóa user.")
            st.rerun()
