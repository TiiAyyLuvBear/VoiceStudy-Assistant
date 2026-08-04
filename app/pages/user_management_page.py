"""User Management Week 1 skeleton."""

from __future__ import annotations

import streamlit as st

from scripts.seed_database import seed_database
from src.database.user_repository import delete_user, list_users, update_embedding_path


def render_user_management_page() -> None:
    st.title("User Management")
    if st.button("Tạo dữ liệu demo"):
        seed_database()
        st.success("Đã tạo dữ liệu demo an toàn.")
    users = list_users()
    if not users:
        st.info("Chưa có user. Tạo dữ liệu demo hoặc dùng Speaker Enrollment.")
        return
    st.dataframe(users, use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Chọn user", [user["user_id"] for user in users])
    selected = next(user for user in users if user["user_id"] == selected_id)
    embedding_path = st.text_input("Embedding path", value=selected["embedding_path"] or "")
    if st.button("Cập nhật embedding path"):
        update_embedding_path(selected_id, embedding_path or None)
        st.success("Đã cập nhật.")
    confirm_delete = st.checkbox(f"Xác nhận xóa {selected_id} cùng lịch và ghi chú")
    if st.button("Xóa user", disabled=not confirm_delete):
        delete_user(selected_id)
        st.success("Đã xóa user.")
        st.rerun()
