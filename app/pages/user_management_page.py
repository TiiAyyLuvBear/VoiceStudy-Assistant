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
    st.subheader("Lịch trình")
    try:
        schedules = get_backend_client().list_schedules(selected_id)
    except BackendRequestError as error:
        st.error(str(error))
        schedules = []
    if schedules:
        st.dataframe(schedules, use_container_width=True, hide_index=True)
    else:
        st.info("User chưa có lịch.")
    with st.form("add_schedule"):
        title = st.text_input("Tiêu đề lịch")
        schedule_date = st.date_input("Ngày")
        schedule_time = st.time_input("Giờ")
        description = st.text_input("Mô tả", value="")
        if st.form_submit_button("Thêm lịch"):
            if not title.strip():
                st.error("Tiêu đề lịch bắt buộc.")
            else:
                try:
                    get_backend_client().add_schedule(
                        selected_id,
                        title=title.strip(),
                        date=schedule_date.isoformat(),
                        time=schedule_time.strftime("%H:%M"),
                        description=description.strip() or None,
                    )
                except BackendRequestError as error:
                    st.error(str(error))
                else:
                    st.success("Đã thêm lịch.")
                    st.rerun()
    if schedules:
        delete_schedule_id = st.selectbox(
            "Xóa lịch",
            [row["schedule_id"] for row in schedules],
            format_func=lambda value: next(
                f"{row['title']} - {row['date']} {row['time']}"
                for row in schedules
                if row["schedule_id"] == value
            ),
        )
        if st.button("Xóa lịch đã chọn"):
            try:
                get_backend_client().delete_schedule(selected_id, int(delete_schedule_id))
            except BackendRequestError as error:
                st.error(str(error))
            else:
                st.success("Đã xóa lịch.")
                st.rerun()

    st.subheader("Ghi chú")
    try:
        notes = get_backend_client().list_notes(selected_id)
    except BackendRequestError as error:
        st.error(str(error))
        notes = []
    if notes:
        st.dataframe(notes, use_container_width=True, hide_index=True)
    else:
        st.info("User chưa có ghi chú.")
    with st.form("add_note"):
        content = st.text_area("Nội dung ghi chú")
        is_private = st.checkbox("Ghi chú riêng tư", value=True)
        if st.form_submit_button("Thêm ghi chú"):
            if not content.strip():
                st.error("Nội dung ghi chú bắt buộc.")
            else:
                try:
                    get_backend_client().add_note(
                        selected_id,
                        content=content.strip(),
                        is_private=is_private,
                    )
                except BackendRequestError as error:
                    st.error(str(error))
                else:
                    st.success("Đã thêm ghi chú.")
                    st.rerun()
    if notes:
        delete_note_id = st.selectbox(
            "Xóa ghi chú",
            [row["note_id"] for row in notes],
            format_func=lambda value: next(
                f"{row['note_id']} - {row['content'][:40]}"
                for row in notes
                if row["note_id"] == value
            ),
        )
        if st.button("Xóa ghi chú đã chọn"):
            try:
                get_backend_client().delete_note(selected_id, int(delete_note_id))
            except BackendRequestError as error:
                st.error(str(error))
            else:
                st.success("Đã xóa ghi chú.")
                st.rerun()

    st.subheader("Cập nhật enrollment")
    refresh_secret = st.text_input(
        "Transcript câu lệnh bí mật mới",
    )
    refresh_secret_audio = st.audio_input("Thu câu lệnh bí mật mới")
    refresh_files = st.file_uploader(
        "Cập nhật enrollment: chọn 3-10 WAV/FLAC", type=["wav", "flac"],
        accept_multiple_files=True, key="refresh_enrollment",
    )
    if st.button("Cập nhật enrollment"):
        if not 3 <= len(refresh_files) <= 10:
            st.error("Cần 3-10 file WAV/FLAC.")
        elif not refresh_secret.strip():
            st.error("Transcript câu lệnh bí mật bắt buộc khi cập nhật enrollment.")
        elif refresh_secret_audio is None:
            st.error("Cần audio đọc câu lệnh bí mật mới.")
        else:
            files = [
                (audio_file.name or f"{index}.wav", audio_file.getvalue())
                for index, audio_file in enumerate(refresh_files, start=1)
            ]
            try:
                result = get_backend_client().enroll(
                    selected_id,
                    selected["name"],
                    refresh_secret.strip(),
                    ("secret_phrase.wav", refresh_secret_audio.getvalue()),
                    files,
                )
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
