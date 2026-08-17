"""Application-user voice enrollment page."""

from __future__ import annotations

import streamlit as st

from app.backend_client import BackendRequestError, get_backend_client


def render_file_status(result: dict) -> None:
    for index, item in enumerate(result.get("file_results", []), start=1):
        label = Path(str(item.get("audio_path", ""))).name or f"audio {index}"
        if item.get("valid"):
            st.success(f"{index}. {label}: hợp lệ")
        else:
            st.error(f"{index}. {label}: {item.get('error') or 'audio không hợp lệ'}")
    if result.get("success"):
        st.success(f"Đăng ký thành công: {result['user_id']}.")
        st.caption(f"Centroid: {result['centroid_path']}")
        st.caption(
            f"Dimension: {result.get('embedding_dim')} · "
            f"L2 norm: {float(result.get('l2_norm') or 0):.6f} · "
            f"Latency: {float(result.get('latency_ms') or 0):.1f} ms"
        )
    else:
        st.error(f"Đăng ký thất bại: {result.get('error', 'UNKNOWN_ERROR')}")


def render_enrollment_page() -> None:
    st.title("Speaker Enrollment")
    st.caption(
        "Thu âm hoặc tải lên đúng 5 WAV độc lập. Mỗi file tạo một embedding "
        "và chỉ centroid application được lưu."
    )
    user_id = st.text_input("User ID", placeholder="user_001")
    name = st.text_input("Tên người dùng", placeholder="Anh")
    uploaded_files = st.file_uploader(
        "Tải WAV", type=["wav"], accept_multiple_files=True,
    )
    recorded = st.audio_input("Hoặc thu một mẫu WAV")
    if "enrollment_recordings" not in st.session_state:
        st.session_state.enrollment_recordings = []
    if st.button("Thêm bản thu vào enrollment", disabled=recorded is None):
        payload = recorded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        existing = {item["sha256"] for item in st.session_state.enrollment_recordings}
        if digest in existing:
            st.warning("Bản thu này đã có trong danh sách.")
        elif len(st.session_state.enrollment_recordings) >= 5:
            st.warning("Danh sách đã đủ 5 bản thu.")
        else:
            st.session_state.enrollment_recordings.append(
                {"bytes": payload, "sha256": digest}
            )
            st.rerun()
    if st.button(
        "Xóa các bản thu đã thêm",
        disabled=not st.session_state.enrollment_recordings,
    ):
        st.session_state.enrollment_recordings = []
        st.rerun()

    recordings = st.session_state.enrollment_recordings
    total_audio = len(uploaded_files) + len(recordings)
    st.write(f"Đã chọn: {total_audio}/5 audio")
    for index, audio_file in enumerate(uploaded_files, start=1):
        st.caption(f"{index}. upload · {audio_file.name} · {audio_file.size} bytes")
    for offset, item in enumerate(recordings, start=len(uploaded_files) + 1):
        st.caption(f"{offset}. recording · {len(item['bytes'])} bytes")

    if st.button("Đăng ký", type="primary"):
        if not user_id.strip() or not name.strip():
            st.error("User ID và tên bắt buộc.")
        elif total_audio != 5:
            st.error("Cần đúng 5 audio WAV, tính cả upload và bản thu.")
        else:
            files = [
                (audio_file.name or f"{index}.wav", audio_file.getvalue())
                for index, audio_file in enumerate(audio_files, start=1)
            ]
            try:
                with st.spinner("Đang tạo embedding và centroid..."):
                    result = get_backend_client().enroll(
                        user_id.strip(),
                        name.strip(),
                        files,
                    )
            except BackendRequestError as error:
                st.error(str(error))
                return
            render_file_status(result)
            if result.get("success"):
                st.session_state.enrollment_recordings = []
