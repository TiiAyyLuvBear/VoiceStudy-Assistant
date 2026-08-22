"""Application-user voice enrollment page."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from app.backend_client import BackendRequestError, get_backend_client

DEFAULT_ENROLLMENT_PROMPTS: tuple[str, ...] = (
    "Bây giờ là mấy giờ rồi?",
    "Cho tôi xem lịch học ngày mai.",
    "Thêm lịch học thống kê lúc tám giờ sáng.",
    "Mở ghi chú riêng tư gần nhất của tôi.",
    "Hôm nay tôi cần kiểm tra lịch học và chuẩn bị bài thuyết trình.",
)


def render_file_status(result: dict) -> None:
    failed_index = result.get("failed_sample_index")
    failed_prompt = result.get("failed_prompt")
    for index, item in enumerate(result.get("file_results", []), start=1):
        label = Path(str(item.get("audio_path", ""))).name or f"audio {index}"
        if item.get("valid"):
            st.success(f"{index}. {label}: hợp lệ")
        else:
            message = (
                item.get("message_vi")
                or (item.get("quality") or {}).get("message_vi")
                or result.get("message_vi")
                or item.get("error")
                or "audio không hợp lệ"
            )
            st.error(f"{index}. {label}: {message}")
    if result.get("success"):
        st.success(f"Đăng ký thành công: {result['user_id']}.")
        st.caption(f"Centroid: {result['centroid_path']}")
        st.caption(
            f"Dimension: {result.get('embedding_dim')} · "
            f"L2 norm: {float(result.get('l2_norm') or 0):.6f} · "
            f"Latency: {float(result.get('latency_ms') or 0):.1f} ms"
        )
    else:
        st.error(result.get("message_vi") or "Đăng ký thất bại. Hãy thử lại.")
        if failed_index and failed_prompt:
            st.warning(
                f"Thu lại voice mẫu {failed_index}. Câu gợi ý: {failed_prompt}. "
                "Không bắt buộc đọc đúng nội dung gợi ý."
            )
        if result.get("secret_phrase_transcript"):
            st.caption(f"Raw secret ASR: {result['secret_phrase_transcript']}")


def build_enrollment_files(uploaded_files, recordings: list[dict]) -> list[tuple[str, bytes]]:
    """Combine uploaded audio files and recorded samples for backend enrollment."""
    files = [
        (audio_file.name or f"{index}.wav", audio_file.getvalue())
        for index, audio_file in enumerate(uploaded_files, start=1)
    ]
    files.extend(
        (f"recording_{index}.wav", item["bytes"])
        for index, item in enumerate(recordings, start=len(files) + 1)
    )
    return files


def render_enrollment_page() -> None:
    st.title("Speaker Enrollment")
    st.caption(
        "Thu âm hoặc tải lên 3-10 WAV/FLAC độc lập, khuyến nghị 5 mẫu. Mỗi file tạo một embedding "
        "và chỉ centroid application được lưu."
    )
    user_id = st.text_input("User ID", placeholder="user_001")
    name = st.text_input("Tên người dùng", placeholder="Anh")
    secret_phrase = st.text_input(
        "Transcript câu lệnh bí mật",
        placeholder="hoa sen xanh an toàn",
    )
    st.caption("Đọc đúng câu này vào audio bí mật bên dưới để backend xác nhận ASR.")
    secret_recorded = st.audio_input("Thu câu lệnh bí mật")
    st.write(
        "Thu 5 mẫu voice đạt chuẩn ECAPA. Câu bên dưới chỉ là gợi ý, "
        "backend không kiểm tra nội dung có khớp câu mẫu không:"
    )
    for index, prompt in enumerate(DEFAULT_ENROLLMENT_PROMPTS, start=1):
        st.caption(f"{index}. {prompt}")
    uploaded_files = st.file_uploader(
        "Tải WAV/FLAC", type=["wav", "flac"], accept_multiple_files=True,
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
        elif len(st.session_state.enrollment_recordings) >= 10:
            st.warning("Danh sách đã đủ 10 bản thu.")
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
    st.write(f"Đã chọn: {total_audio}/5 khuyến nghị, tối thiểu 3 và tối đa 10 audio")
    for index, audio_file in enumerate(uploaded_files, start=1):
        st.caption(f"{index}. upload · {audio_file.name} · {audio_file.size} bytes")
    for offset, item in enumerate(recordings, start=len(uploaded_files) + 1):
        st.caption(f"{offset}. recording · {len(item['bytes'])} bytes")

    if st.button("Đăng ký", type="primary"):
        if not user_id.strip() or not name.strip() or not secret_phrase.strip():
            st.error("User ID, tên và transcript câu lệnh bí mật bắt buộc.")
        elif secret_recorded is None:
            st.error("Cần audio đọc câu lệnh bí mật.")
        elif not 3 <= total_audio <= 10:
            st.error("Cần 3-10 audio WAV/FLAC, tính cả upload và bản thu.")
        else:
            files = build_enrollment_files(uploaded_files, recordings)
            try:
                with st.spinner("Đang tạo embedding và centroid..."):
                    result = get_backend_client().enroll(
                        user_id.strip(),
                        name.strip(),
                        secret_phrase.strip(),
                        ("secret_phrase.wav", secret_recorded.getvalue()),
                        files,
                        DEFAULT_ENROLLMENT_PROMPTS,
                    )
            except BackendRequestError as error:
                st.error(str(error))
                return
            render_file_status(result)
            if result.get("success"):
                st.session_state.enrollment_recordings = []
