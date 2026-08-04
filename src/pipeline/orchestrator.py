"""Week 1 integration contract with explicit deterministic fallbacks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.database.user_repository import get_user
from src.security.access_policy import PUBLIC, REJECT, SID, SID_AND_SV, get_access_policy
from src.tasks.note_tasks import get_private_notes
from src.tasks.schedule_tasks import get_schedules


def _mock_parse(transcript: str) -> tuple[str, dict]:
    text = transcript.lower().strip()
    if any(word in text for word in ("mấy giờ", "giờ hiện tại", "bây giờ")):
        return "GET_TIME", {}
    if any(word in text for word in ("ghi chú", "bảo mật")):
        return "VIEW_PRIVATE_NOTE", {}
    if any(word in text for word in ("thêm lịch", "tạo lịch")):
        return "ADD_SCHEDULE", {"title": "Lịch học mẫu", "date": datetime.now().date().isoformat(), "time": "08:00"}
    if "lịch" in text:
        return "VIEW_SCHEDULE", {}
    return "OUT_OF_SCOPE", {}


def process_request(audio_path: str | Path | None = None, transcript: str | None = None, candidate_user_id: str = "demo-anh", verification_passed: bool = True, database_path: str | Path | None = None) -> dict:
    """Process Week 1 request. Audio is displayed by UI; ASR is mock until module arrives."""
    transcript = (transcript or "").strip()
    if not transcript:
        return {"transcript": "", "intent": "OUT_OF_SCOPE", "entities": {}, "speaker": "MOCK_UNKNOWN", "similarity": None, "verification": None, "response": "ASR chưa sẵn sàng. Nhập mock transcript để thử luồng.", "policy": REJECT}

    intent, entities = _mock_parse(transcript)
    policy = get_access_policy(intent)
    user = get_user(candidate_user_id, database_path) if policy in (SID, SID_AND_SV) else None
    speaker = user["name"] if user else "MOCK_UNKNOWN"
    similarity = 0.91 if user else 0.32
    verification = verification_passed if policy == SID_AND_SV else None

    if policy == PUBLIC:
        response = f"Bây giờ là {datetime.now().strftime('%H:%M')}."
    elif policy == REJECT:
        response = "Câu lệnh ngoài phạm vi hỗ trợ."
    elif not user:
        response = "Không nhận diện được người dùng đã đăng ký."
    elif policy == SID_AND_SV and not verification_passed:
        response = "Xác thực giọng nói thất bại. Không thể xem ghi chú riêng tư."
    elif intent == "VIEW_SCHEDULE":
        schedules = get_schedules(candidate_user_id, database_path=database_path)
        response = "Chưa có lịch học." if not schedules else f"{speaker} có {len(schedules)} lịch. Lịch gần nhất: {schedules[0]['title']} lúc {schedules[0]['time']}, {schedules[0]['date']}."
    elif intent == "VIEW_PRIVATE_NOTE":
        notes = get_private_notes(candidate_user_id, database_path)
        response = "Chưa có ghi chú riêng tư." if not notes else f"Ghi chú gần nhất: {notes[0]['content']}"
    else:
        response = "Mock ADD_SCHEDULE: chờ module NLU thật trước khi ghi lịch."

    return {"transcript": transcript, "intent": intent, "entities": entities, "speaker": speaker, "candidate_user_id": candidate_user_id if user else None, "similarity": similarity, "verification": verification, "response": response, "policy": policy, "audio_path": str(audio_path) if audio_path else None}
