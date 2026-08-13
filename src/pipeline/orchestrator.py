"""Week 1 integration contract with explicit deterministic fallbacks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from src.audio.source import resolve_audio_path
from src.database.user_repository import get_user
from src.pipeline.asr_nlu import run_asr_nlu_pipeline
from src.security.access_policy import PUBLIC, REJECT, SID, SID_AND_SV, get_access_policy
from src.speaker.application import identify_application_user, verify_speaker
from src.tasks.note_tasks import get_private_notes
from src.tasks.schedule_tasks import add_schedule, get_schedules


def _end_to_end_result(**values: object) -> dict:
    result = {
        "success": True, "transcript": "", "normalized_transcript": "",
        "intent": "OUT_OF_SCOPE", "entities": {}, "missing_fields": [],
        "policy": REJECT, "speaker": {"candidate_user_id": None, "similarity": None,
        "cosine_similarity": None, "unknown_threshold": None, "status": None,
        "identified": None, "verified": None, "verification_threshold": None,
        "centroid_path": None, "sid_latency_ms": None, "sv_latency_ms": None},
        "latency_ms": 0.0, "stage_latency_ms": {},
        "response": "", "error": None,
    }
    result.update(values)
    return result


def _finish(started_at: float, **values: object) -> dict:
    values["latency_ms"] = (time.perf_counter() - started_at) * 1000.0
    return _end_to_end_result(**values)


def process_audio_request(
    audio_path: str | Path,
    *,
    database_path: str | Path | None = None,
    config_path: str | Path = "config.yaml",
    reference_date: str | None = None,
    asr_nlu_runner=run_asr_nlu_pipeline,
    identifier=identify_application_user,
    verifier=verify_speaker,
) -> dict:
    """Run production audio → ASR/NLU → Application SID/SV → database flow.

    Personal user IDs come only from Application SID; callers cannot select the
    database owner through a transcript or request parameter.
    """
    started_at = time.perf_counter()
    try:
        resolved_audio_path = resolve_audio_path(audio_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return _finish(
            started_at,
            success=False,
            error=str(exc),
            response="Không thể tải audio. Vui lòng thử lại.",
        )

    try:
        pipeline = asr_nlu_runner(
            resolved_audio_path,
            reference_date=reference_date,
            config_path=config_path,
        )
    except Exception as exc:
        return _finish(
            started_at,
            success=False,
            error=f"ASR_NLU_ERROR: {exc}",
            response="Không thể xử lý audio. Vui lòng thử lại.",
        )
    if not pipeline["success"]:
        return _finish(
            started_at,
            success=False, transcript=pipeline["transcript"],
            intent=pipeline["intent"], error=pipeline["error"],
            response="Không thể xử lý audio. Vui lòng thử lại.",
        )

    if not str(pipeline.get("transcript", "")).strip():
        return _finish(
            started_at,
            success=False,
            transcript="",
            error="EMPTY_TRANSCRIPT",
            response="Không nhận được nội dung giọng nói. Vui lòng thử lại.",
        )

    intent = pipeline["intent"]
    entities = pipeline["entities"]
    policy = get_access_policy(intent)
    common = {
        "transcript": pipeline["transcript"],
        "normalized_transcript": pipeline["normalized_transcript"],
        "intent": intent, "entities": entities,
        "missing_fields": pipeline["missing_fields"], "policy": policy,
        "stage_latency_ms": {"asr_nlu": pipeline.get("latency_ms")},
    }
    if policy == PUBLIC:
        return _finish(
            started_at,
            **common, response=f"Bây giờ là {datetime.now().strftime('%H:%M')}."
        )
    if policy == REJECT:
        return _finish(
            started_at,
            **common,
            response=(
                "Câu lệnh ngoài phạm vi. Hệ thống hỗ trợ xem giờ, xem hoặc "
                "thêm lịch và xem ghi chú riêng tư."
            ),
            error="OUT_OF_SCOPE",
        )

    try:
        sid = identifier(
            resolved_audio_path,
            database_path=database_path,
            config_path=config_path,
        )
    except Exception as exc:
        return _finish(
            started_at,
            **common,
            success=False,
            error=f"SID_ERROR: {exc}",
            response="Không thể nhận diện người dùng.",
        )
    sid_similarity = sid.get("cosine_similarity", sid.get("similarity"))
    sid_identified = bool(
        sid.get("identified", sid.get("status") == "KNOWN")
    )
    speaker = {
        "candidate_user_id": sid.get("candidate_user_id"),
        "similarity": sid_similarity,
        "cosine_similarity": sid_similarity,
        "unknown_threshold": sid.get("unknown_threshold"),
        "status": sid.get("status"),
        "identified": sid_identified,
        "verified": None,
        "verification_threshold": None,
        "centroid_path": sid.get("centroid_path"),
        "sid_latency_ms": sid.get("latency_ms"),
        "sv_latency_ms": None,
    }
    common["stage_latency_ms"]["sid"] = sid.get("latency_ms")
    if not sid.get("success") or not sid_identified:
        return _finish(
            started_at,
            **common, speaker=speaker, error=sid.get("error") or "UNKNOWN_SPEAKER",
            response="Không nhận diện được người dùng đã đăng ký.",
        )
    candidate_user_id = sid["candidate_user_id"]

    if policy == SID_AND_SV:
        try:
            sv = verifier(
                resolved_audio_path,
                candidate_user_id,
                database_path=database_path,
                config_path=config_path,
            )
        except Exception as exc:
            return _finish(
                started_at,
                **common,
                speaker=speaker,
                success=False,
                error=f"SV_ERROR: {exc}",
                response="Không thể xác thực giọng nói.",
            )
        speaker["verification"] = sv
        speaker["verified"] = sv.get("verified")
        speaker["verification_threshold"] = sv.get("verification_threshold")
        speaker["sv_latency_ms"] = sv.get("latency_ms")
        common["stage_latency_ms"]["sv"] = sv.get("latency_ms")
        if not sv.get("success") or not sv.get("verified"):
            return _finish(
                started_at,
                **common, speaker=speaker,
                error=sv.get("error") or "VERIFICATION_FAILED",
                response="Xác thực giọng nói thất bại. Không thể xem ghi chú riêng tư.",
            )

    if intent == "ADD_SCHEDULE" and (
        pipeline["missing_fields"]
        or not all(entities.get(key) for key in ("title", "date", "time"))
    ):
        return _finish(
            started_at,
            **common,
            speaker=speaker,
            error="MISSING_FIELDS",
            response="Thiếu thông tin để thêm lịch: tiêu đề, ngày và giờ.",
        )

    database_started_at = time.perf_counter()
    try:
        if intent == "VIEW_SCHEDULE":
            schedules = get_schedules(
                candidate_user_id, entities.get("date"), database_path
            )
            response = "Chưa có lịch học." if not schedules else (
                f"Bạn có {len(schedules)} lịch. Lịch gần nhất: "
                f"{schedules[0]['title']} lúc {schedules[0]['time']}, "
                f"{schedules[0]['date']}."
            )
        elif intent == "ADD_SCHEDULE":
            schedule = add_schedule(
                candidate_user_id,
                entities["title"],
                entities["date"],
                entities["time"],
                database_path=database_path,
            )
            response = (
                f"Đã thêm lịch {schedule['title']} lúc {schedule['time']}, "
                f"{schedule['date']}."
            )
        else:  # VIEW_PRIVATE_NOTE after successful SV
            notes = get_private_notes(candidate_user_id, database_path)
            response = (
                "Chưa có ghi chú riêng tư."
                if not notes
                else f"Ghi chú gần nhất: {notes[0]['content']}"
            )
    except Exception as exc:
        return _finish(
            started_at,
            **common,
            speaker=speaker,
            success=False,
            error=f"DATABASE_ERROR: {exc}",
            response="Không thể truy cập dữ liệu người dùng.",
        )
    common["stage_latency_ms"]["database"] = (
        time.perf_counter() - database_started_at
    ) * 1000.0
    return _finish(started_at, **common, speaker=speaker, response=response)


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


def process_request(
    audio_path: str | Path | None = None,
    transcript: str | None = None,
    candidate_user_id: str = "user_001",
    verification_passed: bool = True,
    database_path: str | Path | None = None,
    *,
    intent: str | None = None,
    entities: dict | None = None,
    missing_fields: list[str] | None = None,
) -> dict:
    """Apply access policy and execute an already-parsed voice command.

    ``intent`` and ``entities`` come from the ASR/NLU pipeline.  Omitting them
    keeps the deterministic transcript parser available for demo tests.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"transcript": "", "intent": "OUT_OF_SCOPE", "entities": {}, "speaker": "MOCK_UNKNOWN", "similarity": None, "verification": None, "response": "ASR chưa sẵn sàng. Nhập mock transcript để thử luồng.", "policy": REJECT}

    if intent is None:
        intent, entities = _mock_parse(transcript)
    entities = entities or {}
    missing_fields = missing_fields or []
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
    elif intent == "ADD_SCHEDULE":
        if missing_fields or not all(entities.get(key) for key in ("title", "date", "time")):
            response = "Thiếu thông tin để thêm lịch: tiêu đề, ngày và giờ."
        else:
            schedule = add_schedule(
                candidate_user_id,
                entities["title"],
                entities["date"],
                entities["time"],
                database_path=database_path,
            )
            response = f"Đã thêm lịch {schedule['title']} lúc {schedule['time']}, {schedule['date']}."
    else:
        response = "Câu lệnh ngoài phạm vi hỗ trợ."

    return {"transcript": transcript, "intent": intent, "entities": entities, "speaker": speaker, "candidate_user_id": candidate_user_id if user else None, "similarity": similarity, "verification": verification, "response": response, "policy": policy, "audio_path": str(audio_path) if audio_path else None}
