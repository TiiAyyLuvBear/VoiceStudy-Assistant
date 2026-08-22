"""HTTP backend serving speaker enrollment and voice-assistant requests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.asr.whisper_model import get_asr_model, import_transformers_asr, transcribe_audio
from src.database.user_repository import delete_user, get_user, list_users
from src.nlu.command_catalog import fixed_command_catalog
from src.pipeline.orchestrator import process_audio_request
from src.security.secret_phrase import normalize_secret_phrase
from src.speaker.application import enroll_user
from src.speaker.enrollment_quality import DEFAULT_ENROLLMENT_PROMPTS
from src.speaker.embedding import get_embedding_extractor
from src.tasks.note_tasks import add_note, delete_note, get_notes
from src.tasks.schedule_tasks import add_schedule, delete_schedule, get_schedules
from src.tts.text_to_speech import synthesize_vietnamese
from src.utils.config import load_yaml_mapping


CONFIG_PATH = Path("config.yaml")
SERVICE_NAME = "voicestudy-backend"
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
COPY_BLOCK_BYTES = 1024 * 1024
ENROLLMENT_FORM_CACHE_LIMIT = 64
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac"}

PipelineProcessor = Callable[..., dict[str, Any]]
Enroller = Callable[..., dict[str, Any]]
UserLister = Callable[..., list[dict[str, Any]]]
UserDeleter = Callable[..., bool]
ModelLoader = Callable[[str | Path], Any]
TtsSynthesizer = Callable[[str], bytes | None]
SecretTranscriber = Callable[[str | Path, str | Path], dict[str, Any]]


class ScheduleCreate(BaseModel):
    title: str = Field(min_length=1)
    date: str = Field(min_length=1)
    time: str = Field(min_length=1)
    description: str | None = None


class NoteCreate(BaseModel):
    content: str = Field(min_length=1)
    is_private: bool = True


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)


def _settings(config_path: str | Path = CONFIG_PATH) -> tuple[dict, dict, dict]:
    config, _ = load_yaml_mapping(config_path)
    return config.get("backend", {}), config.get("speaker", {}), config.get("asr", {})


def _write_upload(upload: UploadFile, destination: Path, maximum_bytes: int) -> None:
    written = 0
    with destination.open("wb") as stream:
        while True:
            block = upload.file.read(COPY_BLOCK_BYTES)
            if not block:
                break
            written += len(block)
            if written > maximum_bytes:
                raise HTTPException(status_code=413, detail="Uploaded audio is too large")
            stream.write(block)


def _upload_suffix(upload: UploadFile) -> str:
    filename = upload.filename or ""
    return Path(filename).suffix.lower()


def _require_supported_audio(upload: UploadFile) -> None:
    if _upload_suffix(upload) not in SUPPORTED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only WAV or FLAC audio is supported")


def _enrollment_audio_limits(speaker_settings: dict[str, Any]) -> tuple[int, int]:
    default_count = int(speaker_settings.get("enrollment_audio_count", 5))
    minimum = int(speaker_settings.get("min_enrollment_audio_count", min(3, default_count)))
    maximum = int(speaker_settings.get("max_enrollment_audio_count", max(10, default_count)))
    return minimum, maximum


def _render_log_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def _print_field(key: str, value: Any) -> None:
    if isinstance(value, dict):
        if not value:
            print(f"{key}: {{}}", flush=True)
            return
        for child_key, child_value in value.items():
            _print_field(f"{key}.{child_key}", child_value)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            print(f"{key}: []", flush=True)
            return
        for index, item in enumerate(value, start=1):
            _print_field(f"{key}.{index}", item)
        return
    print(f"{key}: {_render_log_value(value)}", flush=True)


def _print_fields(fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        _print_field(key, value)


def _enrollment_upload_metadata(
    uploads: list[UploadFile],
    prompts: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, upload in enumerate(uploads, start=1):
        rows.append(
            {
                "index": index,
                "filename": upload.filename or "",
                "sample_prompt": prompts[index - 1] if index <= len(prompts) else None,
            }
        )
    return rows


def _cache_enrollment_form(
    cache: dict[str, dict[str, Any]],
    *,
    user_id: str,
    username: str,
    name: str,
    prompts: list[str],
    prompts_source: str,
    client_prompt_count: int,
    audio_files: list[UploadFile],
    secret_audio: UploadFile,
) -> None:
    cache[user_id] = {
        "updated_at": time.time(),
        "user_id": user_id,
        "username": username,
        "name": name,
        "prompt_count": len(prompts),
        "prompts_source": prompts_source,
        "client_prompt_count": client_prompt_count,
        "prompts": prompts,
        "audio_count": len(audio_files),
        "audio_files": _enrollment_upload_metadata(audio_files, prompts),
        "secret_audio_filename": secret_audio.filename or "",
        "status": "received",
        "last_error": None,
    }
    while len(cache) > ENROLLMENT_FORM_CACHE_LIMIT:
        oldest_key = min(cache, key=lambda key: float(cache[key].get("updated_at", 0.0)))
        cache.pop(oldest_key, None)


def _update_enrollment_cache_result(
    cache: dict[str, dict[str, Any]],
    user_id: str,
    result: dict[str, Any],
) -> None:
    current = cache.setdefault(
        user_id,
        {"updated_at": time.time(), "user_id": user_id, "status": "unknown"},
    )
    current.update(
        {
            "updated_at": time.time(),
            "status": "success" if result.get("success") else "failed",
            "last_error": result.get("error"),
            "message_vi": result.get("message_vi"),
            "failed_stage": result.get("failed_stage"),
            "failed_sample_index": result.get("failed_sample_index"),
            "sample_prompt": result.get("sample_prompt"),
            "speaker_model": result.get("speaker_model"),
            "asr_model": result.get("asr_model"),
            "embedding_consistency": result.get("embedding_consistency"),
        }
    )


def _log_enrollment_received(user_id: str, cache_row: dict[str, Any]) -> None:
    _print_fields(
        {
            "event": "enrollment_received",
            "user_id": user_id,
            "prompt_count": cache_row.get("prompt_count"),
            "prompts_source": cache_row.get("prompts_source"),
            "client_prompt_count": cache_row.get("client_prompt_count"),
            "audio_count": cache_row.get("audio_count"),
            "audio_files": cache_row.get("audio_files", []),
            "secret_audio_filename": cache_row.get("secret_audio_filename", ""),
        }
    )


def _log_enrollment_result(result: dict[str, Any]) -> None:
    fields = {
        "event": "enrollment_succeeded" if result.get("success") else "enrollment_failed",
        "user_id": result.get("user_id"),
        "error": result.get("error"),
        "message_vi": result.get("message_vi"),
        "failed_stage": result.get("failed_stage"),
        "failed_sample_index": result.get("failed_sample_index"),
        "sample_prompt": result.get("sample_prompt"),
        "speaker_model": result.get("speaker_model"),
        "asr_model": result.get("asr_model"),
    }
    embedding_consistency = result.get("embedding_consistency")
    if embedding_consistency:
        fields["embedding_consistency"] = embedding_consistency
    file_results = result.get("file_results")
    if file_results:
        fields["file_results"] = file_results
    _print_fields(fields)


def _require_user(user_id: str) -> None:
    if not get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")


def _enrollment_error_vi(error: str | None) -> str:
    messages = {
        "INVALID_USER_ID": "Mã người dùng không hợp lệ. Hãy sửa lại mã người dùng.",
        "INVALID_NAME": "Tên hiển thị không hợp lệ. Hãy nhập lại tên.",
        "SECRET_PHRASE_TOO_SHORT": "Câu bí mật cần ít nhất 3 từ. Hãy sửa transcript câu bí mật.",
        "INVALID_SECRET_PHRASE": "Câu bí mật không hợp lệ. Hãy sửa transcript câu bí mật.",
        "SECRET_PHRASE_ASR_FAILED": "Không nhận diện được audio câu bí mật. Hãy đọc lại câu bí mật.",
        "SECRET_PHRASE_TRANSCRIPT_MISMATCH": "Audio câu bí mật không khớp transcript đã nhập. Hãy sửa transcript hoặc đọc lại câu bí mật.",
        "INVALID_ENROLLMENT_PROMPTS": "Danh sách câu mẫu đăng kí không đúng. Hãy tải lại trang và thử lại.",
        "INVALID_ENROLLMENT_AUDIO_COUNT": "Cần 3-10 audio đăng kí. Hãy bổ sung đủ mẫu voice.",
        "DUPLICATE_ENROLLMENT_AUDIO": "Có audio đăng kí bị trùng. Hãy thu lại mẫu bị trùng.",
        "INVALID_AUDIO": "File voice không hợp lệ. Hãy đọc lại hoặc upload lại file WAV/FLAC.",
        "AUDIO_QUALITY_FAILED": "Chất lượng voice chưa đạt. Hãy thu lại mẫu voice bị lỗi, rõ hơn và ít nhiễu hơn.",
        "INVALID_EMBEDDING": "Không tạo được speaker embedding từ một mẫu voice. Hãy thu lại mẫu bị lỗi.",
        "VOICE_INCONSISTENT_WITH_OTHER_SAMPLES": "Một mẫu voice lệch nhiều so với các mẫu còn lại. Hãy nghe lại và thu lại mẫu đó.",
        "INSUFFICIENT_ACCEPTED_SAMPLES": "Chưa đủ mẫu voice hợp lệ để tạo speaker template. Hãy thu thêm hoặc thu lại các mẫu bị lỗi.",
        "EMBEDDING_CONSISTENCY_FAILED": "Các mẫu voice chưa đủ nhất quán cho ECAPA. Hãy nghe lại từng mẫu, bỏ mẫu bị rè/xa mic/khác môi trường rồi thu lại.",
        "CENTROID_WRITE_FAILED": "Không lưu được hồ sơ giọng nói. Hãy thử lại.",
        "INVALID_SPEAKER_CONFIG": "Cấu hình ECAPA fine-tune không hợp lệ. Hãy kiểm tra backend.",
    }
    return messages.get(str(error or ""), "Đăng kí thất bại. Hãy kiểm tra bước bị lỗi và thử lại.")


def _failed_sample_index(result: dict[str, Any]) -> int | None:
    for item in result.get("file_results") or []:
        if item.get("valid") is False:
            stem = Path(str(item.get("audio_path") or "")).stem
            if stem.isdigit():
                return int(stem)
    return None


def _failed_file_result(result: dict[str, Any]) -> dict[str, Any] | None:
    for item in result.get("file_results") or []:
        if item.get("valid") is False:
            return item if isinstance(item, dict) else None
    return None


def _failed_quality_message_vi(result: dict[str, Any]) -> str | None:
    failed = _failed_file_result(result)
    if not failed:
        return None
    message = failed.get("message_vi")
    if message:
        return str(message)
    quality = failed.get("quality")
    if isinstance(quality, dict) and quality.get("message_vi"):
        return str(quality["message_vi"])
    return None


def _enrollment_failure(
    *,
    user_id: str,
    error: str,
    failed_stage: str,
    secret_phrase_transcript: str | None = None,
    failed_sample_index: int | None = None,
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    prompt = (
        prompts[failed_sample_index - 1]
        if prompts and failed_sample_index and 1 <= failed_sample_index <= len(prompts)
        else None
    )
    return {
        "protocol": "APPLICATION_ENROLLMENT",
        "success": False,
        "user_id": user_id,
        "error": error,
        "message_vi": _enrollment_error_vi(error),
        "failed_stage": failed_stage,
        "failed_sample_index": failed_sample_index,
        "failed_prompt": prompt,
        "sample_prompt": prompt,
        "secret_phrase_transcript": secret_phrase_transcript,
    }


def _enrich_enrollment_result(
    result: dict[str, Any],
    *,
    prompts: list[str],
    speaker_model: str,
    asr_model: str,
) -> dict[str, Any]:
    enriched = dict(result)
    enriched["speaker_model"] = speaker_model
    enriched["asr_model"] = asr_model
    enriched["enrollment_prompts"] = prompts
    if enriched.get("success"):
        enriched["message_vi"] = "Đăng kí thành công."
        return enriched
    error = str(enriched.get("error") or "")
    failed_stage = "server"
    failed_index = _failed_sample_index(enriched)
    if error in {"INVALID_USER_ID", "INVALID_NAME"}:
        failed_stage = "profile"
    elif error in {"SECRET_PHRASE_TOO_SHORT", "INVALID_SECRET_PHRASE"}:
        failed_stage = "secret_phrase"
    elif error in {
        "AUDIO_QUALITY_FAILED",
        "INVALID_AUDIO",
        "DUPLICATE_ENROLLMENT_AUDIO",
        "INVALID_EMBEDDING",
        "VOICE_INCONSISTENT_WITH_OTHER_SAMPLES",
    }:
        failed_stage = "speaker_sample"
    elif error in {"EMBEDDING_CONSISTENCY_FAILED", "INSUFFICIENT_ACCEPTED_SAMPLES"}:
        failed_stage = "speaker_samples"
    elif error in {"INVALID_ENROLLMENT_AUDIO_COUNT", "INVALID_ENROLLMENT_PROMPTS"}:
        failed_stage = "speaker_samples"
    enriched["message_vi"] = (
        _failed_quality_message_vi(enriched)
        if error in {"AUDIO_QUALITY_FAILED", "INVALID_EMBEDDING", "VOICE_INCONSISTENT_WITH_OTHER_SAMPLES"}
        else None
    ) or _enrollment_error_vi(error)
    enriched["failed_stage"] = failed_stage
    enriched["failed_sample_index"] = failed_index
    if failed_index and 1 <= failed_index <= len(prompts):
        enriched["failed_prompt"] = prompts[failed_index - 1]
        enriched["sample_prompt"] = prompts[failed_index - 1]
    return enriched


def create_app(
    *,
    pipeline_processor: PipelineProcessor = process_audio_request,
    enroller: Enroller = enroll_user,
    user_lister: UserLister = list_users,
    user_deleter: UserDeleter = delete_user,
    config_path: str | Path = CONFIG_PATH,
    max_upload_bytes: int | None = None,
    preload_models: bool | None = None,
    speaker_loader: ModelLoader = get_embedding_extractor,
    asr_loader: ModelLoader = get_asr_model,
    tts_synthesizer: TtsSynthesizer = synthesize_vietnamese,
    secret_transcriber: SecretTranscriber = transcribe_audio,
) -> FastAPI:
    """Build dependency-injectable FastAPI application."""

    backend_settings, speaker_settings, asr_settings = _settings(config_path)
    upload_limit = int(
        max_upload_bytes
        if max_upload_bytes is not None
        else backend_settings.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES)
    )
    model_version = str(speaker_settings.get("model_version", "unknown"))
    should_preload = bool(
        backend_settings.get("preload_models", True)
        if preload_models is None
        else preload_models
    )
    strict_startup = bool(backend_settings.get("strict_model_startup", True))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        fields: dict[str, Any] = {
            "event": "backend_startup",
            "service": SERVICE_NAME,
            "preload_models": should_preload,
            "speaker_loaded": False,
            "asr_loaded": False,
        }
        startup_error: Exception | None = None
        if should_preload:
            if str(asr_settings.get("backend", "faster-whisper")) == "transformers":
                try:
                    import_transformers_asr()
                    fields["asr_dependencies_loaded"] = True
                except Exception as error:
                    startup_error = error
                    fields["asr_dependencies_loaded"] = False
                    fields["asr_error"] = str(error)
            try:
                speaker = speaker_loader(config_path)
                fields.update(
                    {
                        "speaker_loaded": True,
                        "speaker_model": getattr(speaker, "model_version", model_version),
                        "speaker_epoch": getattr(speaker, "checkpoint_metadata", {}).get("epoch"),
                        "speaker_device": getattr(speaker, "device", "unknown"),
                    }
                )
            except Exception as error:
                startup_error = error
                fields["speaker_error"] = str(error)
            try:
                asr = asr_loader(config_path)
                asr.load_model()
                asr_config = getattr(asr, "config", None)
                fields.update(
                    {
                        "asr_loaded": True,
                        "asr_model": getattr(asr_config, "model_name", "unknown"),
                        "asr_device": getattr(asr_config, "device", "unknown"),
                        "asr_compute_type": getattr(asr_config, "compute_type", "unknown"),
                    }
                )
            except Exception as error:
                startup_error = startup_error or error
                if "asr_error" not in fields:
                    fields["asr_error"] = str(error)
        speaker_device = fields.get("speaker_device")
        asr_device = fields.get("asr_device")
        if speaker_device and speaker_device == asr_device:
            fields["device"] = speaker_device
        _print_fields(fields)
        if startup_error is not None and strict_startup:
            raise RuntimeError("Configured backend model preload failed") from startup_error
        yield

    application = FastAPI(
        title="VoiceStudy Assistant Backend",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    enrollment_form_cache: dict[str, dict[str, Any]] = {}
    application.state.enrollment_form_cache = enrollment_form_cache

    @application.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "model_version": model_version,
        }

    @application.post("/api/v1/process")
    def process_pipeline(
        audio: UploadFile = File(...),
        secret_audio: UploadFile | None = File(default=None),
        reference_date: str | None = Form(default=None),
    ) -> dict[str, Any]:
        _require_supported_audio(audio)
        if secret_audio is not None:
            _require_supported_audio(secret_audio)
        with TemporaryDirectory(prefix="voicestudy-api-command-") as directory:
            path = Path(directory) / f"command{_upload_suffix(audio)}"
            _write_upload(audio, path, upload_limit)
            secret_path = None
            if secret_audio is not None:
                secret_path = Path(directory) / f"secret{_upload_suffix(secret_audio)}"
                _write_upload(secret_audio, secret_path, upload_limit)
            return pipeline_processor(
                path,
                secret_audio_path=secret_path,
                reference_date=reference_date,
                config_path=config_path,
            )

    @application.get("/api/v1/commands")
    def get_commands() -> list[dict[str, Any]]:
        return fixed_command_catalog()

    @application.post("/api/v1/tts")
    def synthesize_response_speech(payload: TtsRequest) -> Response:
        audio = tts_synthesizer(payload.text.strip())
        if audio is None:
            raise HTTPException(status_code=503, detail="Vietnamese TTS is unavailable")
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/v1/enrollment-cache/{user_id}")
    def get_enrollment_form_cache(user_id: str) -> dict[str, Any]:
        cached = enrollment_form_cache.get(user_id)
        if cached is None:
            raise HTTPException(status_code=404, detail="Enrollment form cache not found")
        return cached

    @application.post("/api/v1/enroll")
    def enroll_application_user(
        username: str | None = Form(default=None),
        user_id: str | None = Form(default=None),
        name: str = Form(...),
        secret_phrase: str = Form(...),
        secret_phrase_transcript: str | None = Form(default=None),
        secret_audio: UploadFile = File(...),
        enrollment_prompts: list[str] | None = Form(default=None),
        audio_files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        raw_username = (username or user_id or "").strip()
        if not raw_username:
            raise HTTPException(status_code=422, detail="Username is required")
        normalized_username = re.sub(r"[^A-Za-z0-9_-]", "_", raw_username.removeprefix("user_"))
        resolved_user_id = f"user_{normalized_username}"
        min_audio_count, max_audio_count = _enrollment_audio_limits(speaker_settings)
        if not (min_audio_count <= len(audio_files) <= max_audio_count):
            raise HTTPException(
                status_code=400,
                detail=f"Enrollment requires {min_audio_count}-{max_audio_count} audio files",
            )
        prompts_source = "client" if enrollment_prompts is not None else "default"
        client_prompt_count = len(enrollment_prompts or [])
        prompts = enrollment_prompts or list(DEFAULT_ENROLLMENT_PROMPTS)
        if len(prompts) != len(DEFAULT_ENROLLMENT_PROMPTS):
            raise HTTPException(status_code=400, detail="Exactly 5 enrollment prompts are required")
        _cache_enrollment_form(
            enrollment_form_cache,
            user_id=resolved_user_id,
            username=normalized_username,
            name=name.strip(),
            prompts=prompts,
            prompts_source=prompts_source,
            client_prompt_count=client_prompt_count,
            audio_files=audio_files,
            secret_audio=secret_audio,
        )
        _log_enrollment_received(resolved_user_id, enrollment_form_cache[resolved_user_id])
        asr_model_name = str(asr_settings.get("model_name", "unknown"))
        typed_secret_phrase = (secret_phrase_transcript or secret_phrase).strip()
        if secret_phrase_transcript is not None and (
            normalize_secret_phrase(secret_phrase_transcript)
            != normalize_secret_phrase(secret_phrase)
        ):
            raise HTTPException(
                status_code=400,
                detail="Secret phrase and typed transcript must match",
            )
        _require_supported_audio(secret_audio)
        for upload in audio_files:
            _require_supported_audio(upload)
        with TemporaryDirectory(prefix="voicestudy-api-enrollment-") as directory:
            secret_path = Path(directory) / f"secret{_upload_suffix(secret_audio)}"
            _write_upload(secret_audio, secret_path, upload_limit)
            secret_asr = secret_transcriber(secret_path, config_path)
            raw_secret_transcript = str(secret_asr.get("transcript") or "")
            if not secret_asr.get("success"):
                result = _enrollment_failure(
                    user_id=resolved_user_id,
                    error="SECRET_PHRASE_ASR_FAILED",
                    failed_stage="secret_audio",
                    secret_phrase_transcript=raw_secret_transcript,
                    prompts=prompts,
                ) | {"speaker_model": model_version, "asr_model": asr_model_name}
                _update_enrollment_cache_result(enrollment_form_cache, resolved_user_id, result)
                _log_enrollment_result(result)
                return result
            if normalize_secret_phrase(raw_secret_transcript) != normalize_secret_phrase(typed_secret_phrase):
                result = _enrollment_failure(
                    user_id=resolved_user_id,
                    error="SECRET_PHRASE_TRANSCRIPT_MISMATCH",
                    failed_stage="secret_audio",
                    secret_phrase_transcript=raw_secret_transcript,
                    prompts=prompts,
                ) | {"speaker_model": model_version, "asr_model": asr_model_name}
                _update_enrollment_cache_result(enrollment_form_cache, resolved_user_id, result)
                _log_enrollment_result(result)
                return result
            paths: list[Path] = []
            for index, upload in enumerate(audio_files, start=1):
                path = Path(directory) / f"{index}{_upload_suffix(upload)}"
                _write_upload(upload, path, upload_limit)
                paths.append(path)
            result = enroller(
                resolved_user_id,
                name.strip(),
                paths,
                secret_phrase=typed_secret_phrase,
                enrollment_prompts=prompts,
                config_path=config_path,
            )
            enriched = _enrich_enrollment_result(
                result,
                prompts=prompts,
                speaker_model=model_version,
                asr_model=asr_model_name,
            )
            _update_enrollment_cache_result(enrollment_form_cache, resolved_user_id, enriched)
            _log_enrollment_result(enriched)
            return enriched

    @application.get("/api/v1/users")
    def get_users() -> list[dict[str, Any]]:
        return user_lister()

    @application.delete("/api/v1/users/{user_id}")
    def remove_user(user_id: str) -> dict[str, Any]:
        if not user_deleter(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"success": True, "user_id": user_id}

    @application.get("/api/v1/users/{user_id}/schedules")
    def list_user_schedules(user_id: str, date: str | None = None) -> list[dict[str, Any]]:
        _require_user(user_id)
        return get_schedules(user_id, date)

    @application.post("/api/v1/users/{user_id}/schedules")
    def create_user_schedule(user_id: str, payload: ScheduleCreate) -> dict[str, Any]:
        _require_user(user_id)
        return add_schedule(
            user_id,
            payload.title.strip(),
            payload.date.strip(),
            payload.time.strip(),
            payload.description.strip() if payload.description else None,
        )

    @application.delete("/api/v1/users/{user_id}/schedules/{schedule_id}")
    def remove_user_schedule(user_id: str, schedule_id: int) -> dict[str, Any]:
        _require_user(user_id)
        if not delete_schedule(user_id, schedule_id):
            raise HTTPException(status_code=404, detail="Schedule not found")
        return {"success": True, "user_id": user_id, "schedule_id": schedule_id}

    @application.get("/api/v1/users/{user_id}/notes")
    def list_user_notes(user_id: str) -> list[dict[str, Any]]:
        _require_user(user_id)
        return get_notes(user_id)

    @application.post("/api/v1/users/{user_id}/notes")
    def create_user_note(user_id: str, payload: NoteCreate) -> dict[str, Any]:
        _require_user(user_id)
        return add_note(user_id, payload.content.strip(), payload.is_private)

    @application.delete("/api/v1/users/{user_id}/notes/{note_id}")
    def remove_user_note(user_id: str, note_id: int) -> dict[str, Any]:
        _require_user(user_id)
        if not delete_note(user_id, note_id):
            raise HTTPException(status_code=404, detail="Note not found")
        return {"success": True, "user_id": user_id, "note_id": note_id}

    return application


def run(config_path: str | Path = CONFIG_PATH) -> None:
    """Run Uvicorn using host/port values from YAML configuration."""

    import uvicorn

    backend_settings, _, _ = _settings(config_path)
    uvicorn.run(
        "backend.main:app",
        host=str(backend_settings.get("host", "127.0.0.1")),
        port=int(backend_settings.get("port", 8000)),
        reload=bool(backend_settings.get("reload", False)),
        workers=1,
    )


app = create_app()


if __name__ == "__main__":
    run()
