"""HTTP backend serving speaker enrollment and voice-assistant requests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.database.user_repository import delete_user, list_users
from src.asr.whisper_model import get_asr_model
from src.pipeline.orchestrator import process_audio_request
from src.speaker.application import enroll_user
from src.speaker.embedding import get_embedding_extractor
from src.utils.config import load_yaml_mapping


CONFIG_PATH = Path("config.yaml")
SERVICE_NAME = "voicestudy-backend"
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
COPY_BLOCK_BYTES = 1024 * 1024

PipelineProcessor = Callable[..., dict[str, Any]]
Enroller = Callable[..., dict[str, Any]]
UserLister = Callable[..., list[dict[str, Any]]]
UserDeleter = Callable[..., bool]
ModelLoader = Callable[[str | Path], Any]


def _settings(config_path: str | Path = CONFIG_PATH) -> tuple[dict, dict]:
    config, _ = load_yaml_mapping(config_path)
    return config.get("backend", {}), config.get("speaker", {})


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


def _require_wav(upload: UploadFile) -> None:
    filename = upload.filename or ""
    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Only WAV audio is supported")


def _print_fields(fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        print(f"{key}: {rendered}", flush=True)


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
) -> FastAPI:
    """Build dependency-injectable FastAPI application."""

    backend_settings, speaker_settings = _settings(config_path)
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
        reference_date: str | None = Form(default=None),
    ) -> dict[str, Any]:
        _require_wav(audio)
        with TemporaryDirectory(prefix="voicestudy-api-command-") as directory:
            path = Path(directory) / "command.wav"
            _write_upload(audio, path, upload_limit)
            return pipeline_processor(
                path,
                reference_date=reference_date,
                config_path=config_path,
            )

    @application.post("/api/v1/enroll")
    def enroll_application_user(
        user_id: str = Form(...),
        name: str = Form(...),
        audio_files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        if len(audio_files) != 5:
            raise HTTPException(status_code=400, detail="Exactly 5 WAV files are required")
        for upload in audio_files:
            _require_wav(upload)
        with TemporaryDirectory(prefix="voicestudy-api-enrollment-") as directory:
            paths: list[Path] = []
            for index, upload in enumerate(audio_files, start=1):
                path = Path(directory) / f"{index}.wav"
                _write_upload(upload, path, upload_limit)
                paths.append(path)
            return enroller(
                user_id.strip(),
                name.strip(),
                paths,
                config_path=config_path,
            )

    @application.get("/api/v1/users")
    def get_users() -> list[dict[str, Any]]:
        return user_lister()

    @application.delete("/api/v1/users/{user_id}")
    def remove_user(user_id: str) -> dict[str, Any]:
        if not user_deleter(user_id):
            raise HTTPException(status_code=404, detail="User not found")
        return {"success": True, "user_id": user_id}

    return application


def run(config_path: str | Path = CONFIG_PATH) -> None:
    """Run Uvicorn using host/port values from YAML configuration."""

    import uvicorn

    backend_settings, _ = _settings(config_path)
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
