"""Enrollment prompts, audio quality metrics, and embedding consistency checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioQualityMetrics:
    duration_sec: float
    speech_duration_sec: float
    speech_ratio: float
    rms_energy: float
    clipping_ratio: float
    estimated_snr_db: float | None
    sample_rate: int
    num_channels: int
    passed: bool
    rejection_reasons: list[str]


@dataclass(frozen=True)
class EnrollmentSampleResult:
    sample_id: str
    quality: AudioQualityMetrics
    embedding: np.ndarray | None
    centroid_similarity: float | None
    accepted: bool
    rejection_reasons: list[str]

DEFAULT_ENROLLMENT_PROMPTS: tuple[str, ...] = (
    "Bây giờ là mấy giờ rồi?",
    "Cho tôi xem lịch học ngày mai.",
    "Thêm lịch học thống kê lúc tám giờ sáng.",
    "Mở ghi chú riêng tư gần nhất của tôi.",
    "Hôm nay tôi cần kiểm tra lịch học và chuẩn bị bài thuyết trình.",
)

QUALITY_ISSUE_MESSAGES_VI: dict[str, str] = {
    "non_finite_audio": "File voice lỗi dữ liệu. Hãy thu lại hoặc upload lại file WAV/FLAC.",
    "too_short": "Voice quá ngắn. Hãy đọc lâu hơn, tối thiểu 2 giây.",
    "too_long": "Voice quá dài. Hãy đọc ngắn gọn hơn, tối đa 8 giây.",
    "too_quiet": "Voice quá nhỏ. Hãy đưa mic gần hơn hoặc nói rõ hơn.",
    "clipped": "Voice bị vỡ tiếng do âm lượng quá lớn. Hãy nói nhỏ hơn hoặc để mic xa hơn.",
    "too_much_silence": "Voice có quá nhiều khoảng lặng. Hãy bấm thu rồi đọc ngay, dừng khi đọc xong.",
    "insufficient_speech": "Voice có quá ít phần lời nói. Hãy đọc rõ ít nhất 2 giây.",
    "low_speech_ratio": "Voice có quá ít phần lời nói so với toàn file. Hãy cắt bớt khoảng lặng hoặc thu lại.",
    "low_audio_energy": "Voice quá nhỏ hoặc gần như im lặng. Hãy đưa mic gần hơn và nói rõ hơn.",
    "audio_clipping": "Voice bị vỡ tiếng do âm lượng quá lớn. Hãy nói nhỏ hơn hoặc để mic xa hơn.",
    "embedding_outlier": "Mẫu voice lệch nhiều so với các mẫu còn lại. Hãy nghe lại và thu lại mẫu này.",
}


def quality_issues_vi(issues: Sequence[str]) -> list[str]:
    return [
        QUALITY_ISSUE_MESSAGES_VI.get(str(issue), f"Lỗi chất lượng voice: {issue}.")
        for issue in issues
    ]


def quality_message_vi(issues: Sequence[str]) -> str:
    messages = quality_issues_vi(issues)
    return messages[0] if messages else "Voice đạt chuẩn chất lượng."


def validate_enrollment_prompts(prompts: Sequence[str] | None) -> tuple[bool, str | None]:
    if prompts is None:
        return True, None
    cleaned = [str(value).strip() for value in prompts]
    if tuple(cleaned) != DEFAULT_ENROLLMENT_PROMPTS:
        return False, "INVALID_ENROLLMENT_PROMPTS"
    return True, None


def analyze_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
) -> dict:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    settings = settings or {}
    metrics = metrics or {}
    duration = float(metrics.get("duration_seconds", 0.0) or 0.0)
    if duration <= 0.0:
        duration = float(values.size / sample_rate) if sample_rate > 0 else 0.0
    speech_duration = float(metrics.get("speech_duration_seconds", duration) or 0.0)
    speech_ratio = float(metrics.get("speech_ratio", 1.0 if duration > 0 else 0.0) or 0.0)
    finite = bool(values.size and np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(metrics.get("rms", 0.0) or 0.0)
    if rms <= 0.0 and values.size:
        rms = float(np.sqrt(np.mean(np.square(values))))
    clipping_ratio = float(metrics.get("clipping_ratio", 0.0) or 0.0)
    if clipping_ratio <= 0.0 and values.size:
        clipping_ratio = float(np.mean(np.abs(values) >= 0.99))
    speech_threshold = peak * float(settings.get("silence_peak_ratio", 0.03))
    silence_ratio = (
        float(np.mean(np.abs(values) < speech_threshold))
        if values.size and peak > 0.0
        else 1.0
    )
    metrics = {
        "duration_seconds": duration,
        "speech_duration_seconds": speech_duration,
        "speech_ratio": speech_ratio,
        "peak": peak,
        "rms": rms,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "estimated_snr_db": metrics.get("estimated_snr_db"),
        "sample_rate": int(metrics.get("sample_rate", sample_rate) or sample_rate),
        "num_channels": int(metrics.get("num_channels", 1) or 1),
    }
    issues: list[str] = []
    if not finite:
        issues.append("non_finite_audio")
    if not bool(settings.get("enabled", False)):
        return {
            "valid": not issues,
            "metrics": metrics,
            "issues": issues,
            "issues_vi": quality_issues_vi(issues),
            "message_vi": quality_message_vi(issues),
        }
    if duration < float(settings.get("min_duration_seconds", 2.0)):
        issues.append("too_short")
    if duration > float(settings.get("max_duration_seconds", 8.0)):
        issues.append("too_long")
    if speech_duration < float(settings.get("min_speech_duration_seconds", 2.0)):
        issues.append("insufficient_speech")
    if speech_ratio < float(settings.get("min_speech_ratio", 0.25)):
        issues.append("low_speech_ratio")
    if rms < float(settings.get("min_rms", 0.01)):
        issues.append("low_audio_energy")
    if clipping_ratio > float(settings.get("max_clipping_ratio", 0.02)):
        issues.append("audio_clipping")
    if silence_ratio > float(settings.get("max_silence_ratio", 0.65)):
        issues.append("too_much_silence")
    return {
        "valid": not issues,
        "metrics": metrics,
        "issues": issues,
        "issues_vi": quality_issues_vi(issues),
        "message_vi": quality_message_vi(issues),
    }


def embedding_consistency(
    embeddings: Sequence[np.ndarray],
    settings: Mapping[str, object] | None = None,
) -> dict:
    settings = settings or {}
    if len(embeddings) < 2:
        return {"valid": False, "mean_pairwise_cosine": None, "min_pairwise_cosine": None}
    vectors = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in embeddings]
    scores: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            scores.append(float(np.dot(left, right)))
    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    score_std = float(np.std(scores))
    centroid = l2_normalize(np.mean(np.stack(vectors), axis=0))
    centroid_similarities = [float(np.dot(vector, centroid)) for vector in vectors]
    mean_centroid_similarity = float(np.mean(centroid_similarities))
    min_centroid_similarity = float(np.min(centroid_similarities))
    enabled = bool(settings.get("enabled", False))
    valid = True
    valid_by_pairwise = True
    valid_by_centroid = True
    if enabled:
        valid_by_pairwise = (
            mean_score >= float(settings.get("min_mean_pairwise_cosine", 0.70))
            and min_score >= float(settings.get("min_pairwise_cosine", 0.45))
        )
        valid_by_centroid = (
            mean_centroid_similarity >= float(settings.get("min_mean_centroid_similarity", 0.70))
            and min_centroid_similarity >= float(settings.get("min_centroid_similarity", 0.45))
        )
        valid = valid_by_pairwise or valid_by_centroid
    return {
        "valid": valid,
        "valid_by_pairwise": valid_by_pairwise,
        "valid_by_centroid": valid_by_centroid,
        "mean_pairwise_cosine": mean_score,
        "min_pairwise_cosine": min_score,
        "pairwise_similarity_std": score_std,
        "mean_centroid_similarity": mean_centroid_similarity,
        "min_centroid_similarity": min_centroid_similarity,
        "centroid_similarities": centroid_similarities,
    }


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.size == 0 or not np.isfinite(values).all() or norm <= np.finfo(np.float32).eps:
        raise ValueError("Embedding is empty, non-finite, or zero-norm")
    return values / norm


def centroid_similarities(embeddings: Sequence[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    vectors = [l2_normalize(np.asarray(vector, dtype=np.float32)) for vector in embeddings]
    centroid = l2_normalize(np.mean(np.stack(vectors), axis=0))
    return centroid, [float(vector @ centroid) for vector in vectors]
