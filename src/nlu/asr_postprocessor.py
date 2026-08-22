"""Post-ASR command boundary detection and conservative content handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence, TypedDict
import re
import unicodedata

from rapidfuzz import fuzz

from src.asr.metrics import calculate_error_rates
from src.nlu.text_normalizer import normalize_text


class CommandSpec(TypedDict):
    intent: str
    patterns: tuple[str, ...]
    has_content: bool
    requires_secret: bool


@dataclass(frozen=True)
class ASRPostProcessorConfig:
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.65
    prefix_bonus: float = 4.0
    command_length_bonus: float = 0.8
    max_extra_prefix_tokens: int = 2
    content_normalization_level: int = 1
    log_private_content: bool = False


@dataclass
class ASRProcessingResult:
    raw_transcript: str
    intent: str | None
    intent_confidence: float
    detected_command_text: str | None
    normalized_command_text: str | None
    raw_content: str | None
    normalized_content: str | None
    final_content: str | None
    command_match_score: float | None
    requires_user_confirmation: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_user_edit(self, content: str) -> "ASRProcessingResult":
        self.final_content = _normalize_safe_content(content)
        return self

    def log_record(self, *, include_content: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "raw_asr": self.raw_transcript,
            "intent": self.intent,
            "command_match_score": self.command_match_score,
            "requires_user_confirmation": self.requires_user_confirmation,
            "metadata": dict(self.metadata),
        }
        if include_content:
            record.update(
                {
                    "raw_content": self.raw_content,
                    "normalized_content": self.normalized_content,
                    "final_content": self.final_content,
                }
            )
        return record


_DEFAULT_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("ghi trú", "ghi chú"),
    ("ghi chủ", "ghi chú"),
    ("ghi chỗ", "ghi chú"),
    ("riêng từ", "riêng tư"),
)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_command(text: str, confusions: Mapping[str, str]) -> str:
    normalized = normalize_text(text)
    for source, target in confusions.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _normalize_safe_content(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
    normalized = re.sub(r"([,.!?;:]){2,}", r"\1", normalized)
    if normalized and normalized[0].islower():
        normalized = normalized[0].upper() + normalized[1:]
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _token_prefix(tokens: Sequence[str], length: int) -> str:
    return " ".join(tokens[:length]).strip()


def _token_suffix(tokens: Sequence[str], start: int) -> str | None:
    suffix = " ".join(tokens[start:]).strip(" .,:;-")
    return suffix or None


def _raw_tokens(text: str) -> list[str]:
    return [
        token.strip(" .,:;-")
        for token in unicodedata.normalize("NFC", text).split()
        if token.strip(" .,:;-")
    ]


@dataclass(frozen=True)
class _Candidate:
    intent: str
    pattern: str
    detected_command_text: str
    raw_content: str | None
    confidence: float
    rank_score: float
    token_count: int
    has_content: bool
    requires_secret: bool


class ASRPostProcessor:
    def __init__(
        self,
        commands: Iterable[CommandSpec],
        *,
        config: ASRPostProcessorConfig | None = None,
        confusions: Mapping[str, str] | None = None,
    ) -> None:
        self.commands = tuple(commands)
        self.config = config or ASRPostProcessorConfig()
        self.confusions = dict(_DEFAULT_CONFUSIONS)
        if confusions:
            self.confusions.update(confusions)

    def process(
        self,
        transcript: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ASRProcessingResult:
        raw_transcript = transcript
        command_normalized = _normalize_for_command(transcript, self.confusions)
        raw_tokens = _raw_tokens(transcript)
        normalized_tokens = command_normalized.split()

        best = self._best_candidate(raw_tokens, normalized_tokens)
        if best is None or best.confidence < self.config.low_confidence_threshold:
            return ASRProcessingResult(
                raw_transcript=raw_transcript,
                intent=None,
                intent_confidence=0.0,
                detected_command_text=None,
                normalized_command_text=None,
                raw_content=None,
                normalized_content=None,
                final_content=None,
                command_match_score=best.confidence if best else None,
                requires_user_confirmation=False,
                metadata={"reason": "unknown_command"},
            )

        raw_content = best.raw_content
        normalized_content = (
            _normalize_safe_content(raw_content)
            if raw_content is not None and self.config.content_normalization_level >= 1
            else raw_content
        )
        requires_confirmation = best.confidence < self.config.high_confidence_threshold
        metadata: dict[str, Any] = {
            "matched_pattern": best.pattern,
            "has_content": best.has_content,
            "requires_secret": best.requires_secret,
        }
        if best.has_content and not raw_content:
            requires_confirmation = True
            metadata["reason"] = "missing_content"
        if context:
            metadata["context_keys"] = sorted(context.keys())

        return ASRProcessingResult(
            raw_transcript=raw_transcript,
            intent=best.intent,
            intent_confidence=round(best.confidence, 4),
            detected_command_text=best.detected_command_text,
            normalized_command_text=best.pattern,
            raw_content=raw_content,
            normalized_content=normalized_content,
            final_content=None,
            command_match_score=round(best.confidence, 4),
            requires_user_confirmation=requires_confirmation,
            metadata=metadata,
        )

    def _best_candidate(
        self,
        raw_tokens: Sequence[str],
        normalized_tokens: Sequence[str],
    ) -> _Candidate | None:
        candidates: list[_Candidate] = []
        for command in self.commands:
            for pattern in command["patterns"]:
                pattern_normalized = _normalize_for_command(pattern, self.confusions)
                pattern_len = len(pattern_normalized.split())
                for token_count in range(
                    pattern_len,
                    min(
                        len(normalized_tokens),
                        pattern_len + self.config.max_extra_prefix_tokens,
                    )
                    + 1,
                ):
                    prefix = _token_prefix(normalized_tokens, token_count)
                    if not prefix:
                        continue
                    score = fuzz.ratio(prefix, pattern_normalized)
                    confidence = score / 100.0
                    detected = _token_prefix(raw_tokens, token_count)
                    raw_content = (
                        _token_suffix(raw_tokens, token_count)
                        if command["has_content"]
                        else None
                    )
                    prefix_bonus = (
                        self.config.prefix_bonus
                        if normalized_tokens[:pattern_len] == pattern_normalized.split()
                        else 0.0
                    )
                    rank_score = (
                        score
                        + prefix_bonus
                        + pattern_len * self.config.command_length_bonus
                    )
                    candidates.append(
                        _Candidate(
                            intent=command["intent"],
                            pattern=pattern_normalized,
                            detected_command_text=detected,
                            raw_content=raw_content,
                            confidence=confidence,
                            rank_score=rank_score,
                            token_count=token_count,
                            has_content=command["has_content"],
                            requires_secret=command["requires_secret"],
                        )
                    )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                candidate.confidence >= self.config.low_confidence_threshold,
                candidate.rank_score,
                candidate.token_count,
            ),
        )


def evaluate_postprocessing(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    total = len(rows)
    intent_correct = 0
    command_correct = 0
    command_total = 0
    boundary_correct = 0
    boundary_total = 0
    unknown_correct = 0
    unknown_total = 0
    edit_required = 0
    cer_values: list[float] = []
    wer_values: list[float] = []

    for row in rows:
        expected_intent = row.get("expected_intent")
        actual_intent = row.get("intent")
        if expected_intent == actual_intent:
            intent_correct += 1
        if "expected_command" in row:
            command_total += 1
            if row.get("expected_command") == row.get("normalized_command_text"):
                command_correct += 1
        if "expected_content" in row:
            boundary_total += 1
            if row.get("expected_content") == row.get("raw_content"):
                boundary_correct += 1
        if expected_intent in {None, "UNKNOWN", "OUT_OF_SCOPE"}:
            unknown_total += 1
            if actual_intent in {None, "UNKNOWN", "OUT_OF_SCOPE"}:
                unknown_correct += 1
        normalized_content = row.get("normalized_content")
        final_content = row.get("final_content")
        if normalized_content and final_content:
            metrics = calculate_error_rates(str(final_content), str(normalized_content))
            cer_values.append(metrics["cer"])
            wer_values.append(metrics["wer"])
            if normalized_content != final_content:
                edit_required += 1

    return {
        "sample_count": total,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "command_match_accuracy": (
            command_correct / command_total if command_total else 0.0
        ),
        "boundary_accuracy": boundary_correct / boundary_total if boundary_total else 0.0,
        "unknown_rejection_accuracy": (
            unknown_correct / unknown_total if unknown_total else 0.0
        ),
        "content_cer": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "content_wer": sum(wer_values) / len(wer_values) if wer_values else 0.0,
        "normalizer_edit_rate": edit_required / total if total else 0.0,
    }
