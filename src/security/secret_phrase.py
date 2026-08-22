"""Secret phrase hashing and transcript verification for private actions."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from pathlib import Path

from src.database.user_repository import get_user

_MARKER_PATTERN = re.compile(
    r"\b(?:mật khẩu|câu bí mật|lệnh bí mật|khẩu lệnh|secret phrase|passphrase)\b",
    re.IGNORECASE,
)
_SPACE_PATTERN = re.compile(r"\s+")
MIN_SECRET_WORDS = 3


def normalize_secret_phrase(value: str) -> str:
    """Normalize a user secret phrase before hashing or matching."""

    normalized = unicodedata.normalize("NFC", str(value or "")).strip().lower()
    normalized = re.sub(r"[^\w\sÀ-ỹ-]", " ", normalized, flags=re.UNICODE)
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def validate_secret_phrase(value: str) -> tuple[bool, str | None]:
    normalized = normalize_secret_phrase(value)
    if len(normalized.split()) < MIN_SECRET_WORDS:
        return False, "SECRET_PHRASE_TOO_SHORT"
    return True, None


def hash_secret_phrase(value: str, salt: str | None = None) -> tuple[str, str]:
    valid, error = validate_secret_phrase(value)
    if not valid:
        raise ValueError(error or "INVALID_SECRET_PHRASE")
    secret_salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        normalize_secret_phrase(value).encode("utf-8"),
        secret_salt.encode("ascii"),
        120_000,
    ).hex()
    return digest, secret_salt


def verify_secret_phrase(value: str, digest: str | None, salt: str | None) -> bool:
    if not digest or not salt:
        return False
    try:
        candidate, _ = hash_secret_phrase(value, salt)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, digest)


def extract_secret_phrase(transcript: str) -> str | None:
    """Extract phrase after a spoken marker, e.g. 'mật khẩu hoa sen xanh'."""

    match = None
    for match in _MARKER_PATTERN.finditer(transcript or ""):
        pass
    if match is None:
        return None
    phrase = str(transcript)[match.end() :]
    phrase = re.split(r"[.?!,;:]", phrase, maxsplit=1)[0]
    normalized = normalize_secret_phrase(phrase)
    return normalized or None


def verify_transcript_secret_phrase(
    user_id: str,
    transcript: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[bool, str | None]:
    user = get_user(user_id, database_path)
    if not user or not user.get("secret_phrase_hash") or not user.get("secret_phrase_salt"):
        return False, "SECRET_PHRASE_NOT_CONFIGURED"
    phrase = extract_secret_phrase(transcript)
    if phrase is None:
        return False, "SECRET_PHRASE_REQUIRED"
    if not verify_secret_phrase(
        phrase,
        str(user.get("secret_phrase_hash")),
        str(user.get("secret_phrase_salt")),
    ):
        return False, "SECRET_PHRASE_FAILED"
    return True, None


def verify_spoken_secret_phrase(
    user_id: str,
    phrase: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Verify audio-ASR output that contains only the secret phrase."""

    user = get_user(user_id, database_path)
    if not user or not user.get("secret_phrase_hash") or not user.get("secret_phrase_salt"):
        return False, "SECRET_PHRASE_NOT_CONFIGURED"
    if not verify_secret_phrase(
        phrase,
        str(user.get("secret_phrase_hash")),
        str(user.get("secret_phrase_salt")),
    ):
        return False, "SECRET_PHRASE_FAILED"
    return True, None
