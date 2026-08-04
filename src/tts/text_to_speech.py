"""Best-effort Vietnamese text-to-speech."""

from __future__ import annotations

from io import BytesIO


def synthesize_vietnamese(text: str) -> bytes | None:
    """Return MP3 bytes, or None when TTS is unavailable."""
    if not text.strip():
        return None
    try:
        from gtts import gTTS

        output = BytesIO()
        gTTS(text=text, lang="vi").write_to_fp(output)
        return output.getvalue()
    except Exception:
        return None
