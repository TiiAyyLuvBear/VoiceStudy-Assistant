"""Fuzzy string matching for noisy ASR transcripts.

Phonetic-aware scoring optimized for Vietnamese ASR errors:
dau sai (chu/chu), phu am (not/note), thanh dieu (boi/buoi).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TypedDict

try:
    from rapidfuzz import fuzz as _rf_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_CATEGORIES = ("P", "S")

# Vietnamese phonetic confusion pairs — ASR hay nham giua cac cap nay
_PHONETIC_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ngh", "ng"),
    ("gi", "z"),
    ("d", "z"),
    ("r", "z"),
    ("tr", "ch"),
    ("s", "x"),
    ("n", "l"),
    ("v", "z"),
    ("q", "k"),
    ("c", "k"),
    ("y", "i"),
    ("uô", "ươ"),
    ("iê", "ie"),
    ("yê", "ie"),
    ("ênh", "ên"),
    ("ăng", "ân"),
    ("ươi", "ơi"),
)

_ACCENT_STRIP_RE = re.compile(r"[\u0300-\u036f]")

# Common ASR substitutions (whole word or substring)
_ASR_SYNONYMS: dict[str, list[str]] = {
    "ghi ch\u00fa": ["ghi ch\u1ee7", "g ch\u00fa", "v\u00ec ch\u1ee7"],
    "note": ["n\u00f3t", "n\u00f4t", "not"],
    "ri\u00eang t\u01b0": ["ring t\u01b0", "ri\u00eang tu", "luy\u1ec7n t\u1eeb", "tr\u00e0 tai"],
    "c\u00e1 nh\u00e2n": ["c\u1ea3 nh\u00e2n", "ca nh\u00e2n"],
    "b\u1ea3o m\u1eadt": ["b\u00e1o m\u1eadt", "b\u1ea3o m\u00e2t"],
    "l\u1ecbch": ["l\u1ecbc", "l\u00edch"],
    "bu\u1ed5i": ["b\u1ed1i", "b\u1ed3i", "b\u1ea1i"],
    "hi\u1ec3n th\u1ecb": ["hi\u1ec3n h\u1ec7", "hi\u1ec3n h\u1ecb"],
    "cu\u1ed9c h\u1eb9n": ["cu\u1ed9c h\u00e8n", "c\u1ed9c h\u1eb9n"],
    "l\u1eadp": ["l\u00e0m", "l\u1ea1p"],
    "th\u00eam": ["th\u00eam", "th\u00e8m"],
    "m\u1ea5y gi\u1edf": ["m\u1ea3y gi\u1edf", "m\u1ea5y gi\u1ee3"],
    "gi\u1edf": ["gi\u1ee3", "gi\u00f2"],
}


class FuzzyResult(TypedDict):
    """Ket qua fuzzy match."""

    matched: str
    canonical: str
    score: float


def normalize_for_matching(text: str) -> str:
    """Chuan hoa nhe: lowercase, NFC, bo dau cau, gop khoang trang."""

    text = unicodedata.normalize("NFC", text).lower().strip()
    chars = [
        " " if unicodedata.category(ch).startswith(_PUNCT_CATEGORIES) else ch
        for ch in text
    ]
    return _WHITESPACE_RE.sub(" ", "".join(chars)).strip()


def _phonetic_normalize(text: str) -> str:
    """Chuan hoa phonetic cho ASR Viet: bo dau va gom cac am de nham."""

    result = unicodedata.normalize("NFD", text)
    result = _ACCENT_STRIP_RE.sub("", result).replace("đ", "d")
    result = unicodedata.normalize("NFC", result)
    for source, target in _PHONETIC_REPLACEMENTS:
        result = result.replace(source, target)
    return result


def _expand_asr_variants(text: str) -> str:
    """Thay ASR-garbled substrings bang dang chuan."""

    result = text
    for canonical, variants in _ASR_SYNONYMS.items():
        for variant in variants:
            if variant in result:
                result = result.replace(variant, canonical)
    return result


def _difflib_ratio(query: str, candidate: str) -> float:
    """Fallback scorer dung difflib khi khong co rapidfuzz."""

    from difflib import SequenceMatcher

    return SequenceMatcher(None, query, candidate).ratio() * 100


def _base_score(query: str, candidate: str) -> float:
    """Token_sort (60%) + partial (40%)."""

    if _HAS_RAPIDFUZZ:
        s1 = _rf_fuzz.token_sort_ratio(query, candidate)
        s2 = _rf_fuzz.partial_ratio(query, candidate)
    else:
        tokens_q = " ".join(sorted(query.split()))
        tokens_c = " ".join(sorted(candidate.split()))
        s1 = _difflib_ratio(tokens_q, tokens_c)
        s2 = _difflib_ratio(query, candidate)
    return 0.6 * s1 + 0.4 * s2


def _score_pair(query: str, candidate: str) -> float:
    """Score cuoi cung: max cua raw va phonetic-expanded variants."""

    score_raw = _base_score(query, candidate)

    query_expanded = _expand_asr_variants(query)
    query_phonetic = _phonetic_normalize(query_expanded)
    candidate_phonetic = _phonetic_normalize(candidate)

    if query_phonetic != query or candidate_phonetic != candidate:
        score_phonetic = _base_score(query_phonetic, candidate_phonetic)
        return max(score_raw, score_phonetic)

    if query_expanded != query:
        score_expanded = _base_score(query_expanded, candidate)
        return max(score_raw, score_expanded)

    return score_raw


def _keyword_overlap(query_words: set[str], candidate: str,
                     keywords: Sequence[str] | None) -> bool:
    """Kiem tra xem query co chua it nhat 1 keyword anchor khong.

    Neu keywords la None hoac rong, luon tra True (khong yeu cau keyword).
    """

    if not keywords:
        return True

    candidate_words = set(candidate.split())
    for kw in keywords:
        kw_words = set(kw.split())
        if kw_words & query_words:
            return True
        for syn_list in _ASR_SYNONYMS.values():
            for syn in syn_list:
                syn_words = set(syn.split())
                if syn_words & query_words and kw_words & candidate_words:
                    return True
    return False


def fuzzy_match(
    text: str,
    candidates: Mapping[str, str] | Sequence[str],
    *,
    threshold: float = 65.0,
    margin: float = 10.0,
    min_words: int = 0,
    keywords: Mapping[str, Sequence[str]] | None = None,
) -> FuzzyResult | None:
    """Tim candidate gan nhat voi *text*.

    Parameters
    ----------
    text:
        Chuoi can match (thuong la ASR transcript da normalize).
    candidates:
        - ``Mapping[phrase, canonical_value]`` -- moi phrase map ve mot gia tri
          canonical (vi du intent name).
        - ``Sequence[str]`` -- danh sach phrase; canonical = chinh phrase do.
    threshold:
        Diem toi thieu (0-100) de chap nhan match.
    margin:
        Khoang cach toi thieu giua top-1 canonical va top-1 cua canonical
        khac. Neu top-2 cung canonical voi top-1 thi bo qua margin.
    min_words:
        So tu toi thieu cua query de thuc hien fuzzy match.
    keywords:
        ``Mapping[canonical_value, list_of_anchor_keywords]``.  Khi match mot
        canonical, yeu cau query chua it nhat 1 keyword.  Giup reject false
        positive khi cau truc giong nhung thieu tu khoa then chot.

    Returns
    -------
    ``FuzzyResult`` neu tim duoc match du tin cay, nguoc lai ``None``.
    """

    query = normalize_for_matching(text)
    if not query:
        return None

    if min_words > 0 and len(query.split()) < min_words:
        return None

    if isinstance(candidates, Mapping):
        lookup: dict[str, str] = dict(candidates)
    else:
        lookup = {phrase: phrase for phrase in candidates}

    if not lookup:
        return None

    query_expanded = normalize_for_matching(_expand_asr_variants(query))
    query_words = set(query.split()) | set(query_expanded.split())

    scored: list[tuple[str, float]] = []
    for phrase in lookup:
        norm_phrase = normalize_for_matching(phrase)
        score = _score_pair(query, norm_phrase)

        canonical = lookup[phrase]
        if keywords and canonical in keywords:
            if not _keyword_overlap(query_words, norm_phrase, keywords[canonical]):
                score *= 0.5

        scored.append((phrase, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    best_phrase, best_score = scored[0]
    if best_score < threshold:
        return None

    best_canonical = lookup[best_phrase]

    if len(scored) > 1:
        best_rival_score = 0.0
        for rival_phrase, rival_score in scored[1:]:
            if lookup[rival_phrase] != best_canonical:
                best_rival_score = rival_score
                break
        if best_score - best_rival_score < margin:
            return None

    return FuzzyResult(
        matched=best_phrase,
        canonical=best_canonical,
        score=best_score,
    )
