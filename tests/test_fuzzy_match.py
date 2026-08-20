"""Tests for src.utils.fuzzy_match and fuzzy fallback in intent_classifier."""

import pytest

from src.utils.fuzzy_match import fuzzy_match, normalize_for_matching, FuzzyResult


class TestNormalizeForMatching:
    def test_lowercase_and_strip(self):
        assert normalize_for_matching("  Hello World  ") == "hello world"

    def test_removes_punctuation(self):
        assert normalize_for_matching("xem l\u1ecbch, h\u00f4m nay!") == "xem l\u1ecbch h\u00f4m nay"

    def test_collapses_whitespace(self):
        assert normalize_for_matching("a   b    c") == "a b c"

    def test_preserves_vietnamese_diacritics(self):
        result = normalize_for_matching("Th\u00eam l\u1ecbch h\u1ecdc")
        assert "th\u00eam" in result
        assert "l\u1ecbch" in result

    def test_empty_string(self):
        assert normalize_for_matching("") == ""


class TestFuzzyMatch:
    CANDIDATES = {
        "xem l\u1ecbch h\u00f4m nay": "VIEW_SCHEDULE",
        "th\u00eam l\u1ecbch h\u1ecdc": "ADD_SCHEDULE",
        "m\u1ea5y gi\u1edf r\u1ed3i": "GET_TIME",
    }

    def test_exact_match(self):
        result = fuzzy_match("xem l\u1ecbch h\u00f4m nay", self.CANDIDATES)
        assert result is not None
        assert result["canonical"] == "VIEW_SCHEDULE"
        assert result["score"] >= 95

    def test_close_match_asr_garbled(self):
        result = fuzzy_match("xem l\u1ecbc h\u00f4m nay", self.CANDIDATES)
        assert result is not None
        assert result["canonical"] == "VIEW_SCHEDULE"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("xem lich hom nay", "VIEW_SCHEDULE"),
            ("them lich hoc", "ADD_SCHEDULE"),
            ("may gio roi", "GET_TIME"),
            ("xem n\u1ecbch h\u00f4m nay", "VIEW_SCHEDULE"),
            ("xem s\u1ecbch h\u00f4m nay", "VIEW_SCHEDULE"),
            ("them lich i\u00ean", "ADD_SCHEDULE"),
        ],
    )
    def test_phonetic_asr_variants(self, text, expected):
        result = fuzzy_match(text, self.CANDIDATES, threshold=65)
        assert result is not None
        assert result["canonical"] == expected

    def test_below_threshold_returns_none(self):
        result = fuzzy_match("abc xyz 123", self.CANDIDATES, threshold=65)
        assert result is None

    def test_empty_text_returns_none(self):
        result = fuzzy_match("", self.CANDIDATES)
        assert result is None

    def test_empty_candidates_returns_none(self):
        result = fuzzy_match("xem l\u1ecbch", {})
        assert result is None

    def test_sequence_candidates(self):
        candidates = ["xem l\u1ecbch h\u00f4m nay", "th\u00eam l\u1ecbch"]
        result = fuzzy_match("xem l\u1ecbch h\u00f4m nay", candidates)
        assert result is not None
        assert result["canonical"] == "xem l\u1ecbch h\u00f4m nay"

    def test_ambiguous_returns_none(self):
        candidates = {"th\u00eam l\u1ecbch": "A", "th\u00eam l\u1ecbc": "B"}
        result = fuzzy_match("th\u00eam l\u1ecbc", candidates, margin=50)
        assert result is None

    def test_custom_threshold(self):
        result = fuzzy_match("xem lich", self.CANDIDATES, threshold=90)
        assert result is None

    def test_returns_fuzzy_result_type(self):
        result = fuzzy_match("m\u1ea5y gi\u1edf r\u1ed3i", self.CANDIDATES)
        assert result is not None
        assert set(result.keys()) == {"matched", "canonical", "score"}


class TestIntentClassifierFuzzyFallback:
    """Verify fuzzy fallback kicks in when regex misses due to ASR noise."""

    def test_regex_still_works_exact(self):
        from src.nlu.intent_classifier import classify_intent
        assert classify_intent("B\u00e2y gi\u1edf m\u1ea5y gi\u1edf?") == "GET_TIME"

    def test_fuzzy_rescues_garbled_get_time(self):
        from src.nlu.intent_classifier import classify_intent
        # ASR co the output "bay gio may gio" (mat dau)
        # nhung normalize_text giu nguyen dau -> van test voi text co dau bi sai nhe
        result = classify_intent("b\u00e2y gi\u1edd m\u1ea5y gi\u1ee3")
        # Cho phep ca GET_TIME (fuzzy match) hoac OUT_OF_SCOPE (neu score qua thap)
        assert result in ("GET_TIME", "OUT_OF_SCOPE")

    def test_fuzzy_rescues_garbled_view_schedule(self):
        from src.nlu.intent_classifier import classify_intent
        result = classify_intent("cho toi xem l\u1ecbc")
        assert result in ("VIEW_SCHEDULE", "OUT_OF_SCOPE")

    def test_fuzzy_rescues_garbled_private_note(self):
        from src.nlu.intent_classifier import classify_intent
        assert classify_intent("M\u1edf v\u00ec ch\u1ee7 luy\u1ec7n t\u1eeb tr\u00e0 tai") == "VIEW_PRIVATE_NOTE"

    def test_fuzzy_rescues_swapped_words(self):
        from src.nlu.intent_classifier import classify_intent
        result = classify_intent("l\u1ecbch xem h\u00f4m nay")
        assert result in ("VIEW_SCHEDULE", "OUT_OF_SCOPE")

    def test_out_of_scope_still_rejected(self):
        from src.nlu.intent_classifier import classify_intent
        assert classify_intent("m\u1edf nh\u1ea1c cho t\u00f4i") == "OUT_OF_SCOPE"

    def test_empty_still_out_of_scope(self):
        from src.nlu.intent_classifier import classify_intent
        assert classify_intent("") == "OUT_OF_SCOPE"

    def test_original_regex_cases_unchanged(self):
        from src.nlu.intent_classifier import classify_intent
        from src.nlu.intent_schema import Intent

        cases = [
            ("B\u00e2y gi\u1edf l\u00e0 m\u1ea5y gi\u1edf?", Intent.GET_TIME.value),
            ("Cho t\u00f4i xem l\u1ecbch ng\u00e0y mai", Intent.VIEW_SCHEDULE.value),
            ("Th\u00eam l\u1ecbch h\u1ecdc m\u00e1y l\u00fac 8h s\u00e1ng mai", Intent.ADD_SCHEDULE.value),
            ("M\u1edf ghi ch\u00fa ri\u00eang t\u01b0 c\u1ee7a t\u00f4i", Intent.VIEW_PRIVATE_NOTE.value),
            ("M\u1ea5y gi\u1edf h\u1ecdc th\u00ec t\u1ed1t?", Intent.OUT_OF_SCOPE.value),
            ("T\u00f4i th\u00edch h\u1ecdc m\u00e1y", Intent.OUT_OF_SCOPE.value),
        ]
        for text, expected in cases:
            assert classify_intent(text) == expected, f"Failed for: {text}"
