"""Tests for the rule-based analyzer carried over from the Module 3 lab."""

import pytest

from moodlens import MoodAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return MoodAnalyzer()


class TestPreprocess:
    def test_lowercases_and_splits(self, analyzer):
        assert analyzer.preprocess("I Love This") == ["i", "love", "this"]

    def test_strips_punctuation(self, analyzer):
        assert "amazing" in analyzer.preprocess("that was amazing!!!")

    def test_keeps_emoticons_whole(self, analyzer):
        assert ":)" in analyzer.preprocess("great day :)")

    def test_splits_emoji_into_their_own_tokens(self, analyzer):
        assert "🔥" in analyzer.preprocess("this update is fire🔥")

    def test_collapses_stretched_words(self, analyzer):
        # "soooo" and "soo" must score identically.
        assert analyzer.preprocess("soooo good") == analyzer.preprocess("soo good")

    def test_empty_text_yields_no_tokens(self, analyzer):
        assert analyzer.preprocess("   ") == []


class TestScoring:
    def test_positive_words_raise_the_score(self, analyzer):
        assert analyzer.score_text("happy and excited") > 0

    def test_negative_words_lower_the_score(self, analyzer):
        assert analyzer.score_text("sad and tired") < 0

    def test_negation_flips_polarity(self, analyzer):
        assert analyzer.score_text("not bad") > 0
        assert analyzer.score_text("not happy") < 0

    def test_unknown_words_score_zero(self, analyzer):
        assert analyzer.score_text("the appointment is on tuesday") == 0

    def test_slang_outweighs_a_plain_word(self, analyzer):
        # "fire" is weighted 2, "good" is weighted 1.
        assert analyzer.score_text("this is fire") > analyzer.score_text("this is good")


class TestLabels:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I am so happy and excited", "positive"),
            ("this is terrible and awful", "negative"),
            ("the meeting is at four", "neutral"),
            ("not bad at all", "positive"),
        ],
    )
    def test_labels(self, analyzer, text, expected):
        assert analyzer.predict_label(text) == expected

    def test_two_strong_opposed_signals_give_mixed(self, analyzer):
        assert analyzer.predict_label("stressed but proud") == "mixed"


class TestConsistency:
    def test_score_label_and_explain_never_disagree(self, analyzer):
        """The lab version duplicated this loop three times and they could
        drift. They now share one pass, so this must always hold."""
        for text in [
            "not bad at all",
            "stressed but proud",
            "the meeting is at four",
            "this update is fire 🔥",
            "exhausted, done with everything 💀",
        ]:
            score = analyzer.score_text(text)
            explanation = analyzer.explain(text)
            assert f"score={score}" in explanation


class TestSignal:
    def test_no_lexicon_evidence_gives_low_confidence(self, analyzer):
        # "neutral" from an absence of signal is not the same as a confident
        # judgement that the text is neutral.
        signal = analyzer.analyze("the appointment is on tuesday")
        assert signal.label == "neutral"
        assert signal.confidence <= 0.35

    def test_a_bigger_margin_gives_higher_confidence(self, analyzer):
        weak = analyzer.analyze("good")
        strong = analyzer.analyze("happy excited amazing awesome great")
        assert strong.confidence > weak.confidence

    def test_confidence_is_capped_below_certainty(self, analyzer):
        signal = analyzer.analyze("happy " * 50)
        assert signal.confidence <= 0.80

    def test_evidence_names_the_matched_words(self, analyzer):
        signal = analyzer.analyze("I love this awesome day")
        assert "love" in signal.evidence["positive_hits"]
