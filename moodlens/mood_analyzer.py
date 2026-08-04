"""
Rule-based mood analyzer.

This is the component carried over most directly from the Module 3 "Mood
Machine" lab: tokenize, look up words in a lexicon, apply negation, add up a
score, map the score to a label.

Two changes from the lab version:

1. The lab duplicated the same scoring loop inside `score_text`,
   `predict_label` and `explain`. They now all call one `_score` pass, so the
   three can never drift apart.
2. It reports a calibrated confidence alongside the label, because the agent
   needs to know how much to trust it, not just what it thinks.
"""

import re
from typing import Dict, List, Optional, Tuple

from .dataset import NEGATIONS, NEGATIVE_WORDS, POSITIVE_WORDS, SLANG_SIGNALS
from .signals import Signal

# Unicode ranges we treat as standalone emoji tokens.
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿❤]")


class MoodAnalyzer:
    """A transparent, deterministic lexicon classifier."""

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        self.positive_words = {w.lower() for w in positive_words}
        self.negative_words = {w.lower() for w in negative_words}
        self.slang_signals: Dict[str, int] = dict(SLANG_SIGNALS)
        self.negations = set(NEGATIONS)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """Turn raw text into tokens.

        Keeps text emoticons like ":)" whole, splits unicode emoji into their
        own tokens, strips punctuation off words, and squashes character
        runs so "soooo" and "soo" score the same.
        """
        cleaned = text.strip().lower()
        tokens: List[str] = []

        for raw in cleaned.split():
            # Emoticons are lexicon entries in their own right, keep them intact.
            if raw in self.slang_signals:
                tokens.append(raw)
                continue

            emojis = _EMOJI_RE.findall(raw)
            word = re.sub(r"[^a-z0-9']", "", raw)
            if word:
                # "soooo" -> "soo": collapse runs of 3+ to 2 characters.
                word = re.sub(r"(.)\1{2,}", r"\1\1", word)
                tokens.append(word)
            tokens.extend(emojis)

        return tokens

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _lexicon_value(self, token: str) -> int:
        """Base polarity weight for a single token, 0 if unknown."""
        if token in self.slang_signals:
            return self.slang_signals[token]
        if token in self.positive_words:
            return 1
        if token in self.negative_words:
            return -1
        return 0

    def _score(self, text: str) -> Tuple[int, int, List[str], List[str]]:
        """Single scoring pass.

        Returns (positive_total, negative_total, positive_hits, negative_hits)
        where the totals are magnitudes, both non-negative.
        """
        tokens = self.preprocess(text)
        pos_total = 0
        neg_total = 0
        pos_hits: List[str] = []
        neg_hits: List[str] = []

        for i, token in enumerate(tokens):
            value = self._lexicon_value(token)
            if value == 0:
                continue

            label = token
            # "not bad" flips positive, "not happy" flips negative.
            if i > 0 and tokens[i - 1] in self.negations:
                value = -value
                label = f"not+{token}"

            if value > 0:
                pos_total += value
                pos_hits.append(label)
            else:
                neg_total += -value
                neg_hits.append(label)

        return pos_total, neg_total, pos_hits, neg_hits

    def score_text(self, text: str) -> int:
        """Net mood score. Positive leans happy, negative leans upset."""
        pos, neg, _, _ = self._score(text)
        return pos - neg

    def predict_label(self, text: str) -> str:
        """Map the score to one of positive / negative / neutral / mixed."""
        pos, neg, _, _ = self._score(text)

        # Both sides fired hard: genuinely conflicted, not just cancelling out.
        if pos >= 2 and neg >= 2:
            return "mixed"
        net = pos - neg
        if net > 0:
            return "positive"
        if net < 0:
            return "negative"
        return "neutral"

    def explain(self, text: str) -> str:
        """One-line explanation of the call, for logs and the CLI."""
        pos, neg, pos_hits, neg_hits = self._score(text)
        return (
            f"score={pos - neg} "
            f"(positive: {pos_hits or '[]'}, negative: {neg_hits or '[]'})"
        )

    # ------------------------------------------------------------------
    # Agent-facing interface
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Signal:
        """Return the rule-based opinion as a `Signal`.

        Confidence grows with the size of the winning margin, and is
        deliberately capped: a lexicon can be wrong with any margin, most
        obviously on sarcasm, so it never gets to sound certain.
        """
        pos, neg, pos_hits, neg_hits = self._score(text)
        label = self.predict_label(text)
        margin = abs(pos - neg)
        hits = len(pos_hits) + len(neg_hits)

        if hits == 0:
            # No lexicon evidence at all. "neutral" here is an absence of
            # signal, not a positive finding, so confidence stays low.
            confidence = 0.30
        elif label == "mixed":
            confidence = 0.55
        else:
            confidence = min(0.80, 0.45 + 0.10 * margin)

        return Signal(
            source="rules",
            label=label,
            confidence=confidence,
            rationale=self.explain(text),
            evidence={
                "positive_total": pos,
                "negative_total": neg,
                "positive_hits": pos_hits,
                "negative_hits": neg_hits,
                "tokens": self.preprocess(text),
            },
        )
