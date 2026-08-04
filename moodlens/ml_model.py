"""
The small supervised classifier, carried over from `ml_experiments.py` in the
Module 3 lab and turned into a reusable component.

In the lab this trained on the sample posts and was evaluated on those same
posts, which reports training accuracy and flatters the model. Here it trains
only on KNOWLEDGE_BASE and is evaluated only on HELD_OUT.

It stays deliberately small. With ~40 training examples a bag-of-words logistic
regression is honest about what it is: a weak vote that the agent fuses with
others, not an oracle.
"""

from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .dataset import KNOWLEDGE_BASE
from .signals import Signal

# A ~40-example bag-of-words model does not deserve to sound certain.
MAX_CONFIDENCE = 0.70


class MLMoodModel:
    """Bag-of-words + logistic regression mood classifier."""

    def __init__(self, corpus: Optional[List[Tuple[str, str]]] = None) -> None:
        corpus = corpus if corpus is not None else KNOWLEDGE_BASE
        texts = [t for t, _ in corpus]
        labels = [l for _, l in corpus]

        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if len(set(labels)) < 2:
            raise ValueError("need at least two distinct labels to train")

        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        features = self.vectorizer.fit_transform(texts)

        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(features, labels)
        self.classes = list(self.model.classes_)

    def predict(self, text: str) -> str:
        features = self.vectorizer.transform([text])
        return str(self.model.predict(features)[0])

    def analyze(self, text: str) -> Signal:
        """Predict a label and report the class probability as confidence."""
        if not text.strip():
            return Signal(
                source="ml",
                label=None,
                confidence=0.0,
                rationale="empty input, nothing to classify",
            )

        features = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(features)[0]
        best = int(probabilities.argmax())
        label = str(self.classes[best])
        raw_confidence = float(probabilities[best])

        # If almost none of the query's words appear in the training vocabulary,
        # the model is guessing from a word or two plus the intercept. Say so
        # rather than letting an uninformed guess carry weight in the vote.
        known_terms = int(features.sum())
        word_count = len(text.split())
        too_little_evidence = known_terms == 0 or (known_terms <= 1 and word_count >= 4)

        if too_little_evidence:
            return Signal(
                source="ml",
                label=None,
                confidence=0.0,
                rationale=(
                    f"only {known_terms} of {word_count} words are in the training "
                    "vocabulary, not enough to predict from"
                ),
                evidence={"known_terms": known_terms, "word_count": word_count},
            )

        confidence = min(MAX_CONFIDENCE, raw_confidence)
        distribution = {
            str(cls): round(float(p), 3)
            for cls, p in zip(self.classes, probabilities)
        }

        return Signal(
            source="ml",
            label=label,
            confidence=confidence,
            rationale=(
                f"logistic regression p({label})={raw_confidence:.2f} "
                f"on {known_terms} known term(s)"
            ),
            evidence={"distribution": distribution, "known_terms": known_terms},
        )
