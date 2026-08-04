"""
Retrieval over the labeled knowledge base (the RAG half of the system).

Why retrieval helps here: a lexicon cannot read sarcasm. "I absolutely love
getting stuck in traffic" contains "love" and nothing else the rules can use,
so the rules say positive and are wrong. Retrieval finds previously labeled
sarcastic posts that look like it and votes negative on that evidence.

The retrieved neighbours are not printed next to an answer that was computed
some other way. They are one of the weighted votes that produces the answer,
and on the sarcasm cases they are usually the vote that wins.
"""

from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import KNOWLEDGE_BASE
from .signals import Signal

# Below this cosine similarity a neighbour is treated as unrelated noise.
MIN_SIMILARITY = 0.12

# If even the best neighbour is this weak, retrieval abstains entirely.
ABSTAIN_BELOW = 0.18

# Similarity at which a neighbour is considered strong evidence. Confidence
# scales linearly up to here and is capped after it.
STRONG_SIMILARITY = 0.55


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


class MoodRetriever:
    """TF-IDF nearest-neighbour store over labeled example posts."""

    def __init__(self, corpus: Optional[List[Tuple[str, str]]] = None) -> None:
        corpus = corpus if corpus is not None else KNOWLEDGE_BASE
        if not corpus:
            raise ValueError("Cannot build a retriever over an empty corpus.")

        self.texts = [text for text, _ in corpus]
        self.labels = [label for _, label in corpus]
        self._normalized = [_normalize(t) for t in self.texts]

        # Word unigrams+bigrams catch phrasing ("could have been an email");
        # sublinear_tf keeps a repeated word from dominating a short post.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    # ------------------------------------------------------------------

    def retrieve(
        self,
        text: str,
        k: int = 3,
        exclude_exact: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the top-k most similar labeled posts.

        Args:
            text: the query post.
            k: how many neighbours to return.
            exclude_exact: drop neighbours whose text is identical to the
                query. Used during evaluation so a post in the knowledge base
                cannot retrieve itself and score a free point.
        """
        if not text.strip():
            return []

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        query_norm = _normalize(text)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[Dict[str, Any]] = []

        for i in ranked:
            similarity = float(scores[i])
            if similarity < MIN_SIMILARITY:
                break
            if exclude_exact and self._normalized[i] == query_norm:
                continue
            hits.append(
                {
                    "text": self.texts[i],
                    "label": self.labels[i],
                    "similarity": round(similarity, 4),
                }
            )
            if len(hits) >= k:
                break

        return hits

    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        k: int = 3,
        exclude_exact: bool = False,
    ) -> Signal:
        """Retrieve neighbours and turn them into a weighted vote.

        Each neighbour votes for its own label with weight equal to its cosine
        similarity. Confidence combines two things: how strong the best match
        is, and how much the neighbours agree with each other. Three neighbours
        that disagree should not sound as sure as three that agree.
        """
        hits = self.retrieve(text, k=k, exclude_exact=exclude_exact)

        if not hits or hits[0]["similarity"] < ABSTAIN_BELOW:
            return Signal(
                source="retrieval",
                label=None,
                confidence=0.0,
                rationale=(
                    "no sufficiently similar labeled example in the knowledge "
                    f"base (best similarity {hits[0]['similarity'] if hits else 0.0:.2f} "
                    f"< {ABSTAIN_BELOW})"
                ),
                evidence={"neighbours": hits, "k": k},
            )

        votes: Dict[str, float] = {}
        for hit in hits:
            votes[hit["label"]] = votes.get(hit["label"], 0.0) + hit["similarity"]

        total = sum(votes.values())
        label, winning_weight = max(votes.items(), key=lambda kv: kv[1])
        agreement = winning_weight / total if total else 0.0
        top_similarity = hits[0]["similarity"]

        # Confidence is the product of how strong the best match is and how much
        # the neighbours agree, with no constant floor. An earlier version added
        # a flat 0.35 to every non-abstaining retrieval, which meant a barely
        # related neighbour at similarity 0.31 still spoke with confidence 0.53
        # and could carry the vote on out-of-domain text. Weak evidence should
        # produce weak confidence, not a free baseline.
        strength = min(1.0, top_similarity / STRONG_SIMILARITY)
        confidence = min(0.90, 0.15 + 0.75 * strength * agreement)

        neighbour_summary = "; ".join(
            f'"{h["text"]}" [{h["label"]}] sim={h["similarity"]:.2f}' for h in hits
        )

        return Signal(
            source="retrieval",
            label=label,
            confidence=confidence,
            rationale=(
                f"{len(hits)} neighbour(s), {agreement:.0%} of similarity weight "
                f"on '{label}' :: {neighbour_summary}"
            ),
            evidence={
                "neighbours": hits,
                "votes": {lbl: round(w, 4) for lbl, w in votes.items()},
                "agreement": round(agreement, 3),
                "k": k,
            },
        )
