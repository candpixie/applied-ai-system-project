"""Retrieval tests: does the RAG layer find the right evidence and know when
it has found nothing worth using?"""

import pytest

from moodlens.dataset import KNOWLEDGE_BASE
from moodlens.retrieval import ABSTAIN_BELOW, MoodRetriever


@pytest.fixture(scope="module")
def retriever():
    return MoodRetriever()


class TestRetrieve:
    def test_index_covers_the_whole_knowledge_base(self, retriever):
        assert len(retriever) == len(KNOWLEDGE_BASE)

    def test_empty_corpus_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            MoodRetriever(corpus=[])

    def test_exact_match_is_the_top_hit(self, retriever):
        hits = retriever.retrieve("meeting got moved to 3pm", k=3)
        assert hits[0]["text"] == "meeting got moved to 3pm"
        assert hits[0]["similarity"] == pytest.approx(1.0, abs=0.01)

    def test_exclude_exact_removes_self_matches(self, retriever):
        query = "meeting got moved to 3pm"
        hits = retriever.retrieve(query, k=3, exclude_exact=True)
        assert all(hit["text"] != query for hit in hits)

    def test_k_bounds_the_number_of_neighbours(self, retriever):
        assert len(retriever.retrieve("I love this so much", k=2)) <= 2

    def test_hits_are_sorted_by_descending_similarity(self, retriever):
        hits = retriever.retrieve("the food was great but service was slow", k=5)
        sims = [hit["similarity"] for hit in hits]
        assert sims == sorted(sims, reverse=True)

    def test_empty_query_retrieves_nothing(self, retriever):
        assert retriever.retrieve("", k=3) == []


class TestRetrievalSignal:
    def test_sarcasm_retrieves_negative_neighbours(self, retriever):
        # The whole reason retrieval is in this system. The lexicon reads
        # "perfect" and says positive; the neighbours know better.
        signal = retriever.analyze(
            "oh perfect, my laptop died right before the deadline",
            exclude_exact=True,
        )
        assert signal.label == "negative"

    def test_sarcasm_with_no_shared_vocabulary_still_defeats_retrieval(self, retriever):
        """A documented limitation, asserted so it stays visible.

        "love spending my whole Saturday on a bug I created" is sarcastic, but
        it shares almost no vocabulary with the sarcastic posts in the
        knowledge base, so its nearest neighbours are unrelated posts that
        happen to contain "on". Retrieval only works when something similar
        has already been labeled. Widening the knowledge base is the fix, not
        a cleverer similarity metric.
        """
        signal = retriever.analyze(
            "love spending my whole Saturday on a bug I created",
            exclude_exact=True,
        )
        assert signal.label != "negative"
        assert signal.confidence < 0.5  # at least it does not insist

    def test_unrelated_text_causes_an_abstention(self, retriever):
        signal = retriever.analyze("xylophone quantum bureaucracy zzz")
        assert signal.label is None
        assert signal.confidence == 0.0

    def test_abstention_confidence_is_exactly_zero(self, retriever):
        # An abstaining signal must carry no weight at all in the agent's vote.
        signal = retriever.analyze("qqqq wwww eeee rrrr")
        assert signal.confidence == 0.0

    def test_stronger_match_yields_higher_confidence(self, retriever):
        strong = retriever.analyze("meeting got moved to 3pm")
        weak = retriever.analyze("the appointment was pushed back", exclude_exact=True)
        assert strong.confidence > weak.confidence

    def test_confidence_never_reaches_certainty(self, retriever):
        signal = retriever.analyze("meeting got moved to 3pm")
        assert 0.0 < signal.confidence <= 0.90

    def test_evidence_includes_the_neighbours_that_were_used(self, retriever):
        signal = retriever.analyze("I love this class so much")
        assert signal.evidence["neighbours"]
        assert "similarity" in signal.evidence["neighbours"][0]

    def test_abstain_threshold_is_respected(self, retriever):
        signal = retriever.analyze("zzzzz qqqqq")
        neighbours = signal.evidence.get("neighbours", [])
        if neighbours:
            assert neighbours[0]["similarity"] < ABSTAIN_BELOW
