"""
Tests for learning from corrections.

The risk with a system that learns from users is not that it fails to learn.
It is that learning quietly destroys the thing that made the numbers
trustworthy. Most of these tests are about that.
"""

import pathlib

import pytest

import moodlens.memory as memory_module
from moodlens import MoodAgent
from moodlens.dataset import HELD_OUT, KNOWLEDGE_BASE
from moodlens.memory import LearningStore, RejectedLesson

SARCASM = "imagine having to take 10 classes a day, so fun"
SIMILAR = "imagine having to write five essays tonight, so fun"


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    """Point the learning store at a throwaway file for each test."""
    path = tmp_path / "learned.jsonl"
    monkeypatch.setenv("MOODLENS_MEMORY", str(path))
    monkeypatch.setattr(memory_module, "STORE_PATH", pathlib.Path(path))
    return path


@pytest.fixture
def agent(store_path):
    return MoodAgent(use_memory=True)


class TestEvaluationIntegrity:
    """The reason this feature is dangerous, and what stops it."""

    def test_held_out_posts_cannot_be_taught(self, agent):
        # Teaching an evaluation post would let the system memorise its own
        # answer key. Every accuracy number in the project depends on this.
        text, label = HELD_OUT[0]
        with pytest.raises(RejectedLesson, match="held-out"):
            agent.teach(text, label)

    def test_every_held_out_post_is_protected(self, store_path):
        store = LearningStore(store_path)
        for text, label in HELD_OUT:
            with pytest.raises(RejectedLesson):
                store.validate(text, label)

    def test_memory_is_off_by_default(self):
        # evaluate.py and the test suite must measure the shipped system, not
        # whatever this machine has been taught.
        assert MoodAgent().store is None
        assert MoodAgent().use_memory is False

    def test_learning_does_not_touch_the_frozen_knowledge_base(self, agent):
        before = list(KNOWLEDGE_BASE)
        agent.teach(SARCASM, "negative")
        assert list(KNOWLEDGE_BASE) == before

    def test_a_frozen_agent_ignores_what_another_agent_learned(self, agent):
        agent.teach(SARCASM, "negative")
        assert agent.analyze(SARCASM).label == "negative"
        # Same machine, same store on disk, memory switched off.
        assert MoodAgent(use_memory=False).analyze(SARCASM).label != "negative"


class TestLearning:
    def test_a_correction_changes_the_answer(self, agent):
        assert agent.analyze(SARCASM).label == "positive"  # wrong, to begin with
        agent.teach(SARCASM, "negative")
        assert agent.analyze(SARCASM).label == "negative"

    def test_learning_generalizes_beyond_the_taught_string(self, agent):
        """The point of teaching through retrieval rather than a lookup table.

        A different sentence in the same construction should also flip, because
        the taught post becomes a retrievable neighbour rather than a cached
        answer.
        """
        assert agent.analyze(SIMILAR).label != "negative"
        agent.teach(SARCASM, "negative")
        assert agent.analyze(SIMILAR).label == "negative"

    def test_lessons_persist_across_agents(self, agent, store_path):
        agent.teach(SARCASM, "negative")
        assert MoodAgent(use_memory=True).analyze(SARCASM).label == "negative"

    def test_teaching_reports_what_it_did(self, agent):
        result = agent.teach(SARCASM, "negative")
        assert result["kept"] is True
        assert result["broke"] == 0
        assert "no regression" in result["reason"]


class TestGuardrails:
    def test_crisis_text_is_never_stored(self, agent):
        # A disclosure must not end up in a corpus that gets printed in demos.
        with pytest.raises(RejectedLesson, match="crisis"):
            agent.teach("honestly I want to die", "negative")

    @pytest.mark.parametrize("label", ["happy", "POSITIVE", "", None, 7])
    def test_invalid_labels_are_rejected(self, agent, label):
        with pytest.raises(RejectedLesson):
            agent.teach("some brand new post about nothing", label)

    @pytest.mark.parametrize("text", ["", "   ", None, 42])
    def test_invalid_text_is_rejected(self, agent, text):
        with pytest.raises(RejectedLesson):
            agent.teach(text, "negative")

    def test_knowledge_base_posts_cannot_be_taught(self, agent):
        with pytest.raises(RejectedLesson, match="frozen knowledge base"):
            agent.teach(KNOWLEDGE_BASE[0][0], "negative")

    def test_the_same_post_cannot_be_taught_twice(self, agent):
        agent.teach(SARCASM, "negative")
        with pytest.raises(RejectedLesson, match="already learned"):
            agent.teach(SARCASM, "negative")

    def test_relabelling_requires_removing_the_old_lesson(self, agent):
        agent.teach(SARCASM, "negative")
        with pytest.raises(RejectedLesson, match="already learned as 'negative'"):
            agent.teach(SARCASM, "positive")

    def test_teaching_without_memory_is_an_error(self):
        with pytest.raises(RejectedLesson, match="without memory"):
            MoodAgent(use_memory=False).teach("anything at all", "negative")


class TestRegressionGate:
    def test_a_lesson_that_breaks_working_posts_is_rolled_back(self, agent):
        """The gate exists because adding an example is not a local change.

        A new document shifts the inverse-document-frequency of common words,
        which perturbs every similarity in the index, so a correctly labelled
        post can still make the system worse.
        """
        # "The package arrives on Thursday" is in the knowledge base, labeled
        # neutral. Teaching a near-duplicate with a wrong label poisons the
        # neighbourhood that post depends on.
        result = agent.teach("the package arrives on Friday", "negative")
        assert result["kept"] is False
        assert result["broke"] >= 1
        assert "rolled back" in result["reason"]

    def test_a_rolled_back_lesson_is_not_persisted(self, agent):
        agent.teach("the package arrives on Friday", "negative")
        assert len(agent.store) == 0

    def test_a_rolled_back_lesson_leaves_behaviour_unchanged(self, agent):
        before = agent.analyze("today was rough and I am completely drained").label
        agent.teach("the package arrives on Friday", "negative")
        after = agent.analyze("today was rough and I am completely drained").label
        assert before == after

    def test_the_gate_cannot_tell_whether_a_lesson_is_TRUE(self, agent):
        """Documents a real limit, so nobody mistakes the gate for validation.

        The gate asks "did this damage anything that already worked", not "is
        this label correct". A wrong label on text unlike anything in the
        corpus damages nothing, so it is accepted. The human doing the
        teaching is the authority on correctness, and the system has no
        independent way to check them.
        """
        result = agent.teach("so sad and lonely tonight", "positive")  # plainly wrong
        assert result["kept"] is True
        assert agent.analyze("so sad and lonely tonight").label == "positive"


class TestStoreDurability:
    def test_a_corrupt_line_is_skipped_not_fatal(self, store_path):
        store_path.write_text(
            '{"text": "good post", "label": "positive"}\n'
            "not json at all\n"
            '{"text": "missing label"}\n'
            '{"text": "another", "label": "negative"}\n',
            encoding="utf-8",
        )
        store = LearningStore(store_path)
        assert len(store) == 2

    def test_a_missing_store_is_simply_empty(self, tmp_path):
        assert len(LearningStore(tmp_path / "nope.jsonl")) == 0

    def test_labels_outside_the_allowed_set_are_dropped_on_load(self, store_path):
        store_path.write_text(
            '{"text": "a", "label": "positive"}\n'
            '{"text": "b", "label": "ecstatic"}\n',
            encoding="utf-8",
        )
        assert len(LearningStore(store_path)) == 1
