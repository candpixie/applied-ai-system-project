"""
The agent: plan, act, check, repair.

This is the piece that makes MoodLens a system rather than three classifiers in
a folder. For every input it:

  1. PLAN   - decides which components are worth consulting for this text.
  2. ACT    - runs them and collects one `Signal` each.
  3. CHECK  - fuses the signals into a weighted vote and asks whether the
              result is trustworthy (do the components agree? is the winning
              signal confident?).
  4. REPAIR - if not, it does something about it: widen retrieval to pull in
              more evidence, and optionally ask the LLM adjudicator to break
              the tie. Then it re-checks.
  5. ABSTAIN - if it still cannot get there, it says so and flags the item for
              human review instead of guessing with a confident face.

Source weights encode a prior about which component to believe when they
disagree. Retrieval outranks the lexicon because retrieval is what handles
sarcasm; the small ML model ranks lowest because ~40 training examples is not
much to stand on.
"""

from typing import Dict, List, Optional, Tuple

from . import llm_adjudicator
from .dataset import KNOWLEDGE_BASE
from .guardrails import check_input, detect_crisis_language, validate_output
from .logs import get_logger, log_decision
from .memory import LearningStore, RejectedLesson
from .ml_model import MLMoodModel
from .mood_analyzer import MoodAnalyzer
from .retrieval import MoodRetriever
from .signals import Decision, Signal

SOURCE_WEIGHTS: Dict[str, float] = {
    "retrieval": 1.0,
    "rules": 0.9,
    "ml": 0.7,
    "llm": 1.2,
}

# Below this fused confidence, or this much disagreement, the agent repairs.
REPAIR_CONFIDENCE = 0.55
REPAIR_AGREEMENT = 0.60

# Below this after repair, the answer is flagged for a human.
REVIEW_CONFIDENCE = 0.45

DEFAULT_K = 3
WIDENED_K = 7


class MoodAgent:
    """Orchestrates the components and owns the final answer."""

    def __init__(
        self,
        corpus: Optional[List[Tuple[str, str]]] = None,
        use_llm: bool = False,
        k: int = DEFAULT_K,
        use_memory: bool = False,
    ) -> None:
        """
        Args:
            corpus: override the labeled examples entirely. Used by tests and
                by the evaluation harness.
            use_memory: load human corrections from the learning store on top
                of the frozen knowledge base. Off by default on purpose, so
                that `evaluate.py` and the test suite measure the shipped
                system rather than whatever this machine has been taught.
        """
        self.use_memory = use_memory and corpus is None
        self.store = LearningStore() if self.use_memory else None
        self.use_llm = use_llm
        self.k = k
        self.logger = get_logger()
        self.rules = MoodAnalyzer()
        self._build(corpus)

    def _build(self, corpus: Optional[List[Tuple[str, str]]] = None) -> None:
        """(Re)build the corpus-backed components.

        Called at startup and again after every accepted lesson, because both
        the retriever's index and the ML model's vocabulary are fitted to the
        corpus and cannot be updated in place.
        """
        if corpus is None and self.store is not None:
            corpus = list(KNOWLEDGE_BASE) + self.store.examples()
        self._corpus = corpus
        self.retriever = MoodRetriever(corpus)
        self.ml = MLMoodModel(corpus)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(signals: List[Signal]) -> Tuple[Optional[str], float, float]:
        """Weighted vote across signals.

        Returns (label, confidence, agreement). Signals that abstained
        (label None or zero confidence) contribute nothing.
        """
        votes: Dict[str, float] = {}
        for signal in signals:
            if signal.label is None or signal.confidence <= 0:
                continue
            # Vote weight scales with the SQUARE of confidence, not linearly.
            #
            # Linear weighting let two barely-informed signals outvote one
            # well-informed one. "this sucks" came back positive because
            # retrieval (0.40, matching junk on the words "this is") and the ML
            # model (0.39, voting off one known term) summed to more than the
            # rules component at 0.65, which had correctly read "sucks".
            #
            # Squaring is the cheapest expression of "weak evidence should
            # barely vote": it drops 0.40 to 0.16 while only dropping 0.65 to
            # 0.42, so a confident component beats two vague ones instead of
            # losing to them.
            weight = SOURCE_WEIGHTS.get(signal.source, 0.5) * signal.confidence
            votes[signal.label] = votes.get(signal.label, 0.0) + weight

        if not votes:
            return None, 0.0, 0.0

        total = sum(votes.values())
        label = max(votes.items(), key=lambda kv: kv[1])[0]
        agreement = votes[label] / total

        supporters = [s.confidence for s in signals if s.label == label]

        # Noisy-OR over the supporting signals: two components that each say
        # 0.6 and agree should end up more confident than either alone, since
        # they reach the same answer by different routes (lexicon lookup,
        # nearest-neighbour evidence, learned weights). Taking max() instead
        # would throw that corroboration away.
        agreement_odds = 1.0
        for confidence_value in supporters:
            agreement_odds *= 1.0 - confidence_value
        base = 1.0 - agreement_odds

        # Then damp by how much of the total vote weight the winner actually
        # holds. A contested win is worth less than a unanimous one.
        confidence = min(0.95, base * (0.55 + 0.45 * agreement))
        return label, confidence, agreement

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, text: str, exclude_exact: bool = False) -> Decision:
        """Classify one piece of text end to end.

        Args:
            text: the raw input, straight from a user. Assumed hostile until
                the guardrails have looked at it.
            exclude_exact: forbid retrieval from returning a neighbour whose
                text is identical to the query. Used when evaluating on posts
                that are themselves in the knowledge base.
        """
        trace: List[str] = []

        # --- Guardrails (input) ----------------------------------------
        guard = check_input(text)
        if not guard.ok:
            trace.append(f"guardrails:{guard.status}")
            self.logger.info("input %s: %s", guard.status, guard.reason)
            decision = Decision(
                text=guard.cleaned_text or (text if isinstance(text, str) else ""),
                label="uncertain",
                confidence=0.0,
                status=guard.status,
                rationale=guard.reason,
                trace=trace,
            )
            log_decision(decision.to_dict())
            return decision

        clean = guard.cleaned_text
        trace.append("guardrails:ok")
        if guard.flags:
            # Injection attempts are not blocked: the text is still just text to
            # be classified. They are flagged so the LLM path can be skipped.
            trace.append(f"flags:{len(guard.flags)}")
            self.logger.warning("input carries flags: %s", guard.flags)

        # --- Plan -------------------------------------------------------
        plan = ["rules", "retrieval", "ml"]
        trace.append("plan:" + "+".join(plan))

        # --- Act --------------------------------------------------------
        signals: List[Signal] = [
            self.rules.analyze(clean),
            self.retriever.analyze(clean, k=self.k, exclude_exact=exclude_exact),
            self.ml.analyze(clean),
        ]
        active = [s for s in signals if s.label is not None and s.confidence > 0]
        trace.append(f"act:{len(active)}/{len(signals)} signals")

        # --- Check ------------------------------------------------------
        label, confidence, agreement = self._fuse(signals)
        trace.append(f"check:conf={confidence:.2f},agree={agreement:.2f}")

        repaired = False
        needs_repair = (
            label is None
            or confidence < REPAIR_CONFIDENCE
            or agreement < REPAIR_AGREEMENT
        )

        # --- Repair -----------------------------------------------------
        if needs_repair:
            repaired = True
            self.logger.info(
                "repair pass for %r (conf=%.2f, agree=%.2f)", clean, confidence, agreement
            )

            # Widen the evidence window: more neighbours, more chances that a
            # genuinely similar labeled example is in range.
            widened = self.retriever.analyze(
                clean, k=WIDENED_K, exclude_exact=exclude_exact
            )
            signals = [s for s in signals if s.source != "retrieval"] + [widened]
            trace.append(f"repair:retrieval k={self.k}->{WIDENED_K}")

            label, confidence, agreement = self._fuse(signals)

            # Still contested, and the LLM path is switched on: ask it to break
            # the tie. Skipped when the input tried to inject instructions.
            still_contested = (
                label is None
                or confidence < REPAIR_CONFIDENCE
                or agreement < REPAIR_AGREEMENT
            )
            injection_flagged = any(f.startswith("injection:") for f in guard.flags or [])

            if still_contested and self.use_llm and not injection_flagged:
                candidates = [s.label for s in signals if s.label]
                verdict = llm_adjudicator.adjudicate(clean, candidates)
                if verdict is not None:
                    signals.append(verdict)
                    label, confidence, agreement = self._fuse(signals)
                    trace.append("repair:llm-adjudicated")
                else:
                    trace.append("repair:llm-unavailable")
            elif still_contested and injection_flagged:
                trace.append("repair:llm-skipped(injection-flag)")

            trace.append(f"recheck:conf={confidence:.2f},agree={agreement:.2f}")

        # --- Abstain / finalize ----------------------------------------
        if label is None:
            status = "needs_review"
            rationale = (
                "no component produced a usable opinion: the text matched no "
                "lexicon entry, no training vocabulary and no similar example"
            )
            label, confidence = "uncertain", 0.0
        elif confidence < REVIEW_CONFIDENCE:
            status = "needs_review"
            rationale = (
                f"fused confidence {confidence:.2f} is below the {REVIEW_CONFIDENCE} "
                f"review threshold; best guess '{label}' is reported but should "
                "not be trusted without a human check"
            )
        else:
            status = "ok"
            supporting = [s.source for s in signals if s.label == label]
            rationale = (
                f"{', '.join(supporting)} agreed on '{label}' "
                f"({agreement:.0%} of vote weight)"
            )

        # --- Guardrails (output) ---------------------------------------
        label, confidence = validate_output(label, confidence)
        trace.append(f"final:{label}")

        decision = Decision(
            text=clean,
            label=label,
            confidence=confidence,
            status=status,
            rationale=rationale,
            signals=signals,
            trace=trace,
            repaired=repaired,
        )
        log_decision(decision.to_dict())
        return decision

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Learning from corrections
    # ------------------------------------------------------------------

    def self_check(self) -> Tuple[float, set]:
        """Leave-one-out pass over the frozen knowledge base.

        Every base post is classified with itself excluded from retrieval, so
        it cannot look up its own answer. Returns (accuracy, set of indices the
        agent currently gets right).

        This deliberately never touches the held-out set. Using the evaluation
        data to decide what to learn would contaminate the only honest
        measurement the project has.
        """
        correct = set()
        for i, (text, truth) in enumerate(KNOWLEDGE_BASE):
            if self.analyze(text, exclude_exact=True).label == truth:
                correct.add(i)
        return len(correct) / len(KNOWLEDGE_BASE), correct

    def teach(self, text: str, label: str) -> Dict[str, object]:
        """Accept a human correction, then verify it actually helped.

        The sequence matters. Adding examples to a small corpus is not free: it
        shifts the inverse-document-frequency of common words and perturbs
        every similarity in the index at once, so a lesson can fix the post it
        came from and quietly break unrelated ones. This method measures
        leave-one-out accuracy before and after, and **rolls the lesson back if
        the system got worse**.

        Returns a dict describing what happened, including whether the lesson
        was kept, so a caller can tell the user rather than silently discarding
        their correction.
        """
        if self.store is None:
            raise RejectedLesson(
                "this agent was built without memory (use_memory=False)"
            )

        # A crisis disclosure must never become a labeled training example.
        # It is not a mood-classification problem and storing it would put
        # someone's disclosure into a corpus that gets printed in demos.
        if detect_crisis_language(text if isinstance(text, str) else ""):
            raise RejectedLesson(
                "this text triggers the crisis guardrail and will not be "
                "stored as a training example"
            )

        self.store.validate(text, label)  # raises RejectedLesson with a reason
        before, before_correct = self.self_check()

        self.store.add(text, label)
        self._build()
        after, after_correct = self.self_check()

        # "Do no harm" is stricter than "do not lower the average". Aggregate
        # accuracy can stay flat while one post is fixed and an unrelated one
        # breaks, and an aggregate over 49 posts barely moves for a single
        # lesson anyway: an earlier version of this gate let five of seven
        # deliberately mislabelled lessons through. Rejecting any regression on
        # a post that already worked catches those.
        broke = before_correct - after_correct
        if broke:
            examples = [KNOWLEDGE_BASE[i][0] for i in sorted(broke)[:2]]
            self.store.remove_last()
            self._build()
            self.logger.warning(
                "rejected lesson %r: it broke %d working post(s)", text, len(broke)
            )
            return {
                "kept": False,
                "text": text,
                "label": label,
                "before": before,
                "after": after,
                "broke": len(broke),
                "reason": (
                    f"lesson rolled back: it broke {len(broke)} knowledge-base "
                    f"post(s) that previously classified correctly, for example "
                    + "; ".join(f'"{e}"' for e in examples)
                    + ". Adding an example shifts term weights across the whole "
                    "index, so a correct label can still make the system worse"
                ),
            }

        fixed = after_correct - before_correct
        return {
            "kept": True,
            "text": text,
            "label": label,
            "before": before,
            "after": after,
            "broke": 0,
            "fixed": len(fixed),
            "reason": (
                f"lesson kept: no regression on the knowledge base "
                f"({before:.2f} -> {after:.2f}, {len(fixed)} other post(s) improved)"
            ),
            "learned_total": len(self.store),
        }

    # ------------------------------------------------------------------

    def analyze_batch(
        self, texts: List[str], exclude_exact: bool = False
    ) -> List[Decision]:
        """Classify many inputs. One bad input never stops the batch."""
        results: List[Decision] = []
        for item in texts:
            try:
                results.append(self.analyze(item, exclude_exact=exclude_exact))
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                self.logger.error("unhandled failure on %r: %s", item, exc)
                results.append(
                    Decision(
                        text=str(item),
                        label="uncertain",
                        confidence=0.0,
                        status="blocked",
                        rationale=f"internal error: {exc}",
                        trace=["error"],
                    )
                )
        return results
