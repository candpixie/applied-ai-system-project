"""
Learned examples: the part that lets corrections stick.

The system starts from a frozen knowledge base in `dataset.py`. When a person
corrects a decision, the corrected post is appended here instead, to a separate
JSONL store that the retriever and the ML model load alongside the frozen base.

Three rules make this safe to have.

1. **Learning never touches the frozen base.** `dataset.py` is not written to.
   The store is a separate file that can be deleted to return to the shipped
   behaviour, and `evaluate.py` ignores it by default, so the headline accuracy
   numbers keep measuring the shipped system rather than whatever this
   particular machine has been taught.

2. **Held-out posts cannot be taught.** Teaching the system a post from the
   evaluation set would let it memorise the answer key and report a number that
   means nothing. The store refuses those outright.

3. **A lesson has to earn its place.** Adding examples to a small corpus is not
   free: it shifts the inverse-document-frequency of common words and perturbs
   every similarity in the index, which is how a ten-post addition once dropped
   accuracy on posts that shared no vocabulary with it. So the agent measures
   itself before and after accepting a correction, and rolls the correction back
   if the system got worse. See `MoodAgent.teach`.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dataset import ALLOWED_LABELS, HELD_OUT, KNOWLEDGE_BASE
from .logs import get_logger

STORE_PATH = Path(os.environ.get("MOODLENS_MEMORY", "learned_examples.jsonl"))


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


_HELD_OUT_KEYS = {_normalize(t) for t, _ in HELD_OUT}
_BASE_KEYS = {_normalize(t) for t, _ in KNOWLEDGE_BASE}


class RejectedLesson(Exception):
    """Raised when a correction must not be stored."""


class LearningStore:
    """Append-only store of human corrections."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else STORE_PATH
        self.logger = get_logger()
        self._examples: List[Tuple[str, str]] = []
        self.load()

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read the store. A corrupt line is skipped, never fatal."""
        self._examples = []
        if not self.path.exists():
            return

        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text, label = record["text"], record["label"]
                except (json.JSONDecodeError, KeyError):
                    self.logger.warning("skipping malformed lesson in %s", self.path)
                    continue
                if label in ALLOWED_LABELS:
                    self._examples.append((text, label))

    def examples(self) -> List[Tuple[str, str]]:
        return list(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    # ------------------------------------------------------------------

    def validate(self, text: str, label: str) -> str:
        """Check a correction is allowed. Returns the cleaned text.

        Raises RejectedLesson with a reason a person can act on.
        """
        if not isinstance(text, str) or not text.strip():
            raise RejectedLesson("cannot teach an empty post")

        if label not in ALLOWED_LABELS:
            raise RejectedLesson(
                f"'{label}' is not a valid label, expected one of "
                f"{', '.join(ALLOWED_LABELS)}"
            )

        cleaned = text.strip()
        key = _normalize(cleaned)

        if key in _HELD_OUT_KEYS:
            raise RejectedLesson(
                "this post is in the held-out evaluation set. Teaching it would "
                "let the system memorise its own answer key and report an "
                "accuracy that means nothing"
            )

        if key in _BASE_KEYS:
            raise RejectedLesson(
                "this post is already in the frozen knowledge base. Edit "
                "dataset.py if its label is wrong"
            )

        for existing_text, existing_label in self._examples:
            if _normalize(existing_text) == key:
                if existing_label == label:
                    raise RejectedLesson(f"already learned as '{label}'")
                raise RejectedLesson(
                    f"already learned as '{existing_label}'. Remove it from "
                    f"{self.path} before relabelling"
                )

        return cleaned

    # ------------------------------------------------------------------

    def add(self, text: str, label: str) -> Tuple[str, str]:
        """Validate and append a correction. Returns the stored pair."""
        cleaned = self.validate(text, label)
        self._examples.append((cleaned, label))
        self._flush()
        self.logger.info("learned %r as %s", cleaned, label)
        return (cleaned, label)

    def remove_last(self) -> Optional[Tuple[str, str]]:
        """Drop the most recent lesson. Used to roll back a bad one."""
        if not self._examples:
            return None
        dropped = self._examples.pop()
        self._flush()
        self.logger.info("rolled back lesson %r", dropped[0])
        return dropped

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                for text, label in self._examples:
                    handle.write(
                        json.dumps({"text": text, "label": label}, ensure_ascii=False)
                        + "\n"
                    )
        except OSError as exc:  # pragma: no cover - depends on the filesystem
            self.logger.error("could not write %s: %s", self.path, exc)

    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, label in self._examples:
            counts[label] = counts.get(label, 0) + 1
        return counts
