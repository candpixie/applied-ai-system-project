"""
Guardrails: what the system refuses to do, and what it refuses to emit.

Three layers:

1. Input validation - reject things that cannot be classified meaningfully
   (empty, wrong type, absurdly long) before any model runs.
2. Safety triage - mood text is exactly the kind of text where someone may
   disclose that they are in crisis. MoodLens is a sentiment classifier, not a
   clinical tool. When crisis language appears it stops classifying and hands
   back support resources instead of a cheerful "negative, confidence 0.72".
3. Output validation - nothing leaves the system unless the label is in the
   allowed set and the confidence is a real number in [0, 1].

The safety check is deliberately tuned to over-trigger. Labelling an ordinary
sad post as a safety hold costs a user one confusing response. Missing a real
disclosure costs much more. That asymmetry is the whole design argument, and
the false-positive rate it produces is measured in reports/evaluation_report.md.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from .dataset import ALLOWED_LABELS

MAX_INPUT_CHARS = 1000

# Phrases that trigger a safety hold. Matched on normalized text with
# punctuation stripped, so "i can't go on..." and "i cant go on" both hit.
CRISIS_PATTERNS: List[str] = [
    r"\bkill(ing)? myself\b",
    r"\bend (my life|it all)\b",
    r"\bwant to die\b",
    r"\bdont want to (live|be here|wake up)\b",
    r"\bbetter off (dead|without me)\b",
    r"\bno reason to live\b",
    r"\bsuicid(e|al)\b",
    r"\b(hurt|harm|cut) myself\b",
    r"\bself[ -]?harm\b",
    r"\bcant go on\b",
]

# Text trying to talk to the optional LLM layer rather than be classified by it.
INJECTION_PATTERNS: List[str] = [
    r"ignore (all |any )?(the )?(previous|prior|above) instructions",
    r"disregard (the )?(previous|prior|above)",
    r"you are now\b",
    r"system prompt",
    r"<\s*/?\s*(system|assistant|instructions)\s*>",
]

SUPPORT_MESSAGE = (
    "This text contains language associated with crisis or self-harm. MoodLens "
    "does not assign a mood label to it and does not attempt to assess risk. "
    "If you or someone you know needs support in the US, call or text 988 "
    "(Suicide & Crisis Lifeline) or text HOME to 741741. Outside the US, see "
    "https://findahelpline.com."
)


@dataclass
class GuardResult:
    """Outcome of the input guardrail pass."""

    ok: bool
    cleaned_text: str = ""
    status: str = "ok"  # "ok" | "blocked" | "safety_hold"
    reason: str = ""
    flags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.flags is None:
            self.flags = []


def _normalize_for_matching(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for pattern checks.

    Apostrophes are deleted rather than replaced with a space. Replacing them
    turned "can't go on" into "can t go on", which slipped straight past the
    `\\bcant go on\\b` crisis pattern. A contraction must not be able to
    disable a safety check, and this is the kind of bug that only shows up if
    something actually tests the contracted spelling.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    dropped = re.sub(r"['’ʼ]", "", lowered)
    stripped = re.sub(r"[^a-z0-9\s]", " ", dropped)
    return " ".join(stripped.split())


def _normalize_lightly(text: str) -> str:
    """Lowercase and collapse whitespace, keeping punctuation.

    Injection patterns need this: stripping punctuation would erase the angle
    brackets in "<system>...</system>", which is the exact thing being looked
    for.
    """
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def detect_crisis_language(text: str) -> List[str]:
    """Return the crisis patterns that matched, empty list if none did."""
    normalized = _normalize_for_matching(text)
    return [p for p in CRISIS_PATTERNS if re.search(p, normalized)]


def detect_injection(text: str) -> List[str]:
    """Return prompt-injection patterns that matched, empty list if none did."""
    normalized = _normalize_lightly(text)
    return [p for p in INJECTION_PATTERNS if re.search(p, normalized)]


def check_input(text: object) -> GuardResult:
    """Validate and triage a raw input before any model sees it."""
    if not isinstance(text, str):
        return GuardResult(
            ok=False,
            status="blocked",
            reason=f"input must be a string, got {type(text).__name__}",
        )

    # Strip control characters that would corrupt logs or terminal output.
    cleaned = "".join(
        ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C"
    ).strip()

    if not cleaned:
        return GuardResult(
            ok=False, status="blocked", reason="input is empty or whitespace only"
        )

    if len(cleaned) > MAX_INPUT_CHARS:
        return GuardResult(
            ok=False,
            status="blocked",
            reason=(
                f"input is {len(cleaned)} characters, over the "
                f"{MAX_INPUT_CHARS} character limit for a short post"
            ),
        )

    crisis_hits = detect_crisis_language(cleaned)
    if crisis_hits:
        return GuardResult(
            ok=False,
            cleaned_text=cleaned,
            status="safety_hold",
            reason=SUPPORT_MESSAGE,
            flags=[f"crisis:{p}" for p in crisis_hits],
        )

    flags = [f"injection:{p}" for p in detect_injection(cleaned)]
    return GuardResult(ok=True, cleaned_text=cleaned, status="ok", flags=flags)


def validate_output(label: object, confidence: object) -> tuple[str, float]:
    """Clamp a proposed answer into something the system is allowed to emit.

    Anything outside the allowed label set becomes "uncertain", and any
    confidence that is not a finite number in [0, 1] becomes 0.0. This is the
    last line of defence: even if a component misbehaves, callers only ever
    see a valid label and a valid confidence.
    """
    safe_label = label if label in ALLOWED_LABELS else "uncertain"

    try:
        value = float(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = 0.0
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        value = 0.0
    safe_confidence = max(0.0, min(1.0, value))

    return safe_label, safe_confidence
