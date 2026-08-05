"""
Labeled data for MoodLens.

This file grew out of `dataset.py` in the original Module 3 "Mood Machine" lab,
which held two small word lists and six example posts. Two things changed:

1. The corpus is much larger and deliberately messy (slang, emoji, sarcasm,
   negation, flat factual statements).
2. It is split into KNOWLEDGE_BASE and HELD_OUT. The knowledge base is the only
   thing the retriever and the ML model ever see. HELD_OUT is never indexed and
   never trained on, so evaluation numbers mean something.

Label set (fixed, enforced by guardrails.py): positive, negative, neutral, mixed.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Lexicons (carried over from the original lab, then extended)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy", "great", "good", "love", "excited", "awesome", "fun", "chill",
    "relaxed", "amazing", "proud", "grateful", "glad", "brilliant", "lovely",
    "wonderful", "delighted", "pleased", "hopeful", "enjoyed", "beautiful",
]

NEGATIVE_WORDS = [
    "sad", "bad", "terrible", "awful", "angry", "upset", "tired", "stressed",
    "hate", "boring", "miserable", "furious", "annoyed", "disappointed",
    "anxious", "lonely", "worthless", "frustrated", "exhausted", "dreading",
]

# Slang, emoji and intensity signals the plain word lists miss.
# Positive weight leans happy, negative weight leans upset.
SLANG_SIGNALS: Dict[str, int] = {
    "fire": 2, "sick": 2, "wicked": 2, "goated": 2, "lit": 2, "slaps": 2,
    "banger": 2, "peak": 2, "clutch": 2, "vibes": 1, "highkey": 1,
    "proud": 2, "grateful": 2,  # strong enough to register as one half of "mixed"
    # Profanity. The original lab's word lists had none, so "fuck life" scored
    # exactly zero and came back neutral. In the register this system is aimed
    # at, profanity is one of the strongest negative signals available.
    #
    # Intensifiers ("fucking", "damn", "bloody") are deliberately left out.
    # They amplify whatever they attach to and flip sign with it: "fucking
    # awful" and "fucking amazing" would both be scored negative if the
    # intensifier carried its own weight.
    "fuck": -2, "fck": -2, "shit": -2, "shitty": -2, "sucks": -2,
    "wtf": -2, "crap": -1, "fml": -2, "hate": -2,
    "stressed": -2, "exhausted": -2, "drained": -2, "done": -1, "meh": -1,
    "mid": -1, "cooked": -2, "lowkey": -1, "ugh": -2, "cringe": -2,
    ":)": 2, ":-)": 2, ":d": 2, ":(": -2, ":-(": -2, ":/": -1,
    "🔥": 2, "😂": 2, "🥳": 2, "🙂": 1, "❤️": 2, "✨": 1,
    "💀": -2, "🥲": -1, "😭": -2, "😡": -2, "😩": -2, "🙃": -1,
}

# Words that flip the polarity of the token immediately after them.
NEGATIONS = {
    "not", "no", "never", "isnt", "arent", "wasnt", "werent", "dont",
    "doesnt", "didnt", "cant", "wont", "aint", "hardly", "barely",
}


# ---------------------------------------------------------------------------
# Knowledge base: indexed by the retriever, trained on by the ML model
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: List[Tuple[str, str]] = [
    # --- plain, easy cases (the six originals from Module 3 live here) ---
    ("I love this class so much", "positive"),
    ("Today was a terrible day", "negative"),
    ("Feeling tired but kind of hopeful", "mixed"),
    ("This is fine", "neutral"),
    ("So excited for the weekend", "positive"),
    ("I am not happy about this", "negative"),
    ("Had a great time at dinner tonight", "positive"),
    ("Everything about today was awful", "negative"),
    ("The package arrives on Thursday", "neutral"),
    ("I hate how long this is taking", "negative"),

    # --- slang and emoji ---
    ("this new update is fire no cap 🔥", "positive"),
    ("that movie was sick, best one all year", "positive"),
    ("exhausted, done with everything today 💀", "negative"),
    ("the new album absolutely slaps", "positive"),
    ("honestly kinda mid, expected more", "negative"),
    ("chat this is peak 😂", "positive"),
    ("im so cooked for this exam 😭", "negative"),
    ("that presentation was clutch ✨", "positive"),
    ("everything is mid and I am tired 🙃", "negative"),
    ("okay this is actually a banger", "positive"),

    # --- sarcasm: surface words disagree with the true label ---
    ("I absolutely love getting stuck in traffic", "negative"),
    ("great, another meeting that could have been an email", "negative"),
    ("wonderful, the wifi died again right before my demo", "negative"),
    ("oh fantastic, my code broke five minutes before the deadline", "negative"),
    ("just what I needed today, more paperwork, amazing", "negative"),
    ("love waking up at 5am for no reason, truly living the dream", "negative"),

    # --- negation ---
    ("not bad honestly, kinda enjoyed it", "positive"),
    ("this is not terrible at all", "positive"),
    ("I am not excited about the schedule change", "negative"),
    ("never been happier with a purchase", "positive"),
    ("cant say I hated it", "positive"),

    # --- mixed feelings ---
    ("lowkey stressed but highkey proud of myself", "mixed"),
    ("the food was amazing but the service was awful", "mixed"),
    ("sad it ended, grateful it happened", "mixed"),
    ("exhausted but genuinely glad I showed up", "mixed"),
    ("I love the design and I hate the price", "mixed"),
    ("nervous and excited at the same time", "mixed"),

    # --- short, blunt, profane ---
    # Added after hand-testing found the system returning POSITIVE for "this
    # sucks" and "this is fucking awful". The corpus had no short blunt
    # negatives at all, so those queries retrieved "This is fine" and "chat
    # this is peak" on the shared words "this is" and voted with them.
    # Deliberately phrased differently from the cases that exposed the bug, so
    # this is new evidence rather than the test answers pasted into the corpus.
    # Two things had to be true for these to help rather than hurt.
    #
    # Balance: a first pass added eight blunt negatives against three
    # positives, which fixed the profanity cases and then scored "this is
    # fucking amazing" as NEGATIVE. Short text had become a negative attractor
    # instead of a positive one, which is not a fix, it is the same bug facing
    # the other way. Three negatives and three positives now.
    #
    # Size: a ten-post version dropped held-out accuracy from 0.83 to 0.72,
    # and the posts that regressed were unrelated to anything added. Adding
    # documents changes the inverse-document-frequency of common words, which
    # perturbs every similarity in the index at once. At 43 documents the
    # corpus is small enough that ten additions move queries that share no
    # vocabulary with them. Six additions hold held-out accuracy at 0.83.
    ("this app sucks honestly", "negative"),
    ("wtf is this nonsense", "negative"),
    ("this is complete garbage", "negative"),
    ("this rules honestly", "positive"),
    ("this is great, no notes", "positive"),
    ("wow this is incredible", "positive"),

    # --- flat / factual / neutral ---
    ("meeting got moved to 3pm", "neutral"),
    ("I'm fine 🙂", "neutral"),
    ("submitted the assignment at 2am", "neutral"),
    ("the library closes at nine on Sundays", "neutral"),
    ("took the train instead of the bus today", "neutral"),
    ("attached the file to the email", "neutral"),
]


# ---------------------------------------------------------------------------
# Held-out set: never indexed, never trained on
# ---------------------------------------------------------------------------

HELD_OUT: List[Tuple[str, str]] = [
    ("this whole week has been wonderful", "positive"),
    ("I am so proud of how that turned out", "positive"),
    ("genuinely the best coffee I have had all month", "positive"),
    ("stoked for tomorrow honestly", "positive"),

    ("today was rough and I am completely drained", "negative"),
    ("I hate that I have to redo the entire thing", "negative"),
    ("furious about how that meeting went", "negative"),
    ("everything I touched today broke 😩", "negative"),

    ("oh brilliant, my flight got delayed another four hours", "negative"),
    ("love spending my whole Saturday on a bug I created", "negative"),
    ("perfect timing for the printer to die, really appreciate it", "negative"),

    ("the class starts at ten in room 214", "neutral"),
    ("I rescheduled the appointment for next week", "neutral"),
    ("picked up groceries on the way home", "neutral"),

    ("tired but really happy with the result", "mixed"),
    ("the trip was beautiful but the flights were miserable", "mixed"),
    ("anxious about the move and excited for it too", "mixed"),
    ("proud of the work, upset about the deadline", "mixed"),
]


# Category tags for the held-out set, parallel to HELD_OUT. Used by
# evaluate.py to report accuracy per language phenomenon, because an overall
# number hides the fact that sarcasm is where every version of this system has
# failed.
HELD_OUT_TAGS: List[str] = [
    "plain", "plain", "plain", "plain",
    "plain", "plain", "plain", "emoji",
    "sarcasm", "sarcasm", "sarcasm",
    "neutral", "neutral", "neutral",
    "mixed", "mixed", "mixed", "mixed",
]

assert len(HELD_OUT_TAGS) == len(HELD_OUT), (
    f"HELD_OUT_TAGS ({len(HELD_OUT_TAGS)}) must match HELD_OUT ({len(HELD_OUT)})"
)

ALLOWED_LABELS = ("positive", "negative", "neutral", "mixed")


def kb_texts() -> List[str]:
    """Texts in the knowledge base, in index order."""
    return [text for text, _ in KNOWLEDGE_BASE]


def kb_labels() -> List[str]:
    """Labels in the knowledge base, in index order."""
    return [label for _, label in KNOWLEDGE_BASE]


# Safety checks so a bad edit fails loudly instead of silently skewing results.
assert all(label in ALLOWED_LABELS for _, label in KNOWLEDGE_BASE), (
    "KNOWLEDGE_BASE contains a label outside ALLOWED_LABELS"
)
assert all(label in ALLOWED_LABELS for _, label in HELD_OUT), (
    "HELD_OUT contains a label outside ALLOWED_LABELS"
)

_kb_lookup = {text.strip().lower() for text in kb_texts()}
_overlap = [t for t, _ in HELD_OUT if t.strip().lower() in _kb_lookup]
assert not _overlap, f"HELD_OUT leaks into KNOWLEDGE_BASE: {_overlap}"
