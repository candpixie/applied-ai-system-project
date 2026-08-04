# Human evaluation

Two things here need a person rather than a script: deciding what the correct
label actually is, and deciding whether the system's stated reason for a label
is a real reason or just a plausible-sounding one.

Reviewer: project author. Every number below is reproducible from
`python evaluate.py` (see `reports/evaluation_report.md`, section 6) and
`reports/needs_review_queue.txt`.

---

## 1. Labeling protocol

The 18 held-out posts were written and hand-labeled **before** any of them were
run through the system, so the labels could not be influenced by what the model
said. Rules used while labeling:

| Rule | Meaning |
| --- | --- |
| Sarcasm takes the speaker's real feeling | "oh brilliant, my flight got delayed" is `negative`, not `positive` |
| `mixed` needs two genuinely opposed feelings | not just a hedge or a mild qualifier |
| `neutral` means factual or flat | not "mildly positive"; if there is a feeling, it is not neutral |
| Label the writer's mood, not the event | "picked up groceries" is `neutral` even if groceries are nice |

### Posts I found genuinely hard to label

These are the ones I changed my mind on at least once. Worth recording, because
a label I am unsure about is not a fair thing to grade a model against.

| Post | Label chosen | Why it was hard |
| --- | --- | --- |
| "stoked for tomorrow honestly" | positive | "honestly" reads as slightly defensive, which pulls toward mixed |
| "I rescheduled the appointment for next week" | neutral | rescheduling can imply relief or dread, but nothing in the text says which |
| "anxious about the move and excited for it too" | mixed | close to positive, since the excitement is stated second |
| "proud of the work, upset about the deadline" | mixed | two clean opposed feelings, but a reader anchored on the second clause could call it negative |

---

## 2. Does the review flag land on the right posts?

The system routed **4 of 18** held-out posts to `needs_review`. A review queue
is only worth having if the things in it really are the hard things.

| Post | Agent label | Confidence | Human label | Correct? | Was flagging it right? |
| --- | --- | --- | --- | --- | --- |
| love spending my whole Saturday on a bug I created | negative | 0.41 | negative | Yes | Yes. Right answer, but reached over an explicit objection from the rules component, which read "love" as positive. Low confidence is honest. |
| perfect timing for the printer to die, really appreciate it | neutral | 0.44 | negative | No | Yes. Sarcasm with no lexicon signal at all. It flagged this instead of asserting it. |
| I rescheduled the appointment for next week | neutral | 0.44 | neutral | Yes | Yes. Correct, but on thin evidence: retrieval actively voted `negative` off a bad neighbour and was outvoted. |
| proud of the work, upset about the deadline | positive | 0.41 | mixed | No | Yes. The lexicon weighted "proud" at 2 and "upset" at 1, so the two feelings did not balance and `mixed` never triggered. |

**Flagged accuracy: 2/4 (0.50). Unflagged accuracy: 13/14 (0.93).**

That gap is the point. The flag is carrying real information, not decoration.

### The error that got through

Three held-out posts were wrong overall. Two were flagged. One was not:

| Post | Agent | Truth | Confidence | Status |
| --- | --- | --- | --- | --- |
| stoked for tomorrow honestly | negative | positive | 0.49 | ok |

"stoked" is not in any lexicon, is not in the training vocabulary, and has no
similar post in the knowledge base. The ML model guessed `negative` off
"honestly", which appears in the knowledge base in "honestly kinda mid,
expected more". Two weak components happened to agree, so noisy-OR fusion read
their agreement as corroboration when it was really the same absence of
knowledge twice over.

This is the sharpest limitation I found: **fusion assumes the components fail
independently, and they do not.** When a word is unknown, it tends to be
unknown to all three at once. Nothing in the current design detects that.

---

## 3. Explanation quality

If the stated reason does not match what actually drove the decision, the
explanation is decoration and users will trust the wrong things.

Criteria: does the rationale name the components that actually decided the
label, and would a non-technical reader understand why?

| Post | Rationale given | Faithful to the computation? | Understandable? |
| --- | --- | --- | --- |
| "oh perfect, my laptop died right before the deadline" | "retrieval, ml agreed on 'negative' (82% of vote weight)", with two sarcastic neighbours quoted at similarity 0.47 and 0.41 | Yes. The quoted neighbours are exactly the ones voted on. | Yes. Seeing the two similar posts makes the sarcasm call obvious. |
| "not bad at all, I actually had fun" | all three agree `positive` (100%), rules shows the token `not+bad` | Yes. `not+bad` is literally the negation-flipped token. | Yes, and it shows the negation was handled rather than missed. |
| "the shipment arrives on Tuesday" | all three agree `neutral`, nearest neighbour "The package arrives on Thursday" at 0.62 | Yes. | Yes. The neighbour is nearly the same sentence. |
| "quarterly synergy realignment scheduled for Q3" | "only 1 of 6 words are in the training vocabulary" and "no sufficiently similar labeled example in the knowledge base" | Yes, and unusually useful: it says *why* it does not know. | Yes. The best explanation the system produces. |
| "I rescheduled the appointment for next week" | below-threshold warning, best guess `neutral` | Partly. It reports low confidence but never says the retrieval evidence was actively misleading. | Weaker. A reader could take `neutral` as a finding rather than an absence of evidence. |

**Verdict: 4/5 faithful, 4/5 clear.**

The weak spot is that `neutral` is overloaded. It means both "this is calm
factual text" and "nothing fired, so here is the default". Those are different
claims and should not share an output value. Splitting them is the single
highest-value change I would make next, and no accuracy metric would have
surfaced it. It only showed up from reading the explanations.

---

## 4. What this review actually changed

- The `proud of the work, upset about the deadline` miss traces back to a
  lexicon weight I set by hand: promoting "proud" to weight 2 fixed one test
  and quietly broke `mixed` detection for this post, because `mixed` requires
  both sides to reach 2 and "upset" is only 1. Hand-tuned weights have
  non-local effects and I did not check for them.
- The correlated-failure problem in section 2 was not visible in any aggregate
  number. It came from reading one wrong answer closely.
- The overloaded `neutral` came out of section 3, not from any metric.
