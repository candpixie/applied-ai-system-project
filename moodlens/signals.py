"""
Shared data shapes.

Every component that can express an opinion about a piece of text returns a
`Signal`. Keeping one shape means the agent can fuse rule-based output,
ML output and retrieval output without special-casing each one.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Signal:
    """One component's opinion about a piece of text.

    Attributes:
        source: which component produced this ("rules", "ml", "retrieval", "llm").
        label: predicted mood label, or None if the component had nothing to say.
        confidence: 0.0-1.0. Components must not report 1.0; nothing is certain.
        rationale: one human-readable sentence explaining the call.
        evidence: structured detail (matched tokens, neighbours, probabilities).
    """

    source: str
    label: str | None
    confidence: float
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "evidence": self.evidence,
        }


@dataclass
class Decision:
    """The agent's final answer for one input."""

    text: str
    label: str
    confidence: float
    status: str  # "ok" | "needs_review" | "blocked" | "safety_hold"
    rationale: str
    signals: List[Signal] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    repaired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "rationale": self.rationale,
            "repaired": self.repaired,
            "signals": [s.to_dict() for s in self.signals],
            "trace": self.trace,
        }

    def pretty(self) -> str:
        """Human-readable block used by the CLI and the sample logs."""
        lines = [
            f'input      : "{self.text}"',
            f"label      : {self.label}",
            f"confidence : {self.confidence:.2f}",
            f"status     : {self.status}",
            f"rationale  : {self.rationale}",
        ]
        for signal in self.signals:
            lines.append(
                f"  - {signal.source:<9} {str(signal.label):<9} "
                f"conf={signal.confidence:.2f}  {signal.rationale}"
            )
        if self.trace:
            lines.append("  trace: " + " -> ".join(self.trace))
        return "\n".join(lines)
