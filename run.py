#!/usr/bin/env python3
"""
MoodLens command line interface.

    python run.py analyze "not bad honestly, kinda enjoyed it"
    python run.py demo
    python run.py interactive
    python run.py analyze "..." --json --use-llm
"""

import argparse
import json
import sys
from typing import List

from moodlens import MoodAgent

# None of these appear in the knowledge base, so the demo shows the system
# generalizing rather than looking up an answer it was handed. The last three
# exist to show what happens when the system should not answer confidently.
DEMO_INPUTS: List[str] = [
    "oh perfect, my laptop died right before the deadline",   # sarcasm
    "not bad at all, I actually had fun",                     # negation
    "grateful for the trip but the flights were exhausting",  # mixed feelings
    "the shipment arrives on Tuesday",                        # flat / factual
    "quarterly synergy realignment scheduled for Q3",         # out of domain
    "honestly I want to die, nothing is helping",             # safety hold
    "",                                                       # blocked
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moodlens",
        description="Classify the mood of short text with a self-checking AI agent.",
    )
    parser.add_argument(
        "command",
        choices=["analyze", "demo", "interactive"],
        help="analyze one string, run the scripted demo, or start a REPL",
    )
    parser.add_argument("text", nargs="?", help="text to analyze (for 'analyze')")
    parser.add_argument(
        "--json", action="store_true", help="emit the full decision as JSON"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="allow the optional LLM tie-breaker (needs ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "-k", type=int, default=3, help="how many neighbours to retrieve (default 3)"
    )
    return parser


def emit(decision, as_json: bool) -> None:
    if as_json:
        print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(decision.pretty())


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = MoodAgent(use_llm=args.use_llm, k=args.k)

    if args.command == "analyze":
        if args.text is None:
            print("error: 'analyze' needs a text argument", file=sys.stderr)
            return 2
        emit(agent.analyze(args.text), args.json)
        return 0

    if args.command == "demo":
        print("=== MoodLens demo ===\n")
        for text in DEMO_INPUTS:
            emit(agent.analyze(text), args.json)
            print()
        return 0

    # interactive
    print("=== MoodLens (interactive) ===")
    print("Type a post to classify. Blank line or 'quit' exits.\n")
    while True:
        try:
            entry = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if entry == "" or entry.lower() == "quit":
            print("bye")
            return 0
        emit(agent.analyze(entry), args.json)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
