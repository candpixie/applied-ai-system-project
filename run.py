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
from moodlens.memory import RejectedLesson

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
        choices=["analyze", "demo", "interactive", "teach", "memory"],
        help=(
            "analyze one string, run the scripted demo, start a REPL, "
            "teach a correction, or show what has been learned"
        ),
    )
    parser.add_argument("text", nargs="?", help="text to analyze (for 'analyze')")
    parser.add_argument(
        "label",
        nargs="?",
        help="correct label (for 'teach'): positive, negative, neutral or mixed",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="load previously taught corrections when analyzing",
    )
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

    # Teaching and inspecting memory always use it; analysis only if asked, so
    # the default behaviour of this CLI stays the shipped behaviour.
    wants_memory = args.learn or args.command in {"teach", "memory"}
    agent = MoodAgent(use_llm=args.use_llm, k=args.k, use_memory=wants_memory)

    if args.command == "memory":
        store = agent.store
        print(f"learning store: {store.path}")
        print(f"lessons: {len(store)}  {store.stats() or ''}")
        for text, label in store.examples():
            print(f'  [{label:<8}] "{text}"')
        return 0

    if args.command == "teach":
        if args.text is None or args.label is None:
            print('usage: run.py teach "some post" negative', file=sys.stderr)
            return 2
        before = agent.analyze(args.text).label
        try:
            result = agent.teach(args.text, args.label)
        except RejectedLesson as exc:
            print(f"rejected: {exc}", file=sys.stderr)
            return 1

        print(f'was  : {before}')
        print(f'now  : {agent.analyze(args.text).label}')
        print(f'kept : {result["kept"]}')
        print(f'note : {result["reason"]}')
        return 0

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
