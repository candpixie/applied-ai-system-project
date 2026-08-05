#!/usr/bin/env python3
"""
Three-minute Demo Day script.

Four beats, press Enter between each so you control the pacing:

    python demo_day.py             # press Enter between beats
    python demo_day.py --auto      # run straight through, no keypresses

Everything is loaded once up front, so each beat appears instantly.
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

from moodlens import MoodAgent, MoodAnalyzer

RULE = "=" * 78


def beat(title: str, auto: bool) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")
    if not auto:
        try:
            input("\n[Enter]")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="no keypresses")
    args = parser.parse_args()
    auto = args.auto

    print("\nLoading MoodLens...", end=" ", flush=True)
    agent = MoodAgent()
    rules = MoodAnalyzer()
    print("ready.\n")
    if not auto:
        input("[Enter to start]")

    # ------------------------------------------------------------------
    beat("BEAT 1  The original lab could not read sarcasm", auto)
    sarcasm = "oh perfect, my laptop died right before the deadline"
    print(f'\ninput: "{sarcasm}"\n')
    print(f"  Module 3 lexicon says : {rules.analyze(sarcasm).label}")
    print("  correct answer is     : negative\n")
    print("  It sees the word 'perfect'. It has no way to know that is sarcasm.")

    # ------------------------------------------------------------------
    beat("BEAT 2  Retrieval fixes it, and shows its evidence", auto)
    decision = agent.analyze(sarcasm)
    print()
    print(decision.pretty())
    print(
        "\n  It found two labeled sarcastic posts with the same shape and"
        "\n  outvoted the lexicon. Sarcasm went from 0/3 to 2/3 on held-out data."
    )

    # ------------------------------------------------------------------
    beat("BEAT 3  It knows when to stop", auto)
    for text in [
        "honestly I want to die, nothing is helping",
        "quarterly synergy realignment scheduled for Q3",
    ]:
        result = agent.analyze(text)
        print(f'\ninput: "{text}"')
        print(f"  status     : {result.status}")
        print(f"  label      : {result.label}   confidence: {result.confidence:.2f}")
        for line in textwrap.wrap(result.rationale, width=74)[:3]:
            print(f"  {line}")
    print(
        "\n  Crisis language: no classifier runs at all, support resources instead."
        "\n  Out of domain : two of three components abstain, flagged for a human."
    )

    # ------------------------------------------------------------------
    beat("BEAT 4  The numbers", auto)
    report = Path("reports/evaluation_report.md")
    if report.exists():
        text = report.read_text(encoding="utf-8")
        table = re.search(r"\| Configuration.*?\n\n", text, re.S)
        print()
        print(table.group(0).strip() if table else "(run: python evaluate.py)")
    print(
        "\n  Confidence is calibrated: 1.00 accuracy when it claims >= 0.60,"
        "\n  0.73 below it. 108 tests. Runs offline, no API key."
        "\n\n  github.com/candpixie/applied-ai-system-project\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
