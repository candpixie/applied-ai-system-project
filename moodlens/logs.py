"""
Structured logging.

Every decision the agent makes is appended to `logs/decisions.jsonl` as one
JSON object per line, and human-readable events go to stderr through the
standard `logging` module. JSONL because the point of logging here is to be
able to answer "how often did the agent need a repair pass last week" with a
one-line script, not to read prose.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(os.environ.get("MOODLENS_LOG_DIR", "logs"))
DECISION_LOG = LOG_DIR / "decisions.jsonl"

_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Console logger. Level follows MOODLENS_LOG_LEVEL, default INFO."""
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("moodlens")
    level = os.environ.get("MOODLENS_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [moodlens] %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
    _logger = logger
    return logger


def log_decision(record: Dict[str, Any]) -> None:
    """Append one decision record to the JSONL log.

    Logging must never take the application down, so a filesystem failure here
    is reported and swallowed rather than raised.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with DECISION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        get_logger().warning("could not write decision log: %s", exc)


def read_decisions() -> list[Dict[str, Any]]:
    """Read back every logged decision. Skips lines that are not valid JSON."""
    if not DECISION_LOG.exists():
        return []

    records = []
    with DECISION_LOG.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                get_logger().warning("skipping malformed log line")
    return records
