"""MoodLens: a retrieval-augmented, self-checking mood classification system."""

from .agent import MoodAgent
from .ml_model import MLMoodModel
from .mood_analyzer import MoodAnalyzer
from .retrieval import MoodRetriever
from .signals import Decision, Signal

__version__ = "1.0.0"

__all__ = [
    "MoodAgent",
    "MoodAnalyzer",
    "MoodRetriever",
    "MLMoodModel",
    "Decision",
    "Signal",
]
