"""
Meeting Memory Module
Tracks recurring meetings and provides context from previous instances.
Enables the AI to reference past decisions, carry-forward action items,
and remind the user what happened last time in the same meeting series.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class MeetingMemory:
    """
    Stores summaries of past meetings and retrieves relevant history
    for recurring meetings (matched by title similarity).
    """

    def __init__(
        self,
        memory_dir: str,
        max_history: int = 3,
        similarity_threshold: float = 0.7,
    ):
        """
        Args:
            memory_dir: Directory to store meeting memory JSON files.
            max_history: How many past instances to include as context.
            similarity_threshold: How similar two meeting titles need to be
                                  to be considered the same recurring meeting (0-1).
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold

    def store_meeting(self, meeting_synthesis: dict, calendar_title: str | None = None):
        """
        Store a meeting's key details for future reference.
        Uses the calendar title (if available) as the canonical series name,
        falling back to the LLM-inferred title.
        """
        title = calendar_title or meeting_synthesis.get("meeting_title", "Untitled Meeting")
        series_key = self._make_series_key(title)

        # Load existing history for this series
        history = self._load_series(series_key)

        # Build a compact summary to store (not the full raw output)
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": title,
            "attendees": meeting_synthesis.get("attendees", []),
            "decisions": meeting_synthesis.get("decisions", []),
            "action_items": meeting_synthesis.get("action_items", []),
            "my_next_steps": meeting_synthesis.get("my_next_steps", []),
            "key_topics": meeting_synthesis.get("key_topics", [])[:5],  # Top 5 only
            "open_questions": meeting_synthesis.get("questions_followups", []),
        }

        history.append(entry)

        # Keep only the most recent entries
        if len(history) > self.max_history * 2:
            history = history[-(self.max_history * 2):]

        self._save_series(series_key, history)
        logger.info(f"Stored meeting memory for series: {title} ({len(history)} entries)")

    def get_context_for_meeting(self, title: str) -> str | None:
        """
        Retrieve past context for a recurring meeting.
        Returns a formatted string for prompt injection, or None if no history.
        """
        series_key = self._find_matching_series(title)
        if not series_key:
            return None

        history = self._load_series(series_key)
        if not history:
            return None

        # Get the most recent entries (up to max_history)
        recent = history[-self.max_history:]

        # Format for prompt injection
        parts = [f"RECURRING MEETING HISTORY — This meeting has happened before. Here's context from the last {len(recent)} instance(s):\n"]

        for entry in recent:
            parts.append(f"### {entry['date']} — {entry['title']}")

            if entry.get("decisions"):
                parts.append("Decisions made:")
                for d in entry["decisions"]:
                    parts.append(f"  - {d}")

            if entry.get("action_items"):
                parts.append("Action items assigned:")
                for a in entry["action_items"]:
                    parts.append(f"  - {a}")

            if entry.get("my_next_steps"):
                parts.append("User's action items from last time:")
                for s in entry["my_next_steps"]:
                    parts.append(f"  - {s}")

            if entry.get("open_questions"):
                parts.append("Open questions/follow-ups:")
                for q in entry["open_questions"]:
                    parts.append(f"  - {q}")

            parts.append("")  # Blank line between entries

        parts.append(
            "USE THIS CONTEXT TO:\n"
            "- Note if previous action items were addressed in today's meeting\n"
            "- Flag any open questions from last time that were resolved today\n"
            "- Provide continuity (e.g., 'Continued discussion from last week on X')\n"
            "- Do NOT re-list old action items as new ones unless they were explicitly re-assigned today\n"
        )

        return "\n".join(parts)

    def _find_matching_series(self, title: str) -> str | None:
        """Find an existing series that matches this title."""
        target_key = self._make_series_key(title)

        # Exact match first
        if (self.memory_dir / f"{target_key}.json").exists():
            return target_key

        # Fuzzy match against existing series
        best_match = None
        best_score = 0.0

        for series_file in self.memory_dir.glob("*.json"):
            series_key = series_file.stem
            score = SequenceMatcher(None, target_key, series_key).ratio()
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = series_key

        return best_match

    def _make_series_key(self, title: str) -> str:
        """
        Create a normalized key from a meeting title.
        Strips dates, numbers, and normalizes for fuzzy matching.
        """
        # Lowercase
        key = title.lower().strip()
        # Remove common date patterns (2024-01-15, Jan 15, etc.)
        key = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', key)
        key = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d+', '', key)
        # Remove ordinals and standalone numbers
        key = re.sub(r'\b\d+(st|nd|rd|th)?\b', '', key)
        # Remove special characters, keep spaces
        key = re.sub(r'[^\w\s]', '', key)
        # Collapse whitespace
        key = re.sub(r'\s+', '-', key.strip())
        # Remove trailing/leading hyphens
        key = key.strip('-')

        return key or "untitled"

    def _load_series(self, series_key: str) -> list[dict]:
        """Load the history for a meeting series."""
        filepath = self.memory_dir / f"{series_key}.json"
        if not filepath.exists():
            return []

        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load meeting memory {filepath}: {e}")
            return []

    def _save_series(self, series_key: str, history: list[dict]):
        """Save the history for a meeting series."""
        filepath = self.memory_dir / f"{series_key}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(history, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save meeting memory {filepath}: {e}")
