"""
Weekly Review Generator
Runs Friday at 3:30pm (or manually triggered).
Aggregates the week's meeting notes, daily notes, and action items
into a single "Week in Review" note.
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class WeeklyReviewGenerator:
    """Generates a weekly review note from the past week's data."""

    def __init__(
        self,
        vault_path: str,
        daily_notes_folder: str = "06_Daily Notes",
        meetings_folder: str = "07_Meetings",
        timezone: str = "America/Los_Angeles",
        llm_provider: str = "gemini",
        gemini_api_key: str = "",
        ollama_model: str = "llama3.1:8b",
        user_name: str = "",
    ):
        self.vault_path = Path(vault_path)
        self.daily_notes_dir = self.vault_path / daily_notes_folder
        self.meetings_dir = self.vault_path / meetings_folder
        self.timezone = ZoneInfo(timezone)
        self.llm_provider = llm_provider
        self.gemini_api_key = gemini_api_key
        self.ollama_model = ollama_model
        self.user_name = user_name

    def generate(self) -> Path | None:
        """
        Generate a weekly review note for the current week (Mon-Fri).
        Saves to the daily notes folder with the week's date range.
        """
        now = datetime.now(self.timezone)
        # Find Monday of this week
        monday = now - timedelta(days=now.weekday())
        friday = monday + timedelta(days=4)

        week_label = f"{monday.strftime('%b %d')} - {friday.strftime('%b %d, %Y')}"
        date_str = now.strftime("%Y-%m-%d")

        # Collect data from the week
        meetings_attended = self._get_weeks_meetings(monday, friday)
        daily_note_data = self._get_weeks_daily_data(monday, friday)
        completed_items = daily_note_data.get("completed", [])
        incomplete_items = daily_note_data.get("incomplete", [])
        decisions = daily_note_data.get("decisions", [])

        # Build LLM prompt for intelligent synthesis
        prompt = self._build_review_prompt(
            week_label=week_label,
            meetings=meetings_attended,
            completed=completed_items,
            incomplete=incomplete_items,
            decisions=decisions,
        )

        # Generate review via LLM
        try:
            review_content = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"LLM failed for weekly review: {e}")
            review_content = None

        # Build the note
        note_content = self._build_note(
            week_label=week_label,
            date_str=date_str,
            meetings=meetings_attended,
            completed=completed_items,
            incomplete=incomplete_items,
            llm_review=review_content,
        )

        # Save to vault
        month_folder = now.strftime("%B %Y")
        note_dir = self.daily_notes_dir / month_folder
        note_path = note_dir / f"{date_str}_weekly-review.md"

        try:
            note_dir.mkdir(parents=True, exist_ok=True)
            note_path.write_text(note_content)
            logger.info(f"Weekly review generated: {note_path.name}")
            return note_path
        except Exception as e:
            logger.error(f"Failed to write weekly review: {e}")
            return None

    def _get_weeks_meetings(self, monday: datetime, friday: datetime) -> list[dict]:
        """Get all meeting notes from this week."""
        meetings = []
        if not self.meetings_dir.exists():
            return meetings

        monday_str = monday.strftime("%Y-%m-%d")
        friday_str = friday.strftime("%Y-%m-%d")

        for note_file in sorted(self.meetings_dir.glob("*.md")):
            # Check if file date is within this week
            name = note_file.name
            if len(name) >= 10:
                file_date = name[:10]
                if monday_str <= file_date <= friday_str:
                    # Extract title from frontmatter or first H1
                    try:
                        content = note_file.read_text()
                        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
                        title = title_match.group(1) if title_match else note_file.stem
                        date_match = re.search(r"date: (\d{4}-\d{2}-\d{2})", content)
                        note_date = date_match.group(1) if date_match else file_date
                        meetings.append({
                            "title": title,
                            "date": note_date,
                            "filename": note_file.name,
                        })
                    except Exception:
                        continue

        return meetings

    def _get_weeks_daily_data(self, monday: datetime, friday: datetime) -> dict:
        """Extract completed/incomplete items from this week's daily notes."""
        completed = []
        incomplete = []
        decisions = []

        for i in range(5):  # Mon-Fri
            day = monday + timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            month_folder = day.strftime("%B %Y")
            note_path = self.daily_notes_dir / month_folder / f"{date_str}.md"

            if not note_path.exists():
                continue

            try:
                content = note_path.read_text()
                # Extract completed items (checked checkboxes)
                for match in re.finditer(r"- \[x\] (.+?)(?:\s*✅.*)?$", content, re.MULTILINE):
                    item = match.group(1).strip()
                    if item:
                        completed.append(f"{day.strftime('%A')}: {item}")

                # Extract incomplete items (unchecked checkboxes)
                for match in re.finditer(r"- \[ \] (.+)$", content, re.MULTILINE):
                    item = match.group(1).strip()
                    if item and item != "-":
                        incomplete.append(f"{day.strftime('%A')}: {item}")

            except Exception:
                continue

        return {
            "completed": completed,
            "incomplete": incomplete,
            "decisions": decisions,
        }

    def _build_review_prompt(self, week_label: str, meetings: list, completed: list, incomplete: list, decisions: list) -> str:
        """Build the LLM prompt for weekly review synthesis."""
        meetings_str = "\n".join(f"- {m['date']}: {m['title']}" for m in meetings) or "No meetings recorded"
        completed_str = "\n".join(f"- {item}" for item in completed[:30]) or "None tracked"
        incomplete_str = "\n".join(f"- {item}" for item in incomplete[:20]) or "None"

        return f"""You are an executive assistant helping {self.user_name or 'the user'} reflect on their work week.

WEEK: {week_label}

MEETINGS ATTENDED ({len(meetings)}):
{meetings_str}

COMPLETED ITEMS:
{completed_str}

ITEMS STILL INCOMPLETE:
{incomplete_str}

Generate a brief weekly review with these sections:
1. **Week Summary** (2-3 sentences about what the week was about)
2. **Key Wins** (things that went well or got done)
3. **Patterns** (what kept slipping? what took more time than expected?)
4. **Carry into Next Week** (top 3 priorities for Monday)
5. **Energy Check** (based on the volume and type of meetings, how intense was this week?)

Be honest and direct. This is a private reflection, not a status report.
Keep it concise — this should take 2 minutes to read.
"""

    def _build_note(self, week_label: str, date_str: str, meetings: list, completed: list, incomplete: list, llm_review: str | None) -> str:
        """Build the final weekly review markdown note."""
        meetings_table = "| Date | Meeting |\n| ---- | ------- |\n"
        if meetings:
            for m in meetings:
                meetings_table += f"| {m['date']} | [[{m['filename'].replace('.md', '')}|{m['title']}]] |\n"
        else:
            meetings_table += "| — | No meetings recorded |\n"

        review_section = llm_review or "_LLM review unavailable_"

        return f"""---
tags: weekly-review
date: {date_str}
week: "{week_label}"
---

# Weekly Review — {week_label}

---

{review_section}

---

## Meetings This Week ({len(meetings)})

{meetings_table}

---

## Completed This Week ({len(completed)})

{"".join(f'- [x] {item}{chr(10)}' for item in completed) if completed else "- None tracked"}

---

## Still Open ({len(incomplete)})

{"".join(f'- [ ] {item}{chr(10)}' for item in incomplete) if incomplete else "- All clear!"}

---

## Notes

"""

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM."""
        if self.llm_provider == "gemini" and self.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        else:
            import json
            import urllib.request
            payload = json.dumps({
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
