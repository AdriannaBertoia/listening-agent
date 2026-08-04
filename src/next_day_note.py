"""
Next Day Note Generator
Creates the next workday's daily note FULLY filled in using LLM intelligence:
- Top 3 priorities (inferred from deadlines, carry-forward items, meetings)
- Intention for the day
- Meetings table with correct PST times from calendar
- Time-blocked schedule built around actual meetings
- Meeting prep checklist
- Yesterday's follow-ups (carry-forward from today)
- To-dos categorized by urgency
- Team context carried over
- EdTech brief

Runs M-F only. On Friday, generates Monday's note.
All times are in Pacific Time (America/Los_Angeles).
"""

import json
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DAILY_NOTE_PROMPT = """You are an executive assistant helping someone with ADHD plan their next workday.
You have the following information about tomorrow ({target_date}, {day_of_week}):

USER: {user_name}

CARRY-FORWARD TO-DOS (incomplete from today — these MUST appear in tomorrow's note):
{carry_forward_section}

TOMORROW'S CALENDAR (all times are Pacific Time):
{meetings_section}

TEAM CONTEXT FROM TODAY:
{team_context_section}

Your job is to produce a COMPLETE daily note plan. Fill in EVERY section thoughtfully.

RULES:
- Top 3 Priorities: Pick the 3 most important things based on deadlines, meeting prep, and carry-forward items.
  If something has a deadline tomorrow or this week, it's a top priority.
- Intention: One sentence about what a successful day looks like (based on what's scheduled).
- Time-Blocked Schedule: Build a realistic schedule around the actual meetings. Include:
  - WORK HOURS ARE 9:00 AM - 4:00 PM PST ONLY. Do NOT schedule anything before 9am or after 4pm.
  - Before 9am and after 4pm is family time (Arlo drop-off 8am, pick-up 4:15pm).
  - 30 min before important meetings for prep
  - Deep work blocks in gaps
  - Lunch break (12:00-1:00 PM)
  - Wrap-up / planning block at 3:30-4:00 PM (end of day)
  - All times in Pacific Time, format as "HH:MM AM/PM"
  - Items marked with a time (e.g. "@ 9:00 AM PST") MUST be scheduled at that exact time.
- Meeting Prep: For each meeting tomorrow, suggest 1-2 prep actions.
- To-Dos: Categorize carry-forward items into Must-do (deadline-driven), Should-do (important but flexible), 
  and Carry-forward (can wait until later this week). Add any new to-dos implied by meetings.
- Team Context: Carry over what's relevant. Remove stale items.
- DO NOT fabricate deadlines or meetings that aren't in the data.
- Keep items concise and actionable.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS (use these exact headers):

## Top 3 Priorities
1. [Priority based on deadlines/urgency]
2. [Priority based on meetings/prep needed]
3. [Priority based on carry-forward importance]

## Intention for the Day
> [One sentence about what a successful day looks like]

## Meeting Prep Checklist
- [ ] [Meeting name] — [specific prep action]

## Yesterday's Follow-ups
- [ ] [Carry-forward item from today]

## To-Dos Must-do
- [ ] [Deadline-driven item]

## To-Dos Should-do
- [ ] [Important but flexible item]

## To-Dos Carry-forward
- [ ] [Can wait until later this week]

## Time-Blocked Schedule
| Time | Block |
| ---- | ----- |
| 07:00 - 08:00 AM | [block] |

## Team Context Out-today
- [person/reason]

## Team Context Out-upcoming
- [person/reason]

## Team Context Active-topics
- [topic]
"""


class NextDayNoteGenerator:
    """Generates the next workday's daily note from today's data using LLM."""

    def __init__(
        self,
        vault_path: str,
        daily_notes_folder: str = "06_Daily Notes",
        template_path: str | None = None,
        timezone: str = "America/Los_Angeles",
        llm_provider: str = "ollama",
        gemini_api_key: str = "",
        ollama_model: str = "llama3.1:8b",
        user_name: str = "",
    ):
        self.vault_path = Path(vault_path)
        self.daily_notes_dir = self.vault_path / daily_notes_folder
        self.timezone = ZoneInfo(timezone)
        self.llm_provider = llm_provider
        self.gemini_api_key = gemini_api_key
        self.ollama_model = ollama_model
        self.user_name = user_name

        # Use template if provided, otherwise use built-in
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = self.vault_path / "99_Obsidian_Templates" / "📅 Daily Note Template.md"

    def _get_monthly_subfolder(self, dt: datetime) -> Path:
        """Get the monthly subfolder path for a given date (e.g. 'August 2026')."""
        month_folder = dt.strftime("%B %Y")  # e.g. "August 2026"
        return self.daily_notes_dir / month_folder

    def _get_note_path_for_date(self, dt: datetime) -> Path:
        """Get the full note path for a given date, using monthly subfolders."""
        date_str = dt.strftime("%Y-%m-%d") if not hasattr(dt, 'strftime') else dt.strftime("%Y-%m-%d")
        return self._get_monthly_subfolder(dt) / f"{date_str}.md"

    def get_next_workday(self, from_date: datetime | None = None) -> datetime:
        """
        Get the next workday date.
        Monday-Thursday → next day
        Friday → Monday
        Saturday → Monday
        Sunday → Monday
        """
        if from_date is None:
            from_date = datetime.now(self.timezone)

        next_day = from_date + timedelta(days=1)

        # Skip weekends
        while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_day += timedelta(days=1)

        return next_day

    def generate(
        self,
        carry_forward_todos: list[str] | None = None,
        meeting_prep: list[str] | None = None,
        team_context: dict | None = None,
        tomorrow_meetings: list[dict] | None = None,
        edtech_brief: str | None = None,
    ) -> Path | None:
        """
        Generate the next workday's daily note, fully filled in by LLM.

        Returns:
            Path to the created note, or None on failure.
        """
        next_day = self.get_next_workday()
        date_str = next_day.strftime("%Y-%m-%d")
        day_of_week = next_day.strftime("%A")
        note_path = self._get_note_path_for_date(next_day)

        # Don't overwrite if it already exists
        if note_path.exists():
            logger.info(f"Daily note for {date_str} already exists — skipping generation")
            return note_path

        # Build the LLM prompt with all context
        carry_forward_section = self._format_list(carry_forward_todos, "No carry-forward items")
        meetings_section = self._format_meetings(tomorrow_meetings)
        team_context_section = self._format_team_context_for_prompt(team_context)

        prompt = DAILY_NOTE_PROMPT.format(
            target_date=date_str,
            day_of_week=day_of_week,
            user_name=self.user_name or "the user",
            carry_forward_section=carry_forward_section,
            meetings_section=meetings_section,
            team_context_section=team_context_section,
        )

        # Call LLM to fill in the note intelligently
        try:
            llm_output = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"LLM failed for next-day note: {e}")
            llm_output = None

        # Build the final note from template + LLM output
        content = self._build_note(date_str, llm_output, tomorrow_meetings, edtech_brief)

        # Write the note
        try:
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(content)
            logger.info(f"Generated next workday note: {date_str} ({day_of_week})")
            return note_path
        except Exception as e:
            logger.error(f"Failed to write next day note: {e}")
            return None

    def _build_note(
        self,
        date_str: str,
        llm_output: str | None,
        meetings: list[dict] | None,
        edtech_brief: str | None,
    ) -> str:
        """Build the final daily note markdown from template + LLM sections."""

        # Parse LLM output into sections
        sections = self._parse_llm_sections(llm_output) if llm_output else {}

        # Get priorities
        priorities = sections.get("priorities", ["", "", ""])
        while len(priorities) < 3:
            priorities.append("")

        # Get intention
        intention = sections.get("intention", "")

        # Build meetings table
        meetings_table = self._build_meetings_table(meetings)

        # Get schedule from LLM or build default
        schedule = sections.get("schedule", self._default_schedule(meetings))

        # Get meeting prep
        meeting_prep = sections.get("meeting_prep", "- [ ]")

        # Get follow-ups
        follow_ups = sections.get("follow_ups", "- [ ]")

        # Get to-dos
        must_do = sections.get("must_do", "- [ ]")
        should_do = sections.get("should_do", "- [ ]")
        carry_forward = sections.get("carry_forward", "- [ ]")

        # Get team context
        out_today = sections.get("out_today", "-")
        out_upcoming = sections.get("out_upcoming", "-")
        active_topics = sections.get("active_topics", "-")

        # Build the note
        content = f"""---
tags: daily-note
date: {date_str}
energy_level:
top_priority: {priorities[0] if priorities[0] else ''}
---

# Daily Note — {date_str}

---

## Top 3 Priorities

1. {priorities[0]}
2. {priorities[1]}
3. {priorities[2]}

---

## Intention for the Day

> {intention}

---

## Today's Meetings

{meetings_table}

---

## Time-Blocked Schedule

{schedule}

---

## Meeting Prep Checklist

{meeting_prep}

---

## Yesterday's Follow-ups

{follow_ups}

---

## To-Dos

**Must-do:**
{must_do}

**Should-do:**
{should_do}

**Carry-forward (this week):**
{carry_forward}

---

## Team Context

**Out today:**
{out_today}

**Out upcoming:**
{out_upcoming}

**Active team topics:**
{active_topics}

---

## Watch List

-

---

## End of Day Reflection

**What went well:**
>

**What was hard:**
>

**One thing to carry into tomorrow:**
>

---

## Notes & Scratch Pad

"""

        # Append EdTech Brief if available
        if edtech_brief:
            content += f"\n---\n\n## EdTech Brief\n\n{edtech_brief}\n"

        return content

    def _parse_llm_sections(self, raw: str) -> dict:
        """Parse the LLM output into structured sections."""
        sections = {}

        # Priorities
        priorities = []
        for match in re.finditer(r'^\d+\.\s+(.+)$', raw, re.MULTILINE):
            priorities.append(match.group(1).strip())
        if priorities:
            sections["priorities"] = priorities[:3]

        # Intention
        intention_match = re.search(r'>\s*(.+)', raw)
        if intention_match:
            sections["intention"] = intention_match.group(1).strip()

        # Meeting Prep
        meeting_prep_items = self._extract_section_items(raw, "## Meeting Prep")
        if meeting_prep_items:
            sections["meeting_prep"] = "\n".join(meeting_prep_items)

        # Follow-ups
        follow_up_items = self._extract_section_items(raw, "## Yesterday's Follow-ups")
        if follow_up_items:
            sections["follow_ups"] = "\n".join(follow_up_items)

        # To-Dos
        must_do_items = self._extract_section_items(raw, "## To-Dos Must-do")
        if must_do_items:
            sections["must_do"] = "\n".join(must_do_items)

        should_do_items = self._extract_section_items(raw, "## To-Dos Should-do")
        if should_do_items:
            sections["should_do"] = "\n".join(should_do_items)

        carry_items = self._extract_section_items(raw, "## To-Dos Carry-forward")
        if carry_items:
            sections["carry_forward"] = "\n".join(carry_items)

        # Schedule
        schedule_match = re.search(
            r'## Time-Blocked Schedule\n(\| Time \| Block \|\n\| ---- \| ----- \|\n(?:\|.+\|\n?)+)',
            raw,
        )
        if schedule_match:
            sections["schedule"] = schedule_match.group(1).strip()

        # Team Context
        out_today = self._extract_section_items(raw, "## Team Context Out-today")
        if out_today:
            sections["out_today"] = "\n".join(out_today)

        out_upcoming = self._extract_section_items(raw, "## Team Context Out-upcoming")
        if out_upcoming:
            sections["out_upcoming"] = "\n".join(out_upcoming)

        active_topics = self._extract_section_items(raw, "## Team Context Active-topics")
        if active_topics:
            sections["active_topics"] = "\n".join(active_topics)

        return sections

    def _extract_section_items(self, raw: str, header: str) -> list[str]:
        """Extract bullet/checkbox items from a section in the LLM output."""
        items = []
        in_section = False

        for line in raw.split("\n"):
            stripped = line.strip()

            if stripped.lower().startswith(header.lower().replace("## ", "")):
                in_section = True
                continue
            elif stripped.startswith("## ") and in_section:
                break

            if in_section and stripped.startswith(("- ", "* ")):
                items.append(stripped)

        # Also try matching by exact header
        if not items:
            pattern = re.escape(header)
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                remaining = raw[match.end():]
                for line in remaining.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("## "):
                        break
                    if stripped.startswith(("- ", "* ")):
                        items.append(stripped)

        return items

    def _build_meetings_table(self, meetings: list[dict] | None) -> str:
        """Build the meetings table markdown."""
        header = "| Time | Meeting | Notes |\n| ---- | ------- | ----- |"

        if not meetings:
            return f"{header}\n| (no meetings scheduled) | | |"

        rows = []
        for m in meetings:
            time_str = m.get("time", "")
            title = m.get("title", "Meeting")
            notes = m.get("notes", "")
            rows.append(f"| {time_str} | {title} | {notes} |")

        return f"{header}\n" + "\n".join(rows)

    def _default_schedule(self, meetings: list[dict] | None) -> str:
        """Build a default time-blocked schedule (9am-4pm work window)."""
        header = "| Time | Block |\n| ---- | ----- |"
        rows = [
            "| 09:00 - 10:00 AM | Deep Work / Meeting Blocks |",
            "| 10:00 - 12:00 PM | Deep Work / Meeting Blocks |",
            "| 12:00 - 01:00 PM | Lunch & Recharge |",
            "| 01:00 - 03:30 PM | Execution / Afternoon Meetings |",
            "| 03:30 - 04:00 PM | Wrap up / Planning for Tomorrow |",
        ]
        return f"{header}\n" + "\n".join(rows)

    def _format_list(self, items: list[str] | None, empty_msg: str) -> str:
        """Format a list for prompt injection."""
        if not items:
            return empty_msg
        return "\n".join(f"- {item}" for item in items)

    def _format_meetings(self, meetings: list[dict] | None) -> str:
        """Format meetings for prompt injection."""
        if not meetings:
            return "No meetings scheduled"
        lines = []
        for m in meetings:
            time_str = m.get("time", "TBD")
            title = m.get("title", "Meeting")
            attendees = m.get("notes", "")
            lines.append(f"- {time_str} PST — {title}" + (f" (with: {attendees})" if attendees else ""))
        return "\n".join(lines)

    def _format_team_context_for_prompt(self, team_context: dict | None) -> str:
        """Format team context for prompt injection."""
        if not team_context:
            return "No team context available"
        lines = []
        if team_context.get("out_today"):
            lines.append("Out today: " + ", ".join(team_context["out_today"]))
        if team_context.get("out_upcoming"):
            lines.append("Out upcoming: " + ", ".join(team_context["out_upcoming"]))
        if team_context.get("active_topics"):
            lines.append("Active topics: " + ", ".join(team_context["active_topics"]))
        return "\n".join(lines) if lines else "No team context available"

    def extract_carry_forward_from_today(self) -> dict:
        """
        Read today's daily note and extract incomplete items to carry forward.
        Returns a dict with:
            - carry_forward_todos: list of incomplete checkbox items
            - team_context: dict of team context sections
        """
        today = datetime.now(self.timezone).strftime("%Y-%m-%d")
        now = datetime.now(self.timezone)
        today_note = self._get_note_path_for_date(now)

        # Fallback: check flat structure for backwards compatibility
        if not today_note.exists():
            flat_path = self.daily_notes_dir / f"{today}.md"
            if flat_path.exists():
                today_note = flat_path
            else:
                logger.info(f"No daily note found for today ({today})")
                return {"carry_forward_todos": [], "team_context": {}}

        content = today_note.read_text()

        # Extract incomplete to-dos
        todos = self._extract_incomplete_todos(content)

        # Extract team context
        team_context = self._extract_team_context(content)

        return {
            "carry_forward_todos": todos,
            "team_context": team_context,
        }

    def _extract_incomplete_todos(self, content: str) -> list[str]:
        """Find all unchecked checkbox items in the note."""
        todos = []
        in_todos_section = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Track if we're in the To-Dos section or follow-ups
            if stripped.startswith("## To-Dos") or stripped.startswith("## Yesterday's Follow-ups"):
                in_todos_section = True
                continue
            elif stripped.startswith("## ") and in_todos_section:
                in_todos_section = False
                continue

            # Extract unchecked items
            if in_todos_section and stripped.startswith("- [ ]"):
                item = stripped[5:].strip()
                # Skip empty placeholders
                if item and item not in ("", "-"):
                    todos.append(item)

        return todos

    def _extract_team_context(self, content: str) -> dict:
        """Extract team context sections from today's note."""
        context = {
            "out_today": [],
            "out_upcoming": [],
            "active_topics": [],
        }

        in_team_context = False
        current_sub = None

        for line in content.split("\n"):
            stripped = line.strip()

            if stripped.startswith("## Team Context"):
                in_team_context = True
                continue
            elif stripped.startswith("## ") and in_team_context:
                break

            if not in_team_context:
                continue

            if "out today" in stripped.lower():
                current_sub = "out_today"
                continue
            elif "out upcoming" in stripped.lower():
                current_sub = "out_upcoming"
                continue
            elif "active team topics" in stripped.lower():
                current_sub = "active_topics"
                continue

            if current_sub and stripped.startswith("- ") and stripped != "- " and stripped != "-":
                context[current_sub].append(stripped[2:].strip())

        return context

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM."""
        if self.llm_provider == "gemini" and self.gemini_api_key:
            return self._call_gemini(prompt)
        else:
            return self._call_ollama(prompt)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama."""
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
