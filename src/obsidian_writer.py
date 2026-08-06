"""
Obsidian Writer Module
Writes synthesized data into the correct sections of today's daily note.
Respects existing content — appends to sections rather than overwriting.
"""

import re
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ObsidianWriter:
    """Writes structured synthesis data into Obsidian daily notes."""

    def __init__(
        self,
        vault_path: str,
        daily_notes_folder: str = "06_Daily Notes",
        meetings_folder: str = "07_Meetings",
    ):
        self.vault_path = Path(vault_path)
        self.daily_notes_dir = self.vault_path / daily_notes_folder
        self.meetings_dir = self.vault_path / meetings_folder

    def get_todays_note_path(self) -> Path:
        """Get path to today's daily note using monthly subfolder structure."""
        now = datetime.now()
        month_folder = now.strftime("%B %Y")  # e.g. "August 2026"
        today = now.strftime("%Y-%m-%d")
        note_path = self.daily_notes_dir / month_folder / f"{today}.md"

        # Fallback: check flat structure for backwards compatibility
        if not note_path.exists():
            flat_path = self.daily_notes_dir / f"{today}.md"
            if flat_path.exists():
                return flat_path

        return note_path

    def write_meeting_note(self, meeting_synthesis: dict, app_name: str = "Meeting", transcript: str | None = None) -> Path | None:
        """
        Write a per-meeting note file using the Quadrant (4-Box) method.
        Saves to the meetings folder with a retain flag for auto-cleanup.
        Also links the meeting note from today's daily note.
        Includes full transcript as a collapsible section if provided.

        Returns the path to the created note, or None on failure.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H%M")
        title = meeting_synthesis.get("meeting_title", "Untitled Meeting")
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s\-]', '', title).strip().replace(" ", "-")[:50]
        filename = f"{timestamp}_{safe_title}.md"

        note_path = self.meetings_dir / filename
        note_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the quadrant note content
        attendees = meeting_synthesis.get("attendees", [])
        attendees_str = ", ".join(attendees) if attendees else "Unknown"

        key_topics = self._format_bullets(meeting_synthesis.get("key_topics", []))
        decisions = self._format_bullets(meeting_synthesis.get("decisions", []))
        action_items = self._format_action_items(meeting_synthesis.get("action_items", []))
        questions = self._format_bullets(meeting_synthesis.get("questions_followups", []))
        my_next_steps = self._format_action_items(meeting_synthesis.get("my_next_steps", []))

        content = f"""---
tags: meeting-note
date: {now.strftime("%Y-%m-%d")}
time: {now.strftime("%H:%M")}
app: {app_name}
retain: false
---

# {title}

**Date:** {now.strftime("%Y-%m-%d %I:%M %p")}
**App:** {app_name}
**Attendees:** {attendees_str}

---

## Quadrant 1: Key Topics & Discussion
{key_topics if key_topics else "- (none captured)"}

---

## Quadrant 2: Decisions Made
{decisions if decisions else "- (none captured)"}

---

## Quadrant 3: Action Items & Owners
{action_items if action_items else "- (none captured)"}

---

## Quadrant 4: Questions & Follow-ups
{questions if questions else "- (none captured)"}

---

## My Next Steps
{my_next_steps if my_next_steps else "- (none)"}
"""

        # Add collapsible transcript section if provided
        if transcript:
            content += f"""
---

<details>
<summary><strong>Full Transcript</strong></summary>

{transcript}

</details>
"""

        try:
            note_path.write_text(content)
            logger.info(f"Created meeting note: {filename}")

            # Link from today's daily note
            self._link_meeting_to_daily_note(title, filename)

            return note_path
        except Exception as e:
            logger.error(f"Failed to write meeting note: {e}")
            return None

    def cleanup_old_meeting_notes(self, retention_days: int = 30):
        """
        Delete meeting notes older than retention_days that are NOT marked retain: true.
        Called during end-of-day synthesis or on agent startup.
        """
        if not self.meetings_dir.exists():
            return

        cutoff = datetime.now().timestamp() - (retention_days * 86400)
        deleted_count = 0

        for note_file in self.meetings_dir.glob("*.md"):
            # Skip files newer than cutoff based on filesystem mtime
            if note_file.stat().st_mtime > cutoff:
                continue

            # Check frontmatter for retain flag
            try:
                content = note_file.read_text()
                if self._should_retain(content):
                    continue

                note_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted expired meeting note: {note_file.name}")
            except Exception as e:
                logger.warning(f"Could not process {note_file.name} for cleanup: {e}")

        if deleted_count:
            logger.info(f"Cleaned up {deleted_count} meeting note(s) older than {retention_days} days")

    def _should_retain(self, content: str) -> bool:
        """Check if a meeting note has retain: true in its frontmatter."""
        # Look for YAML frontmatter
        if not content.startswith("---"):
            return False

        end = content.find("---", 3)
        if end == -1:
            return False

        frontmatter = content[3:end]
        # Simple check — look for retain: true
        for line in frontmatter.split("\n"):
            stripped = line.strip()
            if stripped.lower() in ("retain: true", "retain: yes"):
                return True
        return False

    def _link_meeting_to_daily_note(self, title: str, filename: str):
        """Add a link to the meeting note in today's daily note meetings table."""
        note_path = self.get_todays_note_path()

        if not note_path.exists():
            self._create_daily_note(note_path)

        content = note_path.read_text()
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        link_name = filename.replace(".md", "")

        # Add a row to the Today's Meetings table
        meeting_row = f"| {time_str} | [[{link_name}|{title}]] | 4-box note |"

        # Find the meetings table and add a row
        table_marker = "## Today's Meetings"
        if table_marker in content:
            # Find the empty placeholder row and replace it, or append after header rows
            lines = content.split("\n")
            insert_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("## Today's Meetings"):
                    # Skip the header row and separator row
                    # Table starts after: | Time | Meeting | Notes |  and  | ---- | ...
                    insert_idx = i + 4  # After header + separator + empty row
                    break

            if insert_idx and insert_idx <= len(lines):
                # Check if the existing row is just the empty placeholder
                if insert_idx < len(lines) and lines[insert_idx - 1].strip() == "|      |         |       |":
                    lines[insert_idx - 1] = meeting_row
                else:
                    lines.insert(insert_idx, meeting_row)
                content = "\n".join(lines)
                note_path.write_text(content)
                logger.debug(f"Linked meeting note to daily note: {title}")

    def write_synthesis(self, synthesis: dict):
        """
        Write synthesized data into today's daily note.
        Appends to existing sections without destroying manually-entered content.
        """
        note_path = self.get_todays_note_path()

        if not note_path.exists():
            logger.warning(f"Today's daily note doesn't exist: {note_path}")
            logger.info("Creating from template...")
            self._create_daily_note(note_path)

        content = note_path.read_text()
        updated = False

        # Map synthesis data to daily note sections
        if synthesis.get("action_items"):
            content = self._append_to_section(
                content,
                "## To-Dos",
                self._format_action_items(synthesis["action_items"]),
                subsection="**Must-do:**",
            )
            updated = True

        if synthesis.get("follow_ups"):
            content = self._append_to_section(
                content,
                "## To-Dos",
                self._format_checkboxes(synthesis["follow_ups"]),
                subsection="**Should-do:**",
            )
            updated = True

        if synthesis.get("meeting_prep"):
            content = self._append_to_section(
                content,
                "## Meeting Prep Checklist",
                self._format_bullets(synthesis["meeting_prep"]),
            )
            updated = True

        if synthesis.get("team_context"):
            content = self._append_to_section(
                content,
                "## Team Context",
                self._format_team_context(synthesis["team_context"]),
                subsection="**Active team topics:**",
            )
            updated = True

        if synthesis.get("meetings_to_schedule"):
            content = self._append_to_section(
                content,
                "## To-Dos",
                self._format_meetings_to_schedule(synthesis["meetings_to_schedule"]),
                subsection="**Should-do:**",
            )
            updated = True

        if synthesis.get("decisions") or synthesis.get("key_notes"):
            notes = synthesis.get("decisions", []) + synthesis.get("key_notes", [])
            content = self._append_to_section(
                content,
                "## Notes & Scratch Pad",
                self._format_notes(notes),
            )
            updated = True

        if updated:
            note_path.write_text(content)
            logger.info(f"Updated daily note: {note_path.name}")
        else:
            logger.info("No new items to write to daily note")

    def _append_to_section(
        self, content: str, section_header: str, new_content: str, subsection: str | None = None
    ) -> str:
        """
        Append content to a specific section of the markdown file.
        If subsection is specified, appends after that subsection marker.
        """
        if subsection:
            # Find the subsection within the section
            pattern = re.escape(subsection)
            match = re.search(pattern, content)
            if match:
                # Find the end of existing items in this subsection
                insert_pos = match.end()
                # Look for the next line that's either empty, a new subsection, or a new section
                remaining = content[insert_pos:]
                lines = remaining.split("\n")

                # Find where to insert (after existing checkbox items)
                insert_offset = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith("- [") or stripped == "" and i == 0:
                        insert_offset = sum(len(l) + 1 for l in lines[:i + 1])
                    elif stripped.startswith("**") or stripped.startswith("##"):
                        break
                    elif stripped == "" and i > 0:
                        insert_offset = sum(len(l) + 1 for l in lines[:i])
                        break

                if insert_offset == 0:
                    insert_offset = len(lines[0]) + 1 if lines else 0

                actual_pos = insert_pos + insert_offset
                content = content[:actual_pos] + new_content + "\n" + content[actual_pos:]
                return content

        # Fallback: append at end of the section
        section_pattern = re.escape(section_header)
        match = re.search(section_pattern, content)
        if match:
            # Find the next section (## header) or end of file
            next_section = re.search(r"\n## ", content[match.end():])
            if next_section:
                insert_pos = match.end() + next_section.start()
            else:
                insert_pos = len(content)

            # Insert before the next section
            content = content[:insert_pos] + "\n" + new_content + "\n" + content[insert_pos:]

        return content

    def _format_action_items(self, items: list[str]) -> str:
        """Format action items as checkboxes."""
        lines = []
        for item in items:
            # Clean up checkbox markers if LLM included them
            clean = item.lstrip("[]x ").strip()
            if clean:
                lines.append(f"- [ ] {clean}")
        return "\n".join(lines)

    def _format_checkboxes(self, items: list[str]) -> str:
        """Format items as checkboxes."""
        lines = []
        for item in items:
            clean = item.lstrip("[]x ").strip()
            if clean:
                lines.append(f"- [ ] {clean}")
        return "\n".join(lines)

    def _format_bullets(self, items: list[str]) -> str:
        """Format items as bullet points."""
        return "\n".join(f"- {item}" for item in items if item)

    def _format_team_context(self, items: list[str]) -> str:
        """Format team context items."""
        return "\n".join(f"- {item}" for item in items if item)

    def _format_meetings_to_schedule(self, items: list[str]) -> str:
        """Format meetings to schedule as to-do items."""
        lines = []
        for item in items:
            lines.append(f"- [ ] Schedule: {item}")
        return "\n".join(lines)

    def _format_notes(self, items: list[str]) -> str:
        """Format notes and decisions."""
        timestamp = datetime.now().strftime("%I:%M %p")
        header = f"\n### Agent Synthesis ({timestamp})\n"
        bullets = "\n".join(f"- {item}" for item in items if item)
        return header + bullets

    def _create_daily_note(self, note_path: Path):
        """Create a daily note from template if it doesn't exist."""
        today = datetime.now().strftime("%Y-%m-%d")
        template = f"""---
tags: daily-note
date: {today}
energy_level:
top_priority:
---

# Daily Note — {today}

---

## Top 3 Priorities

1.
2.
3.

---

## Intention for the Day

>

---

## Today's Meetings

| Time | Meeting | Notes |
| ---- | ------- | ----- |
|      |         |       |

---

## Time-Blocked Schedule

| Time | Block |
| ---- | ----- |
| 09:00 - 10:00 AM | Deep Work / Meeting Blocks |
| 10:00 - 12:00 PM | Deep Work / Meeting Blocks |
| 12:00 - 01:00 PM | Lunch & Recharge |
| 01:00 - 03:30 PM | Execution / Afternoon Meetings |
| 03:30 - 04:00 PM | Wrap up / Planning for Tomorrow |

---

## Meeting Prep Checklist

-

---

## Yesterday's Follow-ups

- [ ]

---

## To-Dos

**Must-do:**
- [ ]

**Should-do:**
- [ ]

**Carry-forward (this week):**
- [ ]

---

## Team Context

**Out today:**
-

**Out upcoming:**
-

**Active team topics:**
-

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
        # Ensure monthly subfolder exists
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(template)
        logger.info(f"Created daily note: {note_path.name}")
