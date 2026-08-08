"""
Listening Agent — Main Orchestrator
Ties together process watching, audio recording, transcription, and synthesis.
Runs as a background daemon throughout the workday.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import schedule
from zoneinfo import ZoneInfo

from .process_watcher import ProcessWatcher
from .audio_recorder import AudioRecorder
from .transcriber import Transcriber
from .synthesizer import Synthesizer
from .obsidian_writer import ObsidianWriter
from .calendar_client import CalendarClient
from .meeting_memory import MeetingMemory
from .next_day_note import NextDayNoteGenerator
from .edtech_brief import EdTechBrief
from .weekly_review import WeeklyReviewGenerator

logger = logging.getLogger(__name__)


class ListeningAgent:
    """Main orchestrator that runs the full listening pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._running = False

        # Initialize components
        base_dir = Path(config_path).parent
        self.config["_base_dir"] = str(base_dir)
        data_dir = base_dir / "data"

        self.watcher = ProcessWatcher(
            watched_apps=self.config["detection"]["watched_apps"],
            poll_interval=self.config["detection"]["poll_interval"],
        )

        self.recorder = AudioRecorder(
            output_dir=str(data_dir / "recordings"),
            chunk_duration=self.config["recording"]["chunk_duration"],
            sample_rate=self.config["recording"]["sample_rate"],
            system_audio_device=self.config["recording"]["system_audio_device"],
            mic_device=self.config["recording"].get("mic_device"),
        )

        diarize_config = self.config["transcription"].get("diarization", {})
        self.transcriber = Transcriber(
            model_size=self.config["transcription"]["model"],
            language=self.config["transcription"]["language"],
            output_dir=str(data_dir / "transcripts"),
            diarization_enabled=diarize_config.get("enabled", False),
            hf_token=diarize_config.get("hf_token", ""),
            min_speakers=diarize_config.get("min_speakers"),
            max_speakers=diarize_config.get("max_speakers"),
            transcription_provider=self.config["transcription"].get("provider", "whisperx"),
            gemini_api_key=self.config["synthesis"].get("gemini_api_key", ""),
        )

        self.synthesizer = Synthesizer(
            provider=self.config["synthesis"]["llm_provider"],
            gemini_api_key=self.config["synthesis"]["gemini_api_key"],
            ollama_model=self.config["synthesis"].get("ollama_model", "llama3.1"),
            user_name=self.config.get("user", {}).get("name", ""),
            user_aliases=self.config.get("user", {}).get("aliases", []),
        )

        self.writer = ObsidianWriter(
            vault_path=self.config["obsidian"]["vault_path"],
            daily_notes_folder=self.config["obsidian"]["daily_notes_folder"],
            meetings_folder=self.config["obsidian"]["meetings_folder"],
        )

        # Calendar integration
        cal_config = self.config.get("calendar", {})
        self.calendar = CalendarClient(
            ics_url=cal_config.get("ics_url", ""),
            refresh_interval=cal_config.get("refresh_interval", 300),
            match_tolerance_minutes=cal_config.get("match_tolerance_minutes", 10),
            timezone=self.config["synthesis"].get("timezone", "America/Los_Angeles"),
        )

        # Meeting memory for recurring meetings
        memory_config = self.config.get("meeting_notes", {})
        memory_dir = base_dir / "data" / "meeting_memory"
        self.meeting_memory = MeetingMemory(
            memory_dir=str(memory_dir),
            max_history=memory_config.get("memory_lookback", 3),
            similarity_threshold=memory_config.get("similarity_threshold", 0.7),
        )

        # Next-day note generator
        self.next_day_generator = NextDayNoteGenerator(
            vault_path=self.config["obsidian"]["vault_path"],
            daily_notes_folder=self.config["obsidian"]["daily_notes_folder"],
            timezone=self.config["synthesis"].get("timezone", "America/Los_Angeles"),
            llm_provider=self.config["synthesis"]["llm_provider"],
            gemini_api_key=self.config["synthesis"]["gemini_api_key"],
            ollama_model=self.config["synthesis"].get("ollama_model", "llama3.1"),
            user_name=self.config.get("user", {}).get("name", ""),
        )

        # EdTech brief generator
        self.edtech_brief = EdTechBrief(
            llm_provider=self.config["synthesis"]["llm_provider"],
            gemini_api_key=self.config["synthesis"]["gemini_api_key"],
            ollama_model=self.config["synthesis"].get("ollama_model", "llama3.1"),
        )

        # Weekly review generator (Fridays at 3:30pm)
        self.weekly_review = WeeklyReviewGenerator(
            vault_path=self.config["obsidian"]["vault_path"],
            daily_notes_folder=self.config["obsidian"]["daily_notes_folder"],
            meetings_folder=self.config["obsidian"]["meetings_folder"],
            timezone=self.config["synthesis"].get("timezone", "America/Los_Angeles"),
            llm_provider=self.config["synthesis"]["llm_provider"],
            gemini_api_key=self.config["synthesis"]["gemini_api_key"],
            ollama_model=self.config["synthesis"].get("ollama_model", "llama3.1"),
            user_name=self.config.get("user", {}).get("name", ""),
        )

        # Wire up callbacks
        self.watcher.on_meeting_start(self._handle_meeting_start)
        self.watcher.on_meeting_end(self._handle_meeting_end)

        # Track meeting timing
        self._meeting_start_time: datetime | None = None
        self._current_calendar_event = None
        self._declined_at: datetime | None = None
        self._declined_event_title: str | None = None
        self._current_live_note_path: Path | None = None

    def start(self):
        """Start the listening agent."""
        logger.info("=" * 50)
        logger.info("Listening Agent starting up")
        logger.info(f"Watching for: {self.config['detection']['watched_apps']}")
        logger.info(f"Synthesis scheduled at: {self.config['synthesis']['schedule_time']}")
        logger.info("=" * 50)

        self._running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Schedule end-of-day synthesis
        synthesis_time = self.config["synthesis"]["schedule_time"]
        schedule.every().day.at(synthesis_time).do(self.run_synthesis)
        logger.info(f"Synthesis scheduled daily at {synthesis_time}")

        # Schedule weekly review (Fridays at 3:30pm)
        schedule.every().friday.at("15:30").do(self._run_weekly_review)
        logger.info("Weekly review scheduled for Fridays at 15:30")

        # Main loop: poll for meetings + run scheduled tasks
        try:
            while self._running:
                # Check for active meetings
                self.watcher.poll()

                # Run any scheduled tasks (synthesis)
                schedule.run_pending()

                # Transcribe completed chunks in the background
                self._transcribe_pending_chunks()

                # Check for back-to-back meeting transitions
                self._check_meeting_transition()

                time.sleep(self.config["detection"]["poll_interval"])

        except KeyboardInterrupt:
            self._shutdown()

    def stop(self):
        """Stop the agent gracefully."""
        self._running = False
        if self.recorder.is_recording:
            self.recorder.stop()
        logger.info("Listening Agent stopped")

    def run_synthesis(self):
        """Run the end-of-day synthesis pipeline."""
        logger.info("Running end-of-day synthesis...")

        # Cleanup old meeting notes (respects retain flag)
        retention_days = self.config.get("meeting_notes", {}).get("retention_days", 30)
        self.writer.cleanup_old_meeting_notes(retention_days=retention_days)

        # Stop recording if still going
        if self.recorder.is_recording:
            self.recorder.stop()
            time.sleep(2)  # Let the last chunk finish

        # Transcribe any remaining chunks
        self._transcribe_all_remaining()

        # Get all transcripts from today
        transcripts = self.transcriber.get_todays_transcripts()

        if not transcripts:
            logger.info("No transcripts for today — nothing to synthesize")
            return

        # Run LLM synthesis
        logger.info(f"Synthesizing {len(transcripts)} characters of transcripts...")
        synthesis = self.synthesizer.synthesize(transcripts)

        # Write to Obsidian
        self.writer.write_synthesis(synthesis)

        # Cleanup — delete recordings and transcripts (ephemeral)
        if self.config["synthesis"].get("cleanup_after_synthesis", True):
            self.recorder.clear_recordings()
            self.transcriber.clear_transcripts()
            logger.info("Ephemeral data cleaned up")

        logger.info("Synthesis complete — daily note updated")

        # Generate next workday's daily note
        if self.config.get("next_day_note", {}).get("enabled", True):
            self._generate_next_day_note()

    def run_synthesis_now(self):
        """Manual trigger for synthesis (useful for testing)."""
        logger.info("Manual synthesis triggered")
        self.run_synthesis()

    def _run_weekly_review(self):
        """Generate the weekly review note (Fridays only)."""
        now = datetime.now()
        if now.weekday() != 4:  # Only on Fridays
            return

        logger.info("Generating weekly review...")
        try:
            note_path = self.weekly_review.generate()
            if note_path:
                logger.info(f"Weekly review saved: {note_path.name}")
                self._send_notification(
                    title="Weekly Review Ready",
                    message="Your week-in-review note is ready in Obsidian",
                )
            else:
                logger.warning("Weekly review generation failed")
        except Exception as e:
            logger.error(f"Weekly review error: {e}")

    def _generate_next_day_note(self):
        """Generate the next workday's daily note with carry-forward data."""
        logger.info("Generating next workday's daily note...")

        # Skip weekends (shouldn't happen if synthesis only runs M-F, but just in case)
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday/Sunday
            logger.info("Weekend — skipping next day note generation")
            return

        # Extract carry-forward data from today's note
        today_data = self.next_day_generator.extract_carry_forward_from_today()

        # Get tomorrow's meetings from calendar
        next_day = self.next_day_generator.get_next_workday()
        tomorrow_meetings = self._get_meetings_for_date(next_day)

        # Generate EdTech brief
        edtech_brief_content = ""
        if self.config.get("next_day_note", {}).get("edtech_brief", True):
            try:
                edtech_brief_content = self.edtech_brief.generate_brief()
            except Exception as e:
                logger.warning(f"EdTech brief generation failed: {e}")

        # Build meeting prep hints from calendar descriptions
        meeting_prep = []
        for meeting in tomorrow_meetings:
            title = meeting.get("title", "")
            if title:
                meeting_prep.append(f"{title} — review agenda and prep materials")

        # Add recurring to-dos for the target day
        # Format supports "HH:MM|task description" for time-blocked items
        carry_forward = today_data.get("carry_forward_todos", [])
        recurring_config = self.config.get("next_day_note", {}).get("recurring_todos", {})
        day_name = next_day.strftime("%A").lower()  # "monday", "tuesday", etc.
        recurring_for_day = recurring_config.get(day_name, [])
        if recurring_for_day:
            for item in recurring_for_day:
                # Parse "HH:MM|task" format into a display-friendly string
                if "|" in item:
                    time_str, task = item.split("|", 1)
                    # Convert 24h to 12h for display
                    try:
                        from datetime import time as dt_time
                        h, m = map(int, time_str.strip().split(":"))
                        t = dt_time(h, m)
                        display_time = t.strftime("%I:%M %p").lstrip("0")
                        display_item = f"{task.strip()} @ {display_time} PST"
                    except (ValueError, TypeError):
                        display_item = task.strip()
                else:
                    display_item = item

                if display_item not in carry_forward:
                    carry_forward.append(f"[RECURRING] {display_item}")
            logger.info(f"Added {len(recurring_for_day)} recurring to-do(s) for {day_name.title()}")

        # Generate the note
        note_path = self.next_day_generator.generate(
            carry_forward_todos=carry_forward,
            meeting_prep=meeting_prep if meeting_prep else None,
            team_context=today_data.get("team_context"),
            tomorrow_meetings=tomorrow_meetings if tomorrow_meetings else None,
            edtech_brief=edtech_brief_content if edtech_brief_content else None,
        )

        if note_path:
            next_day_str = next_day.strftime("%Y-%m-%d")
            logger.info(f"Next workday note ready: {next_day_str}")
            self._send_notification(
                title="Tomorrow's Note Ready",
                message=f"Daily note for {next_day_str} created with carry-forward items",
            )
        else:
            logger.warning("Failed to generate next day note")

    def _get_meetings_for_date(self, target_date: datetime) -> list[dict]:
        """Get calendar meetings for a specific date."""
        events = self.calendar.get_events_for_date(target_date.date())

        meetings = []
        for event in events:
            if event.start:
                time_str = event.start.strftime("%I:%M %p")
                end_str = event.end.strftime("%I:%M %p") if event.end else ""
                meetings.append({
                    "time": f"{time_str} - {end_str}" if end_str else time_str,
                    "title": event.title,
                    "notes": ", ".join(event.attendees[:3]) if event.attendees else "",
                })

        return meetings

    def _handle_meeting_start(self):
        """Called when a meeting is detected. Shows a popup asking whether to record."""
        app = self.watcher.get_active_meeting_app()
        logger.info(f"Meeting detected in {app}")
        self._meeting_start_time = datetime.now()

        # Look up the calendar event for context
        self._current_calendar_event = self.calendar.get_current_event()
        if self._current_calendar_event:
            meeting_title = self._current_calendar_event.title
            logger.info(f"Matched calendar event: {meeting_title}")
        else:
            meeting_title = None
            logger.info("No matching calendar event found")

        # Show popup asking if user wants to record
        should_record = self._ask_to_record(meeting_title, app)

        if should_record:
            self.recorder.start()

            # Create the meeting note immediately and open in Obsidian
            self._current_live_note_path = self._create_live_meeting_note(
                meeting_title or "Meeting", app
            )

            self._send_notification(
                title="Recording Started",
                message=f"Recording: {meeting_title or 'Meeting'} — note open in Obsidian",
            )
        else:
            logger.info("User declined recording")
            # Reset state so meeting_end doesn't try to generate a note
            self._meeting_start_time = None
            self._current_calendar_event = None
            # Track when this decline happened so we can re-prompt later
            # if a new calendar event starts while Teams stays active
            self._declined_at = datetime.now()
            self._declined_event_title = meeting_title
            # Reset the watcher so it can detect a new meeting later
            # (e.g. if user stays in Teams but starts a different call)
            self.watcher.is_in_meeting = False
            self.watcher._debounce_triggered = False
            self.watcher._active_since = None

    def _ask_to_record(self, meeting_title: str | None, app: str | None) -> bool:
        """
        Show a macOS dialog asking the user if they want to record this meeting.
        Returns True if user clicks Record, False otherwise.
        Times out after 15 seconds (defaults to Skip).
        
        Uses a helper script launched via 'open' to ensure the dialog appears
        in the user's GUI session (not the launchd background context).
        """
        if meeting_title:
            message = f'"{meeting_title}"'
        else:
            message = f"Meeting detected in {app or 'Teams'}"

        try:
            script_path = Path(self.config.get("_base_dir", ".")) / "record_prompt.sh"
            result = subprocess.run(
                ["launchctl", "asuser", str(os.getuid()), str(script_path), message],
                capture_output=True,
                text=True,
                timeout=20,
            )

            output = result.stdout.strip()
            logger.debug(f"Record prompt result: {output}")

            if "RECORD" in output:
                logger.info("User chose to record")
                return True
            elif "TIMEOUT" in output:
                logger.info("Dialog timed out — skipping recording")
                return False
            else:
                logger.info("User chose to skip recording")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("Record prompt timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to show recording dialog: {e}")
            # If dialog fails, record anyway (don't lose data)
            return True

    def _create_live_meeting_note(self, meeting_title: str, app: str | None) -> Path | None:
        """
        Create a meeting note immediately when recording starts.
        Opens it in Obsidian so the user can take live notes during the meeting.
        Returns the path to the note file.
        """
        import re

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H%M")
        safe_title = re.sub(r'[^\w\s\-]', '', meeting_title).strip().replace(" ", "-")[:50]
        filename = f"{timestamp}_{safe_title}.md"

        meetings_dir = self.writer.meetings_dir
        note_path = meetings_dir / filename
        note_path.parent.mkdir(parents=True, exist_ok=True)

        # Get attendees from calendar
        attendees = ""
        if self._current_calendar_event and self._current_calendar_event.attendees:
            attendees = ", ".join(self._current_calendar_event.attendees)

        # Create the live note template
        content = f"""---
tags: meeting-note
date: {now.strftime("%Y-%m-%d")}
time: {now.strftime("%H:%M")}
app: {app or "Meeting"}
retain: false
status: recording
---

# {meeting_title}

**Date:** {now.strftime("%Y-%m-%d %I:%M %p")}
**App:** {app or "Meeting"}
**Attendees:** {attendees or "TBD"}

> Recording in progress...

---

## My Notes

_Jot down action items, decisions, and anything important here during the meeting:_

- 
- 
- 

---

## Quadrant 1: Key Topics & Discussion
_(will be filled after meeting ends)_

---

## Quadrant 2: Decisions Made
_(will be filled after meeting ends)_

---

## Quadrant 3: Action Items & Owners
_(will be filled after meeting ends)_

---

## Quadrant 4: Questions & Follow-ups
_(will be filled after meeting ends)_

---

## My Next Steps
_(will be filled after meeting ends)_
"""

        try:
            note_path.write_text(content)
            logger.info(f"Live meeting note created: {filename}")

            # Open in Obsidian
            vault_name = Path(self.config["obsidian"]["vault_path"]).name
            meetings_folder = self.config["obsidian"]["meetings_folder"]
            file_path = f"{meetings_folder}/{filename}".replace(".md", "")
            obsidian_uri = f"obsidian://open?vault={vault_name}&file={file_path}"
            subprocess.run(["open", obsidian_uri], capture_output=True, timeout=5)
            logger.info(f"Opened note in Obsidian: {meeting_title}")

            return note_path
        except Exception as e:
            logger.error(f"Failed to create live meeting note: {e}")
            return None

    def _handle_meeting_end(self):
        """Called when a meeting ends. Generates a per-meeting quadrant note."""
        app = self.watcher.get_active_meeting_app() or "Meeting"

        # If user declined recording, just reset state
        if not self.recorder.is_recording:
            logger.info(f"Meeting ended ({app}) — was not being recorded")
            return

        logger.info(f"Meeting ended ({app}) — generating meeting note...")
        self.recorder.stop()

        # Check minimum duration
        min_duration = self.config.get("meeting_notes", {}).get("minimum_duration", 120)
        if self._meeting_start_time:
            elapsed = (datetime.now() - self._meeting_start_time).total_seconds()
            self._meeting_start_time = None
            if elapsed < min_duration:
                logger.info(
                    f"Meeting was only {int(elapsed)}s (minimum: {min_duration}s) — skipping note generation"
                )
                return
        else:
            # No start time tracked — proceed anyway
            elapsed = None

        # Give the last chunk a moment to finalize
        time.sleep(2)

        # Transcribe all chunks from this meeting
        self._transcribe_all_remaining()

        # Gather transcripts for this meeting
        transcript_text = self.transcriber.get_todays_transcripts()

        if not transcript_text or not transcript_text.strip():
            logger.info("No transcript captured for this meeting — skipping note generation")
            return

        # Map speaker labels to real names using calendar attendees + LLM
        attendees = []
        if self._current_calendar_event:
            attendees = self._current_calendar_event.attendees or []
        transcript_text = self._map_speaker_names(transcript_text, attendees)

        # Synthesize into quadrant format
        calendar_context = None
        if self._current_calendar_event:
            calendar_context = self._current_calendar_event.to_context_string()

        # Look up recurring meeting history
        meeting_title = (
            self._current_calendar_event.title if self._current_calendar_event else "Meeting"
        )
        meeting_history = self.meeting_memory.get_context_for_meeting(meeting_title)
        if meeting_history:
            logger.info(f"Found recurring meeting history for: {meeting_title}")

        self._current_calendar_event = None

        # Read user's live notes from the existing meeting note (if they wrote any)
        user_live_notes = ""
        if self._current_live_note_path and self._current_live_note_path.exists():
            user_live_notes = self._extract_live_notes(self._current_live_note_path)
            if user_live_notes:
                logger.info(f"Found user's live notes ({len(user_live_notes)} chars)")

        meeting_synthesis = self.synthesizer.synthesize_meeting(
            transcript_text,
            calendar_context=calendar_context,
            meeting_history=meeting_history,
            meeting_title=meeting_title,
            user_notes=user_live_notes,
        )

        # Store this meeting in memory for future recurrence
        self.meeting_memory.store_meeting(meeting_synthesis, calendar_title=meeting_title)

        # Update the existing live note with quadrant sections + transcript
        if self._current_live_note_path and self._current_live_note_path.exists():
            note_path = self._finalize_live_note(
                self._current_live_note_path, meeting_synthesis, transcript_text
            )
        else:
            # Fallback: create a new note (shouldn't happen normally)
            note_path = self.writer.write_meeting_note(
                meeting_synthesis, app_name=app, transcript=transcript_text
            )

        self._current_live_note_path = None

        if note_path:
            title = meeting_synthesis.get("meeting_title", "Meeting")
            logger.info(f"Meeting note saved: {note_path.name}")

            # Send macOS notification with link to open the note
            if self.config.get("meeting_notes", {}).get("notify", True):
                self._send_notification(
                    title="Meeting Note Ready",
                    message=f"{title} — quadrant note saved to Obsidian",
                )
                # Send a follow-up review prompt after a short delay
                self._send_review_prompt(note_path, title)

            # Quick edit popup — capture a thought while it's fresh
            quick_note = self._ask_quick_edit(title)
            if quick_note:
                self._append_quick_note(note_path, quick_note)

            # Also add personal next steps to daily note To-Dos
            if meeting_synthesis.get("my_next_steps"):
                next_steps_synthesis = {"action_items": meeting_synthesis["my_next_steps"]}
                self.writer.write_synthesis(next_steps_synthesis)
                logger.info("Personal next steps added to daily note")
        else:
            logger.warning("Failed to generate meeting note")

    def _check_meeting_transition(self):
        """
        During active recording, check if the calendar indicates a new meeting
        should be starting (back-to-back meetings without leaving Teams).
        
        If the current calendar event has ended and a new one is starting,
        prompt the user to split the recording into a new meeting note.
        """
        # Only relevant if we're currently recording
        if not self.recorder.is_recording:
            return

        # Only check if we had a calendar match
        if not self._current_calendar_event:
            return

        now = datetime.now(ZoneInfo(self.config["synthesis"].get("timezone", "America/Los_Angeles")))

        # Has the current calendar event ended?
        if not self._current_calendar_event.end:
            return

        event_end = self._current_calendar_event.end
        # Add a 2-minute grace period (meetings often run over slightly)
        grace = event_end + timedelta(minutes=2)

        if now <= grace:
            return  # Current meeting hasn't ended yet

        # Check if there's a new calendar event starting now
        next_event = self.calendar.get_current_event()
        if not next_event:
            return

        # Don't re-trigger for the same event
        if next_event.title == self._current_calendar_event.title:
            return

        # New meeting detected! Ask user if they want to split
        logger.info(f"Back-to-back detected: '{self._current_calendar_event.title}' ended, '{next_event.title}' starting")

        should_split = self._ask_meeting_transition(
            ended_title=self._current_calendar_event.title,
            next_title=next_event.title,
        )

        if should_split:
            # End the current recording and generate a note
            logger.info("User chose to split — ending current recording")
            self._handle_meeting_end()

            # Start a new recording for the next meeting
            self._meeting_start_time = datetime.now()
            self._current_calendar_event = next_event
            self.recorder.start()
            self._send_notification(
                title="New Recording Started",
                message=f"Recording: {next_event.title}",
            )
            logger.info(f"Started new recording for: {next_event.title}")
        else:
            # User wants to keep it as one recording — update the event reference
            # so we don't ask again
            logger.info("User chose to continue as one meeting")
            self._current_calendar_event = next_event

    def _ask_meeting_transition(self, ended_title: str, next_title: str) -> bool:
        """
        Show a popup asking if the user wants to split into a new meeting note.
        """
        message = (
            f'"{ended_title}" has ended.\\n\\n'
            f'"{next_title}" is starting.\\n\\n'
            f'Save note and start new recording?'
        )

        try:
            script = (
                f'display dialog "{message}" '
                f'buttons {{"Keep Together", "Split"}} default button "Split" '
                f'with title "Listening Agent — New Meeting" '
                f'with icon caution '
                f'giving up after 15'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=20,
            )

            output = result.stdout.strip()
            if "Split" in output:
                return True
            return False

        except Exception as e:
            logger.warning(f"Transition dialog failed: {e}")
            return False

    def _map_speaker_names(self, transcript: str, attendees: list[str]) -> str:
        """
        Replace generic SPEAKER_XX labels with real names using LLM inference.
        Uses calendar attendees as hints + context clues from the transcript.
        Always maps the user (Adrianna) as one of the speakers.
        """
        import re

        # Find all unique speaker labels in the transcript
        speaker_labels = sorted(set(re.findall(r'\[?(SPEAKER_\d+)\]?', transcript)))

        if not speaker_labels:
            return transcript

        # Build the list of possible participants
        user_name = self.config.get("user", {}).get("name", "Adrianna Bertoia")
        participants = [user_name] + [a for a in attendees if a.lower() != user_name.lower()]

        # If only 2 speakers and 1 attendee in a 1:1, we can map directly
        if len(speaker_labels) == 2 and len(participants) == 2:
            # Use LLM to figure out which speaker is the user based on context
            pass  # Let the LLM handle it below

        # If no attendees, just label as "Adrianna" + "Other"
        if not attendees:
            participants = [user_name, "Other"]

        # Use LLM to map speakers to names
        # Take first 2000 chars of transcript as sample for context
        sample = transcript[:2000]

        prompt = (
            f"Given this meeting transcript with speaker labels, identify who each speaker is.\n\n"
            f"Known participants: {', '.join(participants)}\n"
            f"The user (me) is: {user_name}\n\n"
            f"Transcript sample:\n{sample}\n\n"
            f"Speaker labels found: {', '.join(speaker_labels)}\n\n"
            f"Return ONLY a JSON mapping like: "
            f'{{"SPEAKER_01": "Adrianna", "SPEAKER_02": "Wendy"}}\n'
            f"Use first names only. If unsure about a speaker, use the label as-is."
        )

        try:
            mapping = self._call_llm_for_speaker_map(prompt)
            if mapping:
                for label, name in mapping.items():
                    transcript = transcript.replace(f"[{label}]", f"[{name}]")
                    transcript = transcript.replace(f"**{label}**", f"**{name}**")
                logger.info(f"Speaker names mapped: {mapping}")
        except Exception as e:
            logger.warning(f"Speaker name mapping failed: {e}")

        return transcript

    def _call_llm_for_speaker_map(self, prompt: str) -> dict | None:
        """Call LLM to get speaker name mapping. Returns dict or None."""
        import json

        if self.config["synthesis"]["llm_provider"] == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config["synthesis"]["gemini_api_key"])
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                text = response.text.strip()
                # Extract JSON from response
                if "{" in text:
                    json_str = text[text.index("{"):text.rindex("}") + 1]
                    return json.loads(json_str)
            except Exception as e:
                logger.debug(f"Gemini speaker mapping failed: {e}")
        else:
            try:
                import urllib.request
                payload = json.dumps({
                    "model": self.config["synthesis"].get("ollama_model", "llama3.1:8b"),
                    "prompt": prompt,
                    "stream": False,
                }).encode("utf-8")
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    text = result.get("response", "")
                    if "{" in text:
                        json_str = text[text.index("{"):text.rindex("}") + 1]
                        return json.loads(json_str)
            except Exception as e:
                logger.debug(f"Ollama speaker mapping failed: {e}")

        return None

    def _extract_live_notes(self, note_path: Path) -> str:
        """Extract the user's handwritten notes from the live meeting note."""
        try:
            content = note_path.read_text()
            # Find the "## My Notes" section and extract its content
            import re
            match = re.search(
                r'## My Notes\n(.*?)(?=\n---|\n## Quadrant)',
                content,
                re.DOTALL,
            )
            if match:
                notes = match.group(1).strip()
                # Remove the placeholder text
                notes = notes.replace(
                    "_Jot down action items, decisions, and anything important here during the meeting:_",
                    "",
                ).strip()
                # Remove empty bullet placeholders
                lines = [l for l in notes.split("\n") if l.strip() not in ("- ", "-", "")]
                return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to extract live notes: {e}")
        return ""

    def _finalize_live_note(self, note_path: Path, synthesis: dict, transcript: str) -> Path:
        """
        Update the existing live meeting note with quadrant sections from synthesis.
        Preserves the user's live notes and replaces placeholder sections.
        """
        try:
            content = note_path.read_text()

            # Remove the "Recording in progress..." indicator
            content = content.replace("> Recording in progress...\n", "")
            content = content.replace("status: recording", "status: complete")

            # Build quadrant content
            key_topics = "\n".join(f"- {t}" for t in synthesis.get("key_topics", [])) or "- (none captured)"
            decisions = "\n".join(f"- {d}" for d in synthesis.get("decisions", [])) or "- (none captured)"
            action_items = "\n".join(f"- [ ] {a}" for a in synthesis.get("action_items", [])) or "- (none captured)"
            questions = "\n".join(f"- {q}" for q in synthesis.get("questions_followups", [])) or "- (none captured)"
            next_steps = "\n".join(f"- [ ] {s}" for s in synthesis.get("my_next_steps", [])) or "- (none)"

            # Replace placeholder sections
            import re

            content = re.sub(
                r'## Quadrant 1: Key Topics & Discussion\n_\(will be filled after meeting ends\)_',
                f'## Quadrant 1: Key Topics & Discussion\n{key_topics}',
                content,
            )
            content = re.sub(
                r'## Quadrant 2: Decisions Made\n_\(will be filled after meeting ends\)_',
                f'## Quadrant 2: Decisions Made\n{decisions}',
                content,
            )
            content = re.sub(
                r'## Quadrant 3: Action Items & Owners\n_\(will be filled after meeting ends\)_',
                f'## Quadrant 3: Action Items & Owners\n{action_items}',
                content,
            )
            content = re.sub(
                r'## Quadrant 4: Questions & Follow-ups\n_\(will be filled after meeting ends\)_',
                f'## Quadrant 4: Questions & Follow-ups\n{questions}',
                content,
            )
            content = re.sub(
                r'## My Next Steps\n_\(will be filled after meeting ends\)_',
                f'## My Next Steps\n{next_steps}',
                content,
            )

            # Append collapsible transcript
            if transcript:
                content += f"""
---

<details>
<summary><strong>Full Transcript</strong></summary>

{transcript}

</details>
"""

            note_path.write_text(content)
            logger.info(f"Finalized live meeting note: {note_path.name}")
            return note_path

        except Exception as e:
            logger.error(f"Failed to finalize live note: {e}")
            return note_path

    def _ask_quick_edit(self, meeting_title: str) -> str | None:
        """
        Show a text input dialog after a meeting note is generated.
        Lets the user jot down one quick thought while it's fresh.
        Returns the text or None if skipped/timed out.
        """
        try:
            script = (
                'tell application "System Events"\n'
                '  activate\n'
                'end tell\n'
                f'display dialog "Quick thought about \\"{meeting_title}\\"?\\n\\n'
                f'(Leave blank to skip)" '
                f'default answer "" '
                f'buttons {{"Skip", "Add"}} default button "Add" '
                f'with title "Meeting Note — Quick Edit" '
                f'giving up after 30'
            )
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=35,
            )

            output = result.stdout.strip()
            if "gave up:true" in output or "Skip" in output:
                return None

            # Extract the text entered
            if "text returned:" in output:
                text = output.split("text returned:")[1].strip().rstrip(",")
                if text:
                    logger.info(f"Quick note added: {text[:50]}...")
                    return text

        except Exception as e:
            logger.debug(f"Quick edit dialog failed: {e}")

        return None

    def _append_quick_note(self, note_path: Path, note_text: str):
        """Append a quick note to the meeting note file."""
        try:
            content = note_path.read_text()
            timestamp = datetime.now().strftime("%I:%M %p")
            addition = f"\n\n---\n\n## Quick Note ({timestamp})\n\n{note_text}\n"

            # Insert before the transcript section if it exists
            if "<details>" in content:
                content = content.replace("<details>", f"{addition}\n<details>")
            else:
                content += addition

            note_path.write_text(content)
            logger.info("Quick note appended to meeting note")
        except Exception as e:
            logger.warning(f"Failed to append quick note: {e}")

    def _transcribe_pending_chunks(self):
        """Transcribe any completed audio chunks that haven't been processed."""
        chunks = self.recorder.chunk_files
        if not chunks:
            return

        # Only transcribe chunks that are at least a few seconds old
        # (to avoid grabbing a chunk still being written)
        for chunk_path in chunks[:-1]:  # Skip the most recent (might be in-progress)
            if not Path(chunk_path).exists():
                continue
            transcript_name = f"transcript_{Path(chunk_path).stem.replace('chunk_', '')}.txt"
            transcript_path = Path(self.transcriber.output_dir) / transcript_name

            if not transcript_path.exists():
                result = self.transcriber.transcribe_file(chunk_path)
                # If transcription fails, don't retry every poll cycle
                if result is None:
                    break

    def _send_notification(self, title: str, message: str):
        """Send a macOS notification via osascript."""
        try:
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            logger.debug(f"Notification sent: {title}")
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def _send_review_prompt(self, note_path: Path, title: str):
        """
        Send a follow-up notification prompting the user to review the note.
        Opens the note in Obsidian when clicked.
        """
        try:
            # Build an Obsidian URI to open the note directly
            vault_name = Path(self.config["obsidian"]["vault_path"]).name
            # Obsidian URI format: obsidian://open?vault=VaultName&file=path/to/file
            meetings_folder = self.config["obsidian"]["meetings_folder"]
            file_path = f"{meetings_folder}/{note_path.name}".replace(".md", "")
            obsidian_uri = f"obsidian://open?vault={vault_name}&file={file_path}"

            # Send notification that opens Obsidian on click
            script = (
                f'display notification "Anything the AI missed? Click to review & edit." '
                f'with title "Review: {title}" '
                f'subtitle "30 seconds to review your meeting note"'
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )

            # Also open the note in Obsidian automatically after a brief pause
            time.sleep(2)
            subprocess.run(
                ["open", obsidian_uri],
                capture_output=True,
                timeout=5,
            )
            logger.debug(f"Review prompt sent, opened note in Obsidian: {title}")
        except Exception as e:
            logger.warning(f"Failed to send review prompt: {e}")

    def _transcribe_all_remaining(self):
        """Transcribe all remaining chunks found on disk."""
        recordings_dir = Path(self.config.get("_base_dir", ".")) / "data" / "recordings"
        chunk_files = sorted(recordings_dir.glob("chunk_*.wav"))

        if not chunk_files:
            # Also check the recorder's in-memory list (filter to files that exist)
            chunk_files = [Path(f) for f in self.recorder.chunk_files if Path(f).exists()]

        for chunk_path in chunk_files:
            if not chunk_path.exists():
                continue
            transcript_name = f"transcript_{chunk_path.stem.replace('chunk_', '')}.txt"
            transcript_path = Path(self.transcriber.output_dir) / transcript_name
            if not transcript_path.exists():
                self.transcriber.transcribe_file(str(chunk_path))

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.stop()
        sys.exit(0)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Config not found: {config_path}")
            sys.exit(1)

        with open(config_file) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "listening-agent.log")

        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
