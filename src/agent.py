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
from datetime import datetime

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

        # Wire up callbacks
        self.watcher.on_meeting_start(self._handle_meeting_start)
        self.watcher.on_meeting_end(self._handle_meeting_end)

        # Track meeting timing
        self._meeting_start_time: datetime | None = None
        self._current_calendar_event = None

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

        # Main loop: poll for meetings + run scheduled tasks
        try:
            while self._running:
                # Check for active meetings
                self.watcher.poll()

                # Run any scheduled tasks (synthesis)
                schedule.run_pending()

                # Transcribe completed chunks in the background
                self._transcribe_pending_chunks()

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
        """Called when a meeting is detected."""
        app = self.watcher.get_active_meeting_app()
        logger.info(f"Meeting started in {app}")
        self._meeting_start_time = datetime.now()

        # Look up the calendar event for context
        self._current_calendar_event = self.calendar.get_current_event()
        if self._current_calendar_event:
            logger.info(f"Matched calendar event: {self._current_calendar_event.title}")
        else:
            logger.info("No matching calendar event found")

        self.recorder.start()

    def _handle_meeting_end(self):
        """Called when a meeting ends. Generates a per-meeting quadrant note."""
        app = self.watcher.get_active_meeting_app() or "Meeting"
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

        meeting_synthesis = self.synthesizer.synthesize_meeting(
            transcript_text,
            calendar_context=calendar_context,
            meeting_history=meeting_history,
        )

        # Store this meeting in memory for future recurrence
        self.meeting_memory.store_meeting(meeting_synthesis, calendar_title=meeting_title)

        # Write the meeting note to Obsidian
        note_path = self.writer.write_meeting_note(meeting_synthesis, app_name=app)

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

            # Also add personal next steps to daily note To-Dos
            if meeting_synthesis.get("my_next_steps"):
                next_steps_synthesis = {"action_items": meeting_synthesis["my_next_steps"]}
                self.writer.write_synthesis(next_steps_synthesis)
                logger.info("Personal next steps added to daily note")
        else:
            logger.warning("Failed to generate meeting note")

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
                self.transcriber.transcribe_file(chunk_path)

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
