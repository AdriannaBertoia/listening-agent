"""
Calendar Client Module
Fetches and parses an ICS calendar feed to provide meeting context
(title, attendees, agenda/description) when a meeting is detected.
"""

import logging
import urllib.request
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from icalendar import Calendar

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """Represents a single calendar event with relevant meeting context."""
    title: str = "Unknown Meeting"
    start: datetime | None = None
    end: datetime | None = None
    attendees: list[str] = field(default_factory=list)
    organizer: str = ""
    description: str = ""
    location: str = ""

    def to_context_string(self) -> str:
        """Format as a string suitable for injecting into an LLM prompt."""
        parts = [f"Meeting Title: {self.title}"]

        if self.organizer:
            parts.append(f"Organizer: {self.organizer}")

        if self.attendees:
            parts.append(f"Attendees: {', '.join(self.attendees)}")

        if self.location:
            parts.append(f"Location/Link: {self.location}")

        if self.start:
            parts.append(f"Scheduled Time: {self.start.strftime('%I:%M %p')} - {self.end.strftime('%I:%M %p') if self.end else '?'}")

        if self.description:
            # Truncate long descriptions (agendas can be verbose)
            desc = self.description.strip()
            if len(desc) > 500:
                desc = desc[:500] + "..."
            parts.append(f"Agenda/Description:\n{desc}")

        return "\n".join(parts)


class CalendarClient:
    """Fetches and caches an ICS calendar feed, provides event lookup."""

    def __init__(
        self,
        ics_url: str,
        refresh_interval: int = 300,
        match_tolerance_minutes: int = 10,
        timezone: str = "America/Los_Angeles",
    ):
        self.ics_url = ics_url
        self.refresh_interval = refresh_interval
        self.match_tolerance_minutes = match_tolerance_minutes
        self.timezone = ZoneInfo(timezone)

        self._events: list[CalendarEvent] = []
        self._last_fetched: datetime | None = None
        self._raw_ics: bytes | None = None

    def refresh(self):
        """Fetch the ICS feed and parse events for today."""
        if not self.ics_url:
            logger.warning("No calendar ICS URL configured")
            return

        try:
            logger.debug("Fetching calendar feed...")
            req = urllib.request.Request(
                self.ics_url,
                headers={"User-Agent": "ListeningAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()

            self._raw_ics = raw
            today = datetime.now(self.timezone).date()
            self._events = self._parse_events_fast(raw, today)
            self._last_fetched = datetime.now(self.timezone)
            logger.info(f"Calendar refreshed — {len(self._events)} event(s) today")

        except Exception as e:
            logger.error(f"Failed to fetch calendar: {e}")

    def get_current_event(self) -> CalendarEvent | None:
        """
        Find the calendar event happening right now (or within tolerance).
        Returns the best match or None.
        """
        self._ensure_fresh()

        now = datetime.now(self.timezone)
        tolerance = timedelta(minutes=self.match_tolerance_minutes)

        best_match = None
        smallest_gap = timedelta.max

        for event in self._events:
            if not event.start or not event.end:
                continue

            # Event is currently happening
            if event.start <= now <= event.end:
                return event

            # Event started recently (within tolerance) — meeting might have run over
            if event.end < now and (now - event.end) <= tolerance:
                gap = now - event.end
                if gap < smallest_gap:
                    smallest_gap = gap
                    best_match = event

            # Event is about to start (within tolerance) — joined early
            if event.start > now and (event.start - now) <= tolerance:
                gap = event.start - now
                if gap < smallest_gap:
                    smallest_gap = gap
                    best_match = event

        return best_match

    def _ensure_fresh(self):
        """Refresh the calendar if stale or never fetched."""
        if self._last_fetched is None:
            self.refresh()
            return

        elapsed = (datetime.now(self.timezone) - self._last_fetched).total_seconds()
        if elapsed > self.refresh_interval:
            self.refresh()

    def _parse_todays_events(self, cal: Calendar) -> list[CalendarEvent]:
        """Extract today's events from the parsed calendar."""
        today = datetime.now(self.timezone).date()
        return self._parse_events_for_date(cal, today)

    def _parse_events_for_date(self, cal: Calendar, target_date) -> list[CalendarEvent]:
        """Extract events for a specific date from the parsed calendar."""
        events = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            try:
                event = self._parse_event(component)
                if event.start and event.start.date() == target_date:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Skipped unparseable event: {e}")
                continue

        # Sort by start time
        events.sort(key=lambda e: e.start or datetime.min.replace(tzinfo=self.timezone))
        return events

    def get_events_for_date(self, target_date) -> list[CalendarEvent]:
        """
        Public method to get events for any date.
        Uses cached raw ICS data with pre-filtering for speed.
        """
        self._ensure_fresh()

        if not self.ics_url:
            return []

        try:
            # Use cached raw ICS if we have it (avoid double-fetch)
            if self._raw_ics:
                raw = self._raw_ics
            else:
                req = urllib.request.Request(
                    self.ics_url,
                    headers={"User-Agent": "ListeningAgent/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                self._raw_ics = raw

            return self._parse_events_fast(raw, target_date)
        except Exception as e:
            logger.error(f"Failed to fetch calendar for date {target_date}: {e}")
            return []

    def _parse_events_fast(self, raw_ics: bytes, target_date) -> list[CalendarEvent]:
        """
        Fast event parsing using text pre-filter.
        Only parses VEVENT blocks that contain the target date string,
        avoiding full parse of 1000+ event calendar histories.
        """
        date_str = target_date.strftime("%Y%m%d")  # e.g. "20260804"
        text = raw_ics.decode("utf-8", errors="replace")

        # Extract VEVENT blocks containing our target date
        matching_blocks = []
        current_event = []
        in_event = False

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped == "BEGIN:VEVENT":
                in_event = True
                current_event = [line]
            elif stripped == "END:VEVENT":
                current_event.append(line)
                event_text = "\n".join(current_event)
                if date_str in event_text:
                    matching_blocks.append(event_text)
                in_event = False
                current_event = []
            elif in_event:
                current_event.append(line)

        # Parse only matching events
        events = []
        for block in matching_blocks:
            ics_wrapper = f"BEGIN:VCALENDAR\nVERSION:2.0\n{block}\nEND:VCALENDAR\n"
            try:
                cal = Calendar.from_ical(ics_wrapper)
                for component in cal.walk():
                    if component.name != "VEVENT":
                        continue
                    event = self._parse_event(component)
                    if event.start and event.start.date() == target_date:
                        events.append(event)
            except Exception as e:
                logger.debug(f"Skipped unparseable event: {e}")

        events.sort(key=lambda e: e.start or datetime.min.replace(tzinfo=self.timezone))
        return events

    def _parse_event(self, component) -> CalendarEvent:
        """Parse a VEVENT component into a CalendarEvent."""
        event = CalendarEvent()

        # Title
        summary = component.get("SUMMARY")
        if summary:
            event.title = str(summary)

        # Start/End times
        dtstart = component.get("DTSTART")
        if dtstart:
            dt = dtstart.dt
            if hasattr(dt, "hour"):  # It's a datetime, not just a date
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.timezone)
                else:
                    dt = dt.astimezone(self.timezone)
                event.start = dt
            else:
                # All-day event — skip for meeting matching
                event.start = datetime.combine(dt, datetime.min.time(), tzinfo=self.timezone)

        dtend = component.get("DTEND")
        if dtend:
            dt = dtend.dt
            if hasattr(dt, "hour"):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.timezone)
                else:
                    dt = dt.astimezone(self.timezone)
                event.end = dt

        # Attendees
        attendees = component.get("ATTENDEE")
        if attendees:
            if not isinstance(attendees, list):
                attendees = [attendees]
            for attendee in attendees:
                name = self._extract_attendee_name(attendee)
                if name:
                    event.attendees.append(name)

        # Organizer
        organizer = component.get("ORGANIZER")
        if organizer:
            event.organizer = self._extract_attendee_name(organizer)

        # Description (agenda)
        description = component.get("DESCRIPTION")
        if description:
            event.description = str(description)

        # Location (often a Teams/Zoom link)
        location = component.get("LOCATION")
        if location:
            event.location = str(location)

        return event

    def _extract_attendee_name(self, attendee) -> str:
        """Extract a readable name from a VCALENDAR attendee property."""
        # Try CN (common name) parameter first
        if hasattr(attendee, "params"):
            cn = attendee.params.get("CN")
            if cn:
                return str(cn)

        # Fall back to email from mailto:
        value = str(attendee)
        if "mailto:" in value.lower():
            email = value.split("mailto:")[-1].strip()
            # Try to make a name from the email
            local = email.split("@")[0]
            # Convert "first.last" or "first_last" to "First Last"
            name = local.replace(".", " ").replace("_", " ").title()
            return name

        return str(attendee)
