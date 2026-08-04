"""
Process Watcher Module
Detects when Teams or Zoom has an active call running.
Triggers recording start/stop without requiring the app to be the active window.
"""

import subprocess
import logging
from typing import Callable

import psutil

logger = logging.getLogger(__name__)


class ProcessWatcher:
    """Watches for meeting apps with active audio sessions."""

    def __init__(self, watched_apps: list[str], poll_interval: int = 5, debounce_seconds: int = 30):
        self.watched_apps = watched_apps
        self.poll_interval = poll_interval
        self.debounce_seconds = debounce_seconds
        self.is_in_meeting = False
        self._on_meeting_start: Callable | None = None
        self._on_meeting_end: Callable | None = None
        # Debounce state
        self._active_since: float | None = None  # timestamp when meeting first detected
        self._inactive_since: float | None = None  # timestamp when meeting first disappeared
        self._debounce_triggered = False  # whether we've fired the start callback

    def on_meeting_start(self, callback: Callable):
        """Register callback for when a meeting is detected."""
        self._on_meeting_start = callback

    def on_meeting_end(self, callback: Callable):
        """Register callback for when a meeting ends."""
        self._on_meeting_end = callback

    def check_meeting_active(self) -> bool:
        """
        Check if any watched app is currently in an active call.
        
        For Teams: checks if the process is running and has active audio connections.
        For Zoom: checks if zoom.us CptHost (audio) process is active.
        """
        for app_name in self.watched_apps:
            if self._is_app_in_call(app_name):
                return True
        return False

    def _is_app_in_call(self, app_name: str) -> bool:
        """Detect if a specific app is in an active call."""
        try:
            if "Teams" in app_name:
                return self._check_teams_call()
            elif "zoom" in app_name.lower():
                return self._check_zoom_call()
        except Exception as e:
            logger.debug(f"Error checking {app_name}: {e}")
        return False

    def _check_teams_call(self) -> bool:
        """
        Detect active Teams call by checking for audio-related child processes
        and network activity. Teams spawns specific helper processes during calls.
        """
        for proc in psutil.process_iter(['name', 'connections']):
            try:
                name = proc.info['name'] or ""
                if "Teams" in name:
                    # Teams in a call has active UDP connections (WebRTC)
                    connections = proc.info.get('connections') or proc.connections()
                    udp_connections = [
                        c for c in connections
                        if c.type == 2  # SOCK_DGRAM (UDP)
                        and c.status == 'NONE'
                    ]
                    # Active call typically has multiple UDP streams
                    if len(udp_connections) >= 3:
                        return True
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return False

    def _check_zoom_call(self) -> bool:
        """
        Detect active Zoom call. Zoom runs CptHost for audio processing
        during active meetings.
        """
        zoom_audio_processes = []
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name'] or ""
                if name in ("CptHost", "zoom.us"):
                    zoom_audio_processes.append(name)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        # CptHost running alongside zoom.us = active call
        has_zoom = any("zoom" in p.lower() for p in zoom_audio_processes)
        has_audio = "CptHost" in zoom_audio_processes
        return has_zoom and has_audio

    def poll(self):
        """
        Single poll cycle with debounce.
        - Only fires meeting_start after the meeting has been active for debounce_seconds.
        - Only fires meeting_end after the meeting has been gone for debounce_seconds.
        This prevents flickering from brief Teams state changes.
        """
        import time

        currently_in_meeting = self.check_meeting_active()
        now = time.time()

        if currently_in_meeting:
            # Reset inactive timer
            self._inactive_since = None

            if not self._debounce_triggered:
                # Meeting detected but not yet confirmed
                if self._active_since is None:
                    self._active_since = now
                    logger.debug("Meeting activity detected, waiting for debounce confirmation...")

                elif now - self._active_since >= self.debounce_seconds:
                    # Meeting has been active long enough — it's real
                    logger.info("Meeting confirmed (active for 30s) — starting recording")
                    self.is_in_meeting = True
                    self._debounce_triggered = True
                    if self._on_meeting_start:
                        self._on_meeting_start()

        else:
            # No meeting detected
            self._active_since = None

            if self._debounce_triggered:
                # Meeting was active, now gone — debounce the end too (shorter: 10s)
                if self._inactive_since is None:
                    self._inactive_since = now
                    logger.debug("Meeting activity stopped, waiting to confirm end...")

                elif now - self._inactive_since >= 10:
                    # Meeting has been gone for 10 seconds — it's really over
                    logger.info("Meeting ended — stopping recording")
                    self.is_in_meeting = False
                    self._debounce_triggered = False
                    if self._on_meeting_end:
                        self._on_meeting_end()

    def get_active_meeting_app(self) -> str | None:
        """Return the name of the app currently in a call, or None."""
        for app_name in self.watched_apps:
            if self._is_app_in_call(app_name):
                return app_name
        return None
