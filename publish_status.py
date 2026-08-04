"""
Publish agent status to docs/status.json for GitHub Pages dashboard.
Run periodically (every 5 min) to keep the dashboard up to date.
Auto-commits and pushes to GitHub.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, date

import psutil

PROJECT_DIR = Path(__file__).parent
LOG_PATH = PROJECT_DIR / "listening-agent.log"
RECORDINGS_DIR = PROJECT_DIR / "data" / "recordings"
STATUS_PATH = PROJECT_DIR / "docs" / "status.json"


def get_agent_status() -> dict:
    """Check if the listening agent process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "listening-agent/run.py"],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if pids and pids[0]:
            return {"running": True, "pid": int(pids[0])}
    except Exception:
        pass
    return {"running": False, "pid": None}


def get_todays_activity() -> dict:
    """Parse today's log for meeting activity."""
    today_str = date.today().strftime("%Y-%m-%d")
    meetings = []
    current_meeting = None
    chunks_recorded = 0
    errors = []

    if not LOG_PATH.exists():
        return {"meetings": [], "errors": [], "chunks_recorded": 0}

    with open(LOG_PATH) as f:
        for line in f:
            if today_str not in line:
                continue

            if "Meeting confirmed" in line or "Meeting started in" in line:
                time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                app_match = re.search(r"Meeting (?:started in|confirmed) (.+)$", line)
                current_meeting = {
                    "start": time_match.group(1) if time_match else "",
                    "app": app_match.group(1).strip() if app_match else "Teams",
                    "end": None,
                    "duration": None,
                }
            elif "Meeting ended" in line and current_meeting:
                time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if time_match:
                    current_meeting["end"] = time_match.group(1)
                    try:
                        start = datetime.strptime(current_meeting["start"], "%Y-%m-%d %H:%M:%S")
                        end = datetime.strptime(current_meeting["end"], "%Y-%m-%d %H:%M:%S")
                        duration = (end - start).total_seconds()
                        current_meeting["duration"] = int(duration)
                    except (ValueError, TypeError):
                        pass
                if current_meeting.get("duration") and current_meeting["duration"] >= 120:
                    meetings.append(current_meeting)
                current_meeting = None
            elif "Recorded chunk" in line:
                chunks_recorded += 1
            elif "[ERROR]" in line:
                time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                error_text = line.split("[ERROR]")[-1].strip()[:100]
                errors.append({
                    "time": time_match.group(1).split(" ")[1] if time_match else "",
                    "message": error_text,
                })

    return {
        "meetings": meetings,
        "errors": errors[-5:],
        "chunks_recorded": chunks_recorded,
    }


def get_recordings_count() -> int:
    """Count recordings on disk."""
    if not RECORDINGS_DIR.exists():
        return 0
    return len(list(RECORDINGS_DIR.glob("chunk_*.wav")))


def is_currently_recording() -> dict:
    """Check if a meeting is currently being recorded by reading the log tail."""
    today_str = date.today().strftime("%Y-%m-%d")

    if not LOG_PATH.exists():
        return {"recording": False, "since": None, "app": None}

    # Read the last 100 lines to find the most recent state
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()

        # Walk backwards through today's lines to find last meeting start/end
        last_start = None
        last_end = None

        for line in reversed(lines):
            if today_str not in line:
                continue

            if not last_end and "Meeting ended" in line:
                last_end = line
                break  # Most recent event is an end — not recording
            elif not last_start and ("Meeting confirmed" in line or "Recording started" in line):
                last_start = line
                break  # Most recent event is a start — currently recording

        if last_start and not last_end:
            time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", last_start)
            since = time_match.group(1) if time_match else None

            # Calculate duration
            elapsed = None
            if since:
                try:
                    start_dt = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
                    elapsed = int((datetime.now() - start_dt).total_seconds())
                except ValueError:
                    pass

            return {"recording": True, "since": since, "elapsed_seconds": elapsed}

    except Exception:
        pass

    return {"recording": False, "since": None, "elapsed_seconds": None}


def publish():
    """Build and write status.json."""
    status = get_agent_status()
    activity = get_todays_activity()

    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S PST"),
        "agent": {
            "running": status["running"],
            "pid": status["pid"],
        },
        "today": date.today().strftime("%Y-%m-%d"),
        "day_of_week": date.today().strftime("%A"),
        "recording": is_currently_recording(),
        "activity": {
            "meetings": activity["meetings"],
            "chunks_recorded": activity["chunks_recorded"],
            "recordings_on_disk": get_recordings_count(),
            "errors": activity["errors"],
        },
    }

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Status written: {STATUS_PATH}")

    # Git commit and push
    try:
        subprocess.run(
            ["git", "add", "docs/status.json"],
            cwd=str(PROJECT_DIR), capture_output=True, timeout=10
        )
        subprocess.run(
            ["git", "commit", "-m", f"status update {datetime.now().strftime('%H:%M')}",
             "--allow-empty"],
            cwd=str(PROJECT_DIR), capture_output=True, timeout=10
        )
        result = subprocess.run(
            ["git", "push"],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("Pushed to GitHub")
        else:
            print(f"Push failed: {result.stderr}")
    except Exception as e:
        print(f"Git error: {e}")


if __name__ == "__main__":
    publish()
