"""
Listening Agent Dashboard
A local web dashboard to monitor the listening agent's status,
recording activity, and meeting note output.

Run: python dashboard/app.py
Opens at: http://localhost:5050
"""

import os
import re
import subprocess
import json
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Paths
PROJECT_DIR = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
LOG_PATH = PROJECT_DIR / "listening-agent.log"
RECORDINGS_DIR = PROJECT_DIR / "data" / "recordings"
TRANSCRIPTS_DIR = PROJECT_DIR / "data" / "transcripts"
MEETING_MEMORY_DIR = PROJECT_DIR / "data" / "meeting_memory"

# Load config for vault path
import yaml
try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

VAULT_PATH = Path(CONFIG.get("obsidian", {}).get("vault_path", ""))
MEETINGS_DIR = VAULT_PATH / CONFIG.get("obsidian", {}).get("meetings_folder", "07_Meetings")
DAILY_NOTES_DIR = VAULT_PATH / CONFIG.get("obsidian", {}).get("daily_notes_folder", "06_Daily Notes")


def get_agent_status() -> dict:
    """Check if the listening agent process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "listening-agent/run.py"],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if pids and pids[0]:
            pid = int(pids[0])
            # Get process start time
            ps_result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart=,pcpu=,pmem="],
                capture_output=True, text=True, timeout=5
            )
            info = ps_result.stdout.strip()
            return {
                "running": True,
                "pid": pid,
                "info": info,
            }
    except Exception:
        pass
    return {"running": False, "pid": None, "info": ""}


def get_todays_activity() -> dict:
    """Parse today's log entries for meeting activity."""
    today_str = date.today().strftime("%Y-%m-%d")
    meetings = []
    current_meeting = None
    errors = []

    if not LOG_PATH.exists():
        return {"meetings": [], "errors": [], "chunks_recorded": 0}

    chunks_recorded = 0

    try:
        with open(LOG_PATH) as f:
            for line in f:
                if today_str not in line:
                    continue

                if "Meeting started in" in line:
                    time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    app_match = re.search(r"Meeting started in (.+)$", line)
                    current_meeting = {
                        "start": time_match.group(1) if time_match else "",
                        "app": app_match.group(1).strip() if app_match else "Unknown",
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
                    meetings.append(current_meeting)
                    current_meeting = None
                elif "Recorded chunk" in line:
                    chunks_recorded += 1
                elif "[ERROR]" in line:
                    time_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    error_text = line.split("[ERROR]")[-1].strip()[:120]
                    errors.append({
                        "time": time_match.group(1).split(" ")[1] if time_match else "",
                        "message": error_text,
                    })
    except Exception:
        pass

    return {
        "meetings": meetings,
        "errors": errors[:10],  # Last 10 errors
        "chunks_recorded": chunks_recorded,
    }


def get_recent_meeting_notes(limit: int = 10) -> list[dict]:
    """Get recent meeting notes from the Obsidian meetings folder."""
    notes = []
    if not MEETINGS_DIR.exists():
        return notes

    files = sorted(MEETINGS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)

    for f in files[:limit]:
        try:
            content = f.read_text()
            # Extract title from first H1
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else f.stem

            # Extract date from frontmatter
            date_match = re.search(r"date: (\d{4}-\d{2}-\d{2})", content)
            note_date = date_match.group(1) if date_match else ""

            # Check retain flag
            retain = "retain: true" in content.lower() or "retain: yes" in content.lower()

            notes.append({
                "filename": f.name,
                "title": title,
                "date": note_date,
                "retain": retain,
                "size": len(content),
            })
        except Exception:
            continue

    return notes


def get_recordings_on_disk() -> list[dict]:
    """List current recording files on disk."""
    recordings = []
    if not RECORDINGS_DIR.exists():
        return recordings

    for f in sorted(RECORDINGS_DIR.glob("chunk_*.wav"), reverse=True):
        size_mb = f.stat().st_size / (1024 * 1024)
        recordings.append({
            "filename": f.name,
            "size_mb": round(size_mb, 2),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })

    return recordings


def get_log_tail(lines: int = 30) -> str:
    """Get the last N lines of the log file."""
    if not LOG_PATH.exists():
        return "No log file found"

    try:
        with open(LOG_PATH) as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading log: {e}"


@app.route("/")
def index():
    """Main dashboard page."""
    status = get_agent_status()
    activity = get_todays_activity()
    meeting_notes = get_recent_meeting_notes()
    recordings = get_recordings_on_disk()

    return render_template(
        "index.html",
        status=status,
        activity=activity,
        meeting_notes=meeting_notes,
        recordings=recordings,
        today=date.today().strftime("%A, %B %d, %Y"),
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint for auto-refresh."""
    return jsonify({
        "status": get_agent_status(),
        "activity": get_todays_activity(),
        "recordings": get_recordings_on_disk(),
    })


@app.route("/api/logs")
def api_logs():
    """JSON endpoint for log tail."""
    return jsonify({"logs": get_log_tail(50)})


if __name__ == "__main__":
    print("Listening Agent Dashboard")
    print("http://localhost:5050")
    print()
    app.run(host="127.0.0.1", port=5050, debug=False)
