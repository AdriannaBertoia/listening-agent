#!/usr/bin/env python3
"""
Listening Agent — Entry Point
Usage:
    python run.py              Start the agent (records during meetings, synthesizes at 3pm)
    python run.py --synthesize Run synthesis immediately (useful for testing)
    python run.py --status     Check if meetings are currently detected
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent import ListeningAgent


def main():
    parser = argparse.ArgumentParser(description="Listening Agent — ADHD Second Brain Helper")
    parser.add_argument(
        "--config",
        default=str(project_root / "config.yaml"),
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Run synthesis immediately and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check meeting detection status and exit",
    )

    args = parser.parse_args()

    agent = ListeningAgent(config_path=args.config)

    if args.synthesize:
        print("Running synthesis now...")
        agent.run_synthesis_now()
        print("Done! Check your Obsidian daily note.")
    elif args.status:
        is_meeting = agent.watcher.check_meeting_active()
        app = agent.watcher.get_active_meeting_app()
        if is_meeting:
            print(f"Meeting detected in: {app}")
        else:
            print("No active meeting detected")
        print(f"Watching for: {agent.config['detection']['watched_apps']}")
    else:
        print("Starting Listening Agent...")
        print("Press Ctrl+C to stop")
        print()
        agent.start()


if __name__ == "__main__":
    main()
