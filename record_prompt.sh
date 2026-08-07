#!/bin/bash
# Launched by the listening agent to show a recording prompt dialog.
# Runs in the user's GUI session (not the launchd background context).
# Arg 1: meeting title (or "Meeting detected in Microsoft Teams")

TITLE="${1:-Meeting detected}"

RESULT=$(osascript -e "
tell application \"System Events\" to activate
display dialog \"$TITLE\n\nRecord this meeting?\" buttons {\"Skip\", \"Record\"} default button \"Record\" with title \"Listening Agent\" with icon caution giving up after 15
")

if echo "$RESULT" | grep -q "Record"; then
    echo "RECORD"
elif echo "$RESULT" | grep -q "gave up:true"; then
    echo "TIMEOUT"
else
    echo "SKIP"
fi
