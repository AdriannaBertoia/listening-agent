# Listening Agent — ADHD Second Brain Helper

A passive listening agent that automatically records your meetings, transcribes them locally, and synthesizes action items into your Obsidian daily note at 3pm Pacific every day.

## How It Works

1. **Detects meetings** — Watches for Teams/Zoom active calls (process-level, not window-level)
2. **Records audio** — Captures your mic + system audio (via BlackHole) in 5-min chunks
3. **Transcribes locally** — Whisper runs on your Mac, nothing leaves your machine
4. **Synthesizes at 3pm** — Gemini (free tier) extracts action items, to-dos, and meeting context
5. **Updates Obsidian** — Writes structured data into your daily note's existing sections
6. **Cleans up** — Deletes all audio and transcripts after synthesis (ephemeral)

Nobody in your meetings knows you're recording. No Teams "Recording started" banner. Just your local mic capturing what it already hears.

## Setup

```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh
```

This installs:
- **BlackHole 2ch** — virtual audio device for system audio capture
- **ffmpeg** — audio processing (required by Whisper)
- **PortAudio** — audio recording library
- **Python venv** with all dependencies
- **Whisper base model** — pre-downloaded for fast startup

### BlackHole Configuration (One-Time)

After installing BlackHole, you need to route system audio through it:

1. Open **Audio MIDI Setup** (search in Spotlight)
2. Click **+** in the bottom left → **Create Multi-Output Device**
3. Check both your **speakers/headphones** AND **BlackHole 2ch**
4. Right-click the Multi-Output Device → **Use This Device For Sound Output**

This sends audio to both your ears AND the virtual device simultaneously.

### Gemini API Key (Free)

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a free API key
3. Add it to `config.yaml`:
   ```yaml
   synthesis:
     gemini_api_key: "your-key-here"
   ```

## Usage

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the agent (runs until you stop it)
python run.py

# Check if a meeting is currently detected
python run.py --status

# Run synthesis manually (for testing)
python run.py --synthesize
```

## Run Automatically at Login

To start the agent every time you log in:

```bash
# Copy the launch agent plist
cp com.listening-agent.plist ~/Library/LaunchAgents/

# Load it
launchctl load ~/Library/LaunchAgents/com.listening-agent.plist

# To stop it
launchctl unload ~/Library/LaunchAgents/com.listening-agent.plist
```

## Configuration

Edit `config.yaml` to customize:

- **chunk_duration** — How long each audio segment is (default: 5 min)
- **watched_apps** — Which apps trigger recording
- **model** — Whisper model size (tiny/base/small/medium/large)
- **schedule_time** — When synthesis runs (default: 15:00 Pacific)
- **llm_provider** — "gemini" (free) or "ollama" (fully local)

## What Gets Written to Your Daily Note

The agent appends to these sections:

| Extracted Data | Daily Note Section |
|---|---|
| Action items assigned to you | **To-Dos → Must-do** |
| Follow-ups and softer items | **To-Dos → Should-do** |
| Meetings to schedule | **To-Dos → Should-do** |
| Meeting prep needed | **Meeting Prep Checklist** |
| Team updates | **Team Context → Active team topics** |
| Decisions & key notes | **Notes & Scratch Pad** |

## Privacy & Legal

- All audio is processed **locally** via Whisper — recordings never leave your Mac
- Only the transcript text is sent to Gemini for synthesis (or use Ollama for fully local)
- Recordings are deleted after synthesis (same day)
- This captures your own mic input — equivalent to taking notes
- Check your state's recording consent laws and company policy

## Troubleshooting

**"No active meeting detected" but I'm in a call:**
- Teams detection relies on UDP connections. If Teams updated, the heuristic may need tuning.
- Try running `python run.py --status` during a call to debug.

**BlackHole not capturing system audio:**
- Make sure your Mac output is set to the Multi-Output Device (not just speakers)
- Check Audio MIDI Setup — BlackHole must be checked in the Multi-Output Device

**Transcription quality is poor:**
- Upgrade Whisper model in config: change `model: "base"` to `model: "small"` (slower but better)
- Ensure your mic isn't picking up too much background noise

**Gemini API errors:**
- Free tier allows 15 requests/min. If you hit limits, switch to `ollama` in config.
- Make sure your API key is set in config.yaml

## Architecture

```
run.py                  — Entry point
config.yaml             — All settings
src/
  agent.py              — Main orchestrator / daemon
  process_watcher.py    — Detects Teams/Zoom active calls
  audio_recorder.py     — Records mic + system audio in chunks
  transcriber.py        — Whisper transcription (local)
  synthesizer.py        — LLM extraction of action items
  obsidian_writer.py    — Writes to Obsidian daily note
```

## Future (v2+)

- Screen capture + OCR for documents viewed
- Microsoft Graph integration (calendar, email subjects)
- Real-time rolling summaries
- Meeting-specific notes in `07_Meetings/`
- Weekly review auto-population
