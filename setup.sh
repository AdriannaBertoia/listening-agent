#!/bin/bash
# Listening Agent — Setup Script
# Installs all dependencies needed for the ADHD listening agent
# Requires Python 3.12 for WhisperX / pyannote compatibility

set -e

echo "========================================="
echo "  Listening Agent Setup"
echo "========================================="
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "ERROR: Homebrew not found. Install from https://brew.sh"
    exit 1
fi
echo "✓ Homebrew found"

# Ensure Python 3.12 is available
echo ""
echo "--- Checking Python 3.12 ---"
if command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
    echo "✓ Python 3.12 found: $(python3.12 --version)"
elif command -v /opt/homebrew/bin/python3.12 &> /dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
    echo "✓ Python 3.12 found (Homebrew): $($PYTHON_BIN --version)"
else
    echo "Python 3.12 not found — installing via Homebrew..."
    brew install python@3.12
    PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"
    echo "✓ Python 3.12 installed: $($PYTHON_BIN --version)"
fi

# Install BlackHole (virtual audio device for system audio capture)
echo ""
echo "--- Installing BlackHole (system audio loopback) ---"
if brew list blackhole-2ch &> /dev/null 2>&1; then
    echo "✓ BlackHole already installed"
else
    brew install blackhole-2ch
    echo "✓ BlackHole installed"
    echo ""
    echo "  IMPORTANT: After install, you need to create a Multi-Output Device:"
    echo "  1. Open 'Audio MIDI Setup' (search in Spotlight)"
    echo "  2. Click '+' in the bottom left → 'Create Multi-Output Device'"
    echo "  3. Check BOTH your speakers/headphones AND 'BlackHole 2ch'"
    echo "  4. Set your Mac's sound output to this Multi-Output Device"
    echo "  This routes audio to both your ears AND the virtual device for recording."
    echo ""
fi

# Install ffmpeg (required by Whisper for audio processing)
echo ""
echo "--- Installing ffmpeg ---"
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg already installed"
else
    brew install ffmpeg
    echo "✓ ffmpeg installed"
fi

# Install PortAudio (required by sounddevice)
echo ""
echo "--- Installing PortAudio ---"
if brew list portaudio &> /dev/null 2>&1; then
    echo "✓ PortAudio already installed"
else
    brew install portaudio
    echo "✓ PortAudio installed"
fi

# Create Python virtual environment (with Python 3.12)
echo ""
echo "--- Setting up Python 3.12 environment ---"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -d "$SCRIPT_DIR/venv" ]; then
    # Check if existing venv is the right Python version
    VENV_PYTHON_VERSION=$("$SCRIPT_DIR/venv/bin/python" --version 2>/dev/null | grep -oE '3\.[0-9]+')
    if [ "$VENV_PYTHON_VERSION" != "3.12" ]; then
        echo "⚠ Existing venv is Python $VENV_PYTHON_VERSION — recreating with 3.12..."
        rm -rf "$SCRIPT_DIR/venv"
        $PYTHON_BIN -m venv "$SCRIPT_DIR/venv"
        echo "✓ Virtual environment recreated with Python 3.12"
    else
        echo "✓ Virtual environment already exists (Python 3.12)"
    fi
else
    $PYTHON_BIN -m venv "$SCRIPT_DIR/venv"
    echo "✓ Virtual environment created (Python 3.12)"
fi

# Activate and install Python packages
source "$SCRIPT_DIR/venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "✓ Python packages installed"

# Download WhisperX model
echo ""
echo "--- Pre-downloading WhisperX 'small' model ---"
python3 -c "
import whisperx
model = whisperx.load_model('small', device='cpu', compute_type='int8', language='en')
print('Model loaded successfully')
" 2>/dev/null
echo "✓ WhisperX model ready"

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Set up BlackHole Multi-Output Device (see instructions above)"
echo ""
echo "  2. Accept pyannote model terms (required for speaker diarization):"
echo "     → https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "     → https://huggingface.co/pyannote/segmentation-3.0"
echo ""
echo "  3. Get a free Gemini API key (if using Gemini):"
echo "     → https://aistudio.google.com/apikey"
echo "     Then add it to config.yaml under synthesis.gemini_api_key"
echo ""
echo "  4. Start the agent:"
echo "     source venv/bin/activate"
echo "     python run.py"
echo ""
echo "  5. (Optional) Test synthesis manually:"
echo "     python run.py --synthesize"
echo ""
echo "  6. (Optional) Run at login automatically — see README.md"
echo ""
