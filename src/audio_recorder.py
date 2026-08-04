"""
Audio Recorder Module
Records mic + system audio (via BlackHole) in configurable chunks.
Runs in a background thread, outputs WAV files for transcription.

Uses sounddevice InputStream for real-time recording so that partial
chunks (when a meeting ends early) are still saved.
"""

import os
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

import sounddevice as sd
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from mic and system audio loopback in chunks."""

    def __init__(
        self,
        output_dir: str,
        chunk_duration: int = 300,
        sample_rate: int = 16000,
        system_audio_device: str = "BlackHole 2ch",
        mic_device: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.system_audio_device = system_audio_device
        self.mic_device = mic_device

        self._recording = False
        self._thread: threading.Thread | None = None
        self._current_chunk_files: list[str] = []
        # Buffer for accumulating audio data
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def chunk_files(self) -> list[str]:
        """Return list of all recorded chunk file paths."""
        return list(self._current_chunk_files)

    def start(self):
        """Start recording in a background thread."""
        if self._recording:
            logger.warning("Already recording")
            return

        self._recording = True
        self._audio_buffer = []
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Recording started")

    def stop(self):
        """Stop recording gracefully, saving any partial chunk."""
        self._recording = False
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("Recording stopped")

    def _get_device_index(self, device_name: str | None) -> int | None:
        """Find device index by name. Returns None for default."""
        if device_name is None:
            return None

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if device_name.lower() in dev['name'].lower():
                return i

        logger.warning(f"Device '{device_name}' not found, using default")
        return None

    def _record_loop(self):
        """Main recording loop — uses InputStream for real-time capture."""
        # Determine which device to use
        # Priority: system_audio_device (Loopback, BlackHole, etc.) if available and configured.
        # Fallback: mic device (captures room audio including speakers).
        # Default: system default input (usually MacBook Pro Microphone).
        mic_idx = self._get_device_index(self.mic_device)
        sys_idx = self._get_device_index(self.system_audio_device)

        if sys_idx is not None and self.system_audio_device:
            # Use virtual audio device (Loopback, BlackHole, etc.)
            device = sys_idx
            device_name = self.system_audio_device
        elif mic_idx is not None:
            device = mic_idx
            device_name = self.mic_device
        else:
            # Default mic — picks up your voice + speakers in the room
            device = None
            device_name = "system default (mic)"

        logger.info(f"Recording from device: {device_name} (index: {device})")

        while self._recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_path = self.output_dir / f"chunk_{timestamp}.wav"

            try:
                self._record_chunk_stream(chunk_path, device)
            except Exception as e:
                logger.error(f"Error recording chunk: {e}")
                time.sleep(5)

    def _record_chunk_stream(self, output_path: Path, device: int | None):
        """Record a chunk using InputStream for real-time capture with early-stop support."""
        # Query the device's native sample rate to avoid resampling issues
        if device is not None:
            dev_info = sd.query_devices(device)
            native_rate = int(dev_info['default_samplerate'])
        else:
            native_rate = self.sample_rate

        audio_chunks: list[np.ndarray] = []

        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio block."""
            if status:
                logger.debug(f"Audio status: {status}")
            audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=native_rate,
                channels=1,
                device=device,
                dtype="float32",
                callback=audio_callback,
                blocksize=native_rate,  # 1 second blocks
            ):
                # Wait for chunk duration OR stop signal
                elapsed = 0
                while elapsed < self.chunk_duration and self._recording:
                    time.sleep(1)
                    elapsed += 1

        except Exception as e:
            logger.error(f"InputStream error: {e}")
            return

        # Combine all captured audio
        if not audio_chunks:
            return

        audio_data = np.concatenate(audio_chunks).flatten()

        # Only save if we have meaningful audio (at least 5 seconds and not silence)
        min_samples = native_rate * 5  # 5 seconds minimum
        if len(audio_data) < min_samples:
            logger.debug("Chunk too short, discarding")
            return

        if np.abs(audio_data).max() < 0.005:
            logger.debug("Chunk is silence, discarding")
            return

        # Resample to target sample rate if needed (Whisper expects 16kHz)
        if native_rate != self.sample_rate:
            # Simple linear resampling
            duration = len(audio_data) / native_rate
            target_samples = int(duration * self.sample_rate)
            indices = np.linspace(0, len(audio_data) - 1, target_samples)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)

        # Save the audio file
        sf.write(str(output_path), audio_data, self.sample_rate)
        self._current_chunk_files.append(str(output_path))
        duration_sec = len(audio_data) / self.sample_rate
        logger.info(f"Recorded chunk: {output_path.name} ({duration_sec:.0f}s)")

    def clear_recordings(self):
        """Delete all recorded audio files."""
        for filepath in self._current_chunk_files:
            try:
                os.unlink(filepath)
                logger.debug(f"Deleted: {filepath}")
            except OSError:
                pass

        # Also clean up any chunk files in the output dir
        for f in self.output_dir.glob("chunk_*.wav"):
            try:
                f.unlink()
            except OSError:
                pass

        self._current_chunk_files.clear()
        logger.info("All recordings cleared")
