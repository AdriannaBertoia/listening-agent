"""
Transcription Module
Processes audio chunks through WhisperX or Gemini for speech-to-text with speaker diarization.

Providers:
  - "gemini": Uploads audio to Gemini API for fast cloud transcription with speaker labels.
              Recommended for speed (seconds vs minutes per chunk).
  - "whisperx": Local WhisperX + pyannote diarization on CPU.
              Slower but fully private (no data leaves your machine).

Falls back to WhisperX if Gemini fails or is unavailable.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes audio files using Gemini (fast, cloud) or WhisperX (slow, local)."""

    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        output_dir: str = "data/transcripts",
        diarization_enabled: bool = True,
        hf_token: str = "",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        transcription_provider: str = "whisperx",
        gemini_api_key: str = "",
    ):
        self.model_size = model_size
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diarization_enabled = diarization_enabled and bool(hf_token)
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Provider selection
        self.provider = transcription_provider
        self.gemini_api_key = gemini_api_key

        self._model = None
        self._diarize_model = None

    @property
    def model(self):
        """Lazy-load the WhisperX model."""
        if self._model is None:
            import whisperx
            logger.info(f"Loading WhisperX model: {self.model_size}")
            self._model = whisperx.load_model(
                self.model_size,
                device="cpu",
                compute_type="int8",
                language=self.language,
            )
            logger.info("WhisperX model loaded")
        return self._model

    @property
    def diarize_model(self):
        """Lazy-load the diarization pipeline."""
        if self._diarize_model is None and self.diarization_enabled:
            from whisperx.diarize import DiarizationPipeline
            logger.info("Loading speaker diarization model...")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device="cpu",
            )
            logger.info("Diarization model loaded")
        return self._diarize_model

    def transcribe_file(self, audio_path: str) -> str | None:
        """
        Transcribe a single audio file.
        Routes to Gemini (fast cloud) or WhisperX (slow local) based on config.
        Falls back to WhisperX if Gemini fails.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if self.provider == "gemini" and self.gemini_api_key:
            result = self._transcribe_gemini(audio_path)
            if result:
                return result
            logger.warning("Gemini transcription failed — falling back to WhisperX")

        return self._transcribe_whisperx(audio_path)

    def _transcribe_gemini(self, audio_path: Path) -> str | None:
        """
        Transcribe using Gemini API (audio upload).
        Fast — typically 10-30 seconds for a 5-minute chunk.
        Returns speaker-labeled transcript text.
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            logger.info(f"Transcribing via Gemini: {audio_path.name}")

            # Upload the audio file
            audio_file = genai.upload_file(str(audio_path), mime_type="audio/wav")

            # Request transcription with speaker labels
            prompt = (
                "Transcribe this audio with speaker diarization. "
                "Format each speaker turn as: [SPEAKER_XX] text\n"
                "Use consistent speaker labels (SPEAKER_01, SPEAKER_02, etc.) throughout. "
                "Include all speech, even small interjections. "
                "Do NOT summarize — provide the full verbatim transcript."
            )

            response = model.generate_content([prompt, audio_file])
            transcript = response.text.strip()

            if not transcript:
                logger.debug(f"Gemini returned empty transcript for {audio_path.name}")
                return None

            # Save transcript to file
            self._save_transcript_text(audio_path.stem, transcript)
            logger.info(f"Gemini transcript saved: {audio_path.name} ({len(transcript)} chars)")

            # Clean up uploaded file
            try:
                audio_file.delete()
            except Exception:
                pass

            return transcript

        except Exception as e:
            logger.error(f"Gemini transcription failed for {audio_path.name}: {e}")
            return None

    def _save_transcript_text(self, chunk_name: str, text: str) -> Path:
        """Save raw transcript text to a file."""
        transcript_name = f"transcript_{chunk_name.replace('chunk_', '')}.txt"
        transcript_path = self.output_dir / transcript_name
        transcript_path.write_text(text)
        return transcript_path

    def _transcribe_whisperx(self, audio_path: Path) -> str | None:
        """
        Transcribe using local WhisperX with optional speaker diarization.
        Slower but fully private — no data leaves your machine.
        """
        import whisperx

        try:
            logger.info(f"Transcribing: {audio_path.name}")

            # Step 1: Load audio
            audio = whisperx.load_audio(str(audio_path))

            # Step 2: Transcribe with WhisperX
            result = self.model.transcribe(audio, batch_size=8)

            if not result.get("segments"):
                logger.debug(f"No speech detected in {audio_path.name}")
                return None

            # Step 3: Align timestamps at word level
            align_model, metadata = whisperx.load_align_model(
                language_code=self.language, device="cpu"
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device="cpu"
            )

            # Step 4: Speaker diarization (if enabled)
            if self.diarization_enabled and self.diarize_model:
                try:
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers,
                    )
                    result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
                    logger.info("Speaker diarization applied")
                except Exception as e:
                    logger.warning(f"Diarization failed, continuing without speakers: {e}")

            # Step 5: Format output
            segments = result.get("segments", [])
            formatted_segments = []
            full_text_parts = []

            for seg in segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "").strip()
                start = seg.get("start", 0)
                end = seg.get("end", 0)

                if not text:
                    continue

                formatted_segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                })

                if self.diarization_enabled:
                    full_text_parts.append(f"[{speaker}] {text}")
                else:
                    full_text_parts.append(text)

            full_text = "\n".join(full_text_parts) if self.diarization_enabled else " ".join(full_text_parts)

            if not full_text:
                logger.debug(f"No speech detected in {audio_path.name}")
                return None

            # Save transcript to file
            transcript_path = self._save_transcript(audio_path.stem, full_text, formatted_segments)
            logger.info(f"Transcript saved: {transcript_path}")

            return full_text

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path.name}: {e}")
            return None

    def transcribe_all(self, audio_files: list[str]) -> list[dict]:
        """
        Transcribe all audio files and return structured results.
        Returns list of dicts with timestamp, text, and source file info.
        """
        results = []

        for audio_path in sorted(audio_files):
            text = self.transcribe_file(audio_path)
            if text:
                filename = Path(audio_path).stem
                timestamp = self._parse_timestamp(filename)

                results.append({
                    "timestamp": timestamp,
                    "text": text,
                    "source_file": audio_path,
                })

        return results

    def get_todays_transcripts(self) -> str:
        """Read all transcript files from today and return combined text."""
        today = datetime.now().strftime("%Y%m%d")
        transcripts = []

        for f in sorted(self.output_dir.glob(f"transcript_{today}_*.txt")):
            content = f.read_text().strip()
            if content:
                transcripts.append(content)

        return "\n\n---\n\n".join(transcripts)

    def _save_transcript(self, chunk_name: str, text: str, segments: list[dict]) -> Path:
        """Save transcript with timestamps and speaker labels to a text file."""
        transcript_name = f"transcript_{chunk_name.replace('chunk_', '')}.txt"
        transcript_path = self.output_dir / transcript_name

        lines = []
        if segments:
            current_speaker = None
            for seg in segments:
                start = self._format_time(seg["start"])
                end = self._format_time(seg["end"])
                speaker = seg.get("speaker", "")

                if self.diarization_enabled and speaker:
                    # Group consecutive segments by speaker for readability
                    if speaker != current_speaker:
                        if lines:
                            lines.append("")  # Blank line between speakers
                        lines.append(f"**{speaker}** [{start}]")
                        current_speaker = speaker
                    lines.append(f"  {seg['text']}")
                else:
                    lines.append(f"[{start} - {end}] {seg['text']}")
        else:
            lines.append(text)

        transcript_path.write_text("\n".join(lines))
        return transcript_path

    def _parse_timestamp(self, filename: str) -> str:
        """Parse timestamp from chunk filename."""
        try:
            parts = filename.replace("chunk_", "").split("_")
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return dt.strftime("%I:%M %p")
        except (ValueError, IndexError):
            pass
        return "Unknown time"

    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def clear_transcripts(self):
        """Delete all transcript files."""
        for f in self.output_dir.glob("transcript_*.txt"):
            f.unlink()
        logger.info("All transcripts cleared")
