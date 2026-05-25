"""
AzureSTTService — Streaming Speech-to-Text via Azure Cognitive Services.

Key design decisions:
- _loop is captured inside start_stream() (not __init__) using get_running_loop()
  to safely reference the correct event loop from Azure's C callback thread.
- Blocking ResultFuture.get() calls are offloaded to a thread executor so they
  never stall the asyncio event loop.
- Transcripts are timestamped at recognition time so callers can discard stale entries.
- get_transcript() uses asyncio.wait_for(queue.get()) — zero CPU polling.
"""

import asyncio
import logging
import time
import azure.cognitiveservices.speech as speechsdk
from config import settings

logger = logging.getLogger(__name__)


class AzureSTTService:
    """Streams audio to Azure STT and returns transcripts."""

    def __init__(self):
        self._speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        self._speech_config.speech_recognition_language = "en-US"
        # Wait 1.5s of silence before declaring end of utterance.
        # Prevents premature firing when the user pauses between sentences.
        self._speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500"
        )
        self._stream = None
        self._audio_config = None
        self._speech_recognizer = None

        # Queue entries: (transcript_text, monotonic_timestamp)
        self._transcript_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        self._is_finals: list[str] = []

        # Captured in start_stream() once the event loop is running
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    async def start_stream(self) -> None:
        """Open a new streaming connection to Azure."""
        # Capture the running loop here — safe to reference from Azure's callback thread
        self._loop = asyncio.get_running_loop()

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=settings.SAMPLE_RATE,
            bits_per_sample=16,
            channels=1
        )
        self._stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        self._audio_config = speechsdk.audio.AudioConfig(stream=self._stream)

        self._speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=self._audio_config
        )

        def recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            # Interim results — could be used for speculative LLM in future
            pass

        def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text.strip()
                if text:
                    self._is_finals.append(text)
                    full_transcript = " ".join(self._is_finals)
                    self._is_finals.clear()
                    timestamp = time.monotonic()
                    # call_soon_threadsafe is the correct bridge from Azure's C thread
                    self._loop.call_soon_threadsafe(
                        self._transcript_queue.put_nowait,
                        (full_transcript, timestamp)
                    )
                    logger.info("Utterance complete: %s", full_transcript)

        def canceled_cb(evt: speechsdk.SessionEventArgs):
            logger.warning("Azure STT Canceled: %s", evt)

        self._speech_recognizer.recognizing.connect(recognizing_cb)
        self._speech_recognizer.recognized.connect(recognized_cb)
        self._speech_recognizer.canceled.connect(canceled_cb)

        # Offload blocking .get() to a thread so the event loop stays free
        future = self._speech_recognizer.start_continuous_recognition_async()
        await self._loop.run_in_executor(None, future.get)
        logger.info("Azure STT stream started")

    async def close(self) -> None:
        """Close the Azure STT connection."""
        if self._speech_recognizer:
            future = self._speech_recognizer.stop_continuous_recognition_async()
            if self._loop:
                await self._loop.run_in_executor(None, future.get)
            self._speech_recognizer = None
        if self._stream:
            self._stream.close()
            self._stream = None
        logger.info("Azure STT stream closed")

    # ── Audio & Transcript I/O ────────────────────────────────

    async def send_audio(self, frame: bytes) -> None:
        """Send a PCM16 audio frame to Azure."""
        if self._stream:
            self._stream.write(frame)

    async def get_transcript(
        self, timeout: float = 2.0, max_age: float | None = None
    ) -> str:
        """Wait for and return the next available transcript.

        Args:
            timeout:  Seconds to wait before returning empty string.
            max_age:  If set, discard transcripts older than this many seconds.
                      Uses settings.STT_TRANSCRIPT_MAX_AGE_S when not overridden.
        """
        age_limit = max_age if max_age is not None else settings.STT_TRANSCRIPT_MAX_AGE_S
        try:
            text, ts = await asyncio.wait_for(
                self._transcript_queue.get(),
                timeout=timeout
            )
            age = time.monotonic() - ts
            if age > age_limit:
                logger.warning(
                    "Discarding stale transcript (%.1fs old): %s", age, text[:40]
                )
                return ""
            return text
        except asyncio.TimeoutError:
            return ""

    def clear(self) -> None:
        """Flush all queued transcripts and internal buffers."""
        self._is_finals.clear()
        while not self._transcript_queue.empty():
            try:
                self._transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("STT buffers cleared")

    def has_buffered_audio(self) -> bool:
        return len(self._is_finals) > 0
