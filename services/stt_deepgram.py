"""
DeepgramSTTService — Streaming Speech-to-Text via Deepgram nova-2.
Uses Deepgram SDK v6 async context-manager API.

Architecture: The service manages a background listener task that reads
messages from the Deepgram WebSocket and populates a transcript queue.
The main loop sends audio frames and retrieves transcripts via the queue.
"""

import asyncio
import logging
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd
from config import settings

logger = logging.getLogger(__name__)


class DeepgramSTTService:
    """Streams audio to Deepgram and returns transcripts."""

    def __init__(self):
        self.client = AsyncDeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        self._socket = None
        self._ctx_manager = None
        self._listener_task: asyncio.Task | None = None
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._is_finals: list[str] = []

    # ── Lifecycle ─────────────────────────────────────────────

    async def start_stream(self) -> None:
        """Open a new streaming connection to Deepgram."""
        self._ctx_manager = self.client.listen.v1.connect(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            sample_rate=str(settings.SAMPLE_RATE),
            channels="1",
            interim_results="true",
            utterance_end_ms="1000",
            vad_events="true",
        )
        self._socket = await self._ctx_manager.__aenter__()
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Deepgram STT stream started (SDK v6)")

    async def _listen_loop(self) -> None:
        """Background task: iterate messages from the Deepgram WebSocket."""
        try:
            async for message in self._socket:
                if isinstance(message, ListenV1Results):
                    self._handle_result(message)
                elif isinstance(message, ListenV1UtteranceEnd):
                    self._handle_utterance_end()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Deepgram listener error: %s", exc)

    def _handle_result(self, result: ListenV1Results) -> None:
        """Process a transcript result."""
        try:
            if result.channel and result.channel.alternatives:
                sentence = result.channel.alternatives[0].transcript or ""
                if result.is_final and sentence.strip():
                    self._is_finals.append(sentence)
                    logger.debug("Final transcript chunk: %s", sentence)
        except (IndexError, AttributeError) as exc:
            logger.warning("Transcript extraction error: %s", exc)

    def _handle_utterance_end(self) -> None:
        """Join accumulated finals into a complete utterance."""
        if self._is_finals:
            full_transcript = " ".join(self._is_finals)
            self._is_finals.clear()
            self._transcript_queue.put_nowait(full_transcript)
            logger.info("Utterance complete: %s", full_transcript)

    async def close(self) -> None:
        """Close the Deepgram connection."""
        # Cancel listener task
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        # Close the context manager
        if self._ctx_manager:
            try:
                if self._socket:
                    await self._socket.send_close_stream()
                await self._ctx_manager.__aexit__(None, None, None)
                logger.info("Deepgram STT stream closed")
            except Exception as exc:
                logger.error("Error closing Deepgram connection: %s", exc)
            self._socket = None
            self._ctx_manager = None

    # ── Audio & Transcript I/O ────────────────────────────────

    async def send_audio(self, frame: bytes) -> None:
        """Send a PCM16 audio frame to Deepgram."""
        if self._socket:
            try:
                await self._socket.send_media(frame)
            except Exception as exc:
                logger.error("Error sending audio to Deepgram: %s", exc)

    async def get_transcript(self, timeout: float = 2.0) -> str:
        """Wait for and return the first available transcript."""
        # 1. Check if one is already in the queue
        try:
            return self._transcript_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # 2. Wait for pending results or a queue item
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check the finals buffer (partial results)
            if self._is_finals:
                txt = " ".join(self._is_finals)
                self._is_finals.clear()
                return txt
            
            # Check the queue again
            try:
                return self._transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
                
            await asyncio.sleep(0.05)
        
        return ""

    def clear(self) -> None:
        """Flush state."""
        self._is_finals.clear()
        while not self._transcript_queue.empty():
            try:
                self._transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("STT buffers cleared")

    def has_buffered_audio(self) -> bool:
        return len(self._is_finals) > 0
