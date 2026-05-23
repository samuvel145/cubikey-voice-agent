"""
CartesiaTTSService — Streaming Text-to-Speech via Cartesia.

Pipeline design:
- A producer task consumes LLM tokens and segments them into sentences,
  pushing each sentence into an asyncio.Queue.
- The synthesize() async generator drains that queue and calls Cartesia
  concurrently — while Cartesia synthesizes sentence N, the LLM producer
  is already buffering sentence N+1, eliminating inter-sentence silence gaps.
- _generate_audio() retries up to 2 times on transient Cartesia errors.
"""

import asyncio
import logging
from typing import AsyncGenerator
from cartesia import AsyncCartesia
from config import settings

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = frozenset({".", "!", "?", "\n"})
_MIN_CHARS_FOR_EARLY_EMIT = 60


class CartesiaTTSService:
    """Synthesizes speech audio from text using Cartesia."""

    def __init__(self):
        self.client = AsyncCartesia(api_key=settings.CARTESIA_API_KEY)
        self.voice_id = settings.CARTESIA_VOICE_ID

    async def synthesize(
        self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from a token stream using a producer-consumer queue.

        The LLM producer and Cartesia TTS consumer run concurrently:
        while Cartesia synthesizes sentence N the LLM is already buffering N+1.
        """
        # Unlimited queue: producer never blocks on put(), preventing a deadlock
        # where the consumer has exited (on interrupt) but the producer's finally
        # block is stuck waiting to put the sentinel None into a full queue.
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def producer() -> None:
            current = ""
            try:
                async for token in token_stream:
                    current += token
                    should_emit = (
                        any(t in token for t in _SENTENCE_TERMINATORS) and len(current.strip()) > 5
                    ) or len(current.strip()) >= _MIN_CHARS_FOR_EARLY_EMIT

                    if should_emit:
                        await sentence_queue.put(current)
                        current = ""

                if current.strip():
                    await sentence_queue.put(current)
            finally:
                # put_nowait so this never blocks even if the consumer has already exited.
                # With an unlimited queue this always succeeds.
                sentence_queue.put_nowait(None)

        producer_task = asyncio.create_task(producer())

        try:
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                logger.info("Synthesizing: %.40s...", sentence)
                async for chunk in self._generate_audio(sentence):
                    yield chunk
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _generate_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Call Cartesia TTS with up to 2 retries on transient failure."""
        for attempt in range(3):
            try:
                audio_iter = await self.client.tts.bytes(
                    model_id="sonic-2",
                    transcript=text,
                    voice={"mode": "id", "id": self.voice_id},
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": settings.SAMPLE_RATE,
                    },
                )
                async for chunk in audio_iter:
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt < 2:
                    wait = 0.2 * (attempt + 1)
                    logger.warning(
                        "Cartesia attempt %d failed: %s — retrying in %.1fs", attempt + 1, exc, wait
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("Cartesia TTS failed after 3 attempts: %s", exc)

    async def close(self) -> None:
        """Close the Cartesia client."""
        try:
            await self.client.close()
            logger.info("Cartesia TTS client closed")
        except Exception as exc:
            logger.error("Error closing Cartesia client: %s", exc)
