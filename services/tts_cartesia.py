"""
CartesiaTTSService — Streaming Text-to-Speech via Cartesia.
Collects LLM tokens into full text, then synthesizes PCM16 audio chunks
using the Cartesia SDK v3 tts.bytes() async iterator method.
"""

import logging
from typing import AsyncGenerator
from cartesia import AsyncCartesia
from config import settings

logger = logging.getLogger(__name__)


class CartesiaTTSService:
    """Synthesizes speech audio from text using Cartesia."""

    def __init__(self):
        self.client = AsyncCartesia(api_key=settings.CARTESIA_API_KEY)
        self.voice_id = settings.CARTESIA_VOICE_ID

    async def synthesize(
        self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize audio from tokens sentence-by-sentence to reduce latency.
        """
        current_sentence = ""
        terminators = {".", "!", "?", "\n"}

        async for token in token_stream:
            current_sentence += token
            
            # If we hit a sentence terminator, synthesize and yield
            if any(t in token for t in terminators) and len(current_sentence.strip()) > 5:
                logger.info("Synthesizing sentence: %s", current_sentence[:40] + "...")
                async for chunk in self._generate_audio(current_sentence):
                    yield chunk
                current_sentence = ""

        # Handle any remaining text
        if current_sentence.strip():
            logger.info("Synthesizing final chunk: %s", current_sentence[:40] + "...")
            async for chunk in self._generate_audio(current_sentence):
                yield chunk

    async def _generate_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Internal helper to call Cartesia TTS API."""
        try:
            audio_iter = await self.client.tts.bytes(
                model_id="sonic",
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
        except Exception as exc:
            logger.error("Cartesia TTS generation error: %s", exc)

    async def close(self) -> None:
        """Close the Cartesia client."""
        try:
            await self.client.close()
            logger.info("Cartesia TTS client closed")
        except Exception as exc:
            logger.error("Error closing Cartesia client: %s", exc)
