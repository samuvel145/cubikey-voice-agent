"""
GroqLLMService — Streaming LLM response generation via Groq (llama3-70b-8192).
Sends the user transcript + conversation history and yields tokens as they arrive.
"""

import logging
from typing import AsyncGenerator
from groq import AsyncGroq
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful and concise voice assistant. Answer the user's question briefly in 1-2 "
    "simple sentences. However, ALWAYS follow the user's specific instructions for length (e.g., "
    "if they ask for two lines, provide them). If a new topic is introduced, switch to it "
    "smoothly. Keep it natural, conversational, and user-friendly with no lists or markdown."
)


class GroqLLMService:
    """Streams LLM tokens from Groq's llama3-70b-8192 model."""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    async def generate(
        self, transcript: str, history: list[dict]
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens for the given transcript.

        Args:
            transcript: The user's spoken text.
            history:    Conversation history (list of role/content dicts).

        Yields:
            Token strings as they arrive from Groq.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": transcript})

        try:
            stream = await self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                stream=True,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.7,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as exc:
            logger.error("Groq LLM error: %s", exc)
            yield ""
