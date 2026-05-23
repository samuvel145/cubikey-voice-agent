"""
Session Manager — Per-connection state and conversation history.
Each WebSocket connection gets an isolated Session with rolling history.
"""

import asyncio
import logging
import time
from config import settings

logger = logging.getLogger(__name__)


class Session:
    """Holds the state of a single voice agent session."""

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.history: list[dict] = []
        self.is_agent_speaking: bool = False
        self.is_interrupted: bool = False
        self.is_greeting: bool = False
        self.last_vocal_start: float | None = None
        self.tts_task: asyncio.Task | None = None
        self.created_at: float = time.time()

    def add_turn(self, role: str, content: str) -> None:
        """Append a message and drop the oldest pair if over the limit."""
        self.history.append({"role": role, "content": content})
        max_messages = settings.MAX_HISTORY_TURNS * 2
        while len(self.history) > max_messages:
            self.history.pop(0)

    def clear_history(self) -> None:
        self.history.clear()

    async def cancel_tasks(self) -> None:
        """Cancel any active TTS task and wait for it to finish."""
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await asyncio.gather(self.tts_task, return_exceptions=True)
            except Exception:
                pass
            logger.debug("TTS task cancelled for session %s", self.session_id)

        # Allow the loop to flush any pending callbacks before callers proceed.
        await asyncio.sleep(0)

    def reset_speaking_state(self) -> None:
        """Reset agent speaking / interrupted flags."""
        self.is_agent_speaking = False
        self.is_interrupted = False


class SessionManager:
    """Manages all active voice agent sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self, session_id: str) -> Session:
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        logger.info("Session created: %s", session_id)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session deleted: %s", session_id)

    def active_count(self) -> int:
        return len(self._sessions)


# Global singleton instance
session_manager = SessionManager()
