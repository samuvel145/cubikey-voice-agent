"""
Centralized configuration for the Voice Agent.
All settings are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── API Keys ──────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    CARTESIA_API_KEY: str = ""
    CARTESIA_VOICE_ID: str = "default"

    # ── Audio ─────────────────────────────────────────────────
    SAMPLE_RATE: int = 16000
    FRAME_DURATION_MS: int = 20
    CHANNELS: int = 1

    @property
    def FRAME_SIZE(self) -> int:
        """Bytes per audio frame: 16kHz × 2 bytes × 0.02s = 640."""
        return self.SAMPLE_RATE * 2 * self.FRAME_DURATION_MS // 1000

    # ── VAD ───────────────────────────────────────────────────
    VAD_AGGRESSIVENESS: int = 2
    SILENCE_THRESHOLD_MS: int = 800

    # ── LLM ───────────────────────────────────────────────────
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    MAX_HISTORY_TURNS: int = 5
    LLM_MAX_TOKENS: int = 300

    # ── Server ────────────────────────────────────────────────
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
