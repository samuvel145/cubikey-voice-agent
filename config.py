"""
Centralized configuration for the Voice Agent.
All settings are loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── API Keys ──────────────────────────────────────────────
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = ""
    AZURE_SPEECH_ENDPOINT: str = ""

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = ""

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

    @property
    def MAX_AUDIO_BUFFER_BYTES(self) -> int:
        """Cap incoming audio buffer to 2 seconds to prevent stale frame buildup."""
        return self.SAMPLE_RATE * 2 * 2

    # ── VAD ───────────────────────────────────────────────────
    VAD_AGGRESSIVENESS: int = 2
    SILENCE_THRESHOLD_MS: int = 450

    # ── Interruption ──────────────────────────────────────────
    # Seconds after agent starts speaking before a barge-in can fire.
    # Prevents false interrupts from the agent's own audio bleeding into mic.
    INTERRUPT_COOLDOWN_S: float = 0.5
    # Consecutive speech frames required to trigger a barge-in interrupt.
    # At 20ms/frame: 3 frames = 60ms of sustained speech required.
    INTERRUPT_MIN_FRAMES: int = 3

    # ── STT ───────────────────────────────────────────────────
    # Discard queued transcripts older than this (seconds).
    STT_TRANSCRIPT_MAX_AGE_S: float = 3.0

    # ── LLM ───────────────────────────────────────────────────
    MAX_HISTORY_TURNS: int = 10
    LLM_MAX_TOKENS: int = 400

    # ── Server ────────────────────────────────────────────────
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
