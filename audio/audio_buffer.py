"""
AudioBuffer — PCM16 frame accumulation buffer.
Stores incoming audio frames and provides access to buffered audio data.
"""

from config import settings


class AudioBuffer:
    """Buffer for accumulating PCM16 audio frames."""

    def __init__(self, sample_rate: int | None = None):
        self._buffer = bytearray()
        self._sample_rate = sample_rate or settings.SAMPLE_RATE

    def append(self, frame: bytes) -> None:
        """Add a PCM16 audio frame to the buffer."""
        self._buffer.extend(frame)

    def get_audio(self) -> bytes:
        """Return all buffered audio as bytes."""
        return bytes(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def has_audio(self) -> bool:
        """Return True if the buffer contains any data."""
        return len(self._buffer) > 0

    @property
    def duration_ms(self) -> float:
        """Calculate the duration of buffered audio in milliseconds.

        Formula: len(buffer) / (sample_rate * 2 bytes_per_sample) * 1000
        """
        return len(self._buffer) / (self._sample_rate * 2) * 1000
