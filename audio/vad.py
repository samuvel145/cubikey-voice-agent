"""
VADProcessor — WebRTC Voice Activity Detection wrapper.
Detects speech vs. silence in incoming PCM16 audio frames.
Returns 'speech', 'silence', or 'end_of_speech' per frame.
"""

import logging
import webrtcvad
from config import settings

logger = logging.getLogger(__name__)


class VADProcessor:
    """Voice Activity Detection using WebRTC VAD."""

    def __init__(self):
        self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
        self.sample_rate = settings.SAMPLE_RATE
        self.silence_counter: int = 0
        self.speech_active: bool = False
        self.silence_threshold_frames: int = (
            settings.SILENCE_THRESHOLD_MS // settings.FRAME_DURATION_MS
        )
        logger.info(
            "VAD initialized: aggressiveness=%d, silence_threshold=%d frames",
            settings.VAD_AGGRESSIVENESS,
            self.silence_threshold_frames,
        )

    def is_speech(self, frame: bytes) -> bool:
        """Return True if the frame contains speech."""
        import numpy as np
        samples = np.frombuffer(frame, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32)**2))
        variance = np.var(samples)

        # Human vocal footprint check
        if energy < 1000 and variance < 1000000:
            return False

        # Final check with WebRTC VAD
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def process_frame(self, frame: bytes) -> str:
        """Process a single audio frame and return the current state.

        Returns:
            'speech'        — frame contains speech
            'silence'       — frame is silence (no prior speech)
            'end_of_speech' — silence threshold reached after speech
        """
        if self.is_speech(frame):
            self.silence_counter = 0
            self.speech_active = True
            return "speech"

        if self.speech_active:
            self.silence_counter += 1
            if self.silence_counter >= self.silence_threshold_frames:
                self.silence_counter = 0
                self.speech_active = False
                logger.debug("End of speech detected")
                return "end_of_speech"
            # Brief pause within speech — treat as still speaking
            return "speech"

        return "silence"

    def reset(self) -> None:
        """Reset VAD state counters."""
        self.silence_counter = 0
        self.speech_active = False
        logger.debug("VAD state reset")
