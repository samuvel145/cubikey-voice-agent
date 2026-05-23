"""
VADProcessor — WebRTC Voice Activity Detection wrapper.
Detects speech vs. silence in incoming PCM16 audio frames.
Returns 'speech', 'silence', or 'end_of_speech' per frame.
"""

import logging
import numpy as np
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

        # Adaptive noise floor — calibrated from first ~1s of silence at session start.
        # Prevents background noise (AC, office, TV) from triggering false speech detections.
        self._noise_floor: float = 800.0
        self._calibration_frames: list[float] = []
        self._calibrated: bool = False

        # Consecutive speech frames tracked specifically for barge-in interrupt detection.
        # Requires sustained speech (not a cough or click) before interrupt fires.
        self._interrupt_frame_count: int = 0

        logger.info(
            "VAD initialized: aggressiveness=%d, silence_threshold=%d frames",
            settings.VAD_AGGRESSIVENESS,
            self.silence_threshold_frames,
        )

    def _calibrate(self, energy: float) -> None:
        """Collect ambient noise samples and set the adaptive threshold."""
        self._calibration_frames.append(energy)
        if len(self._calibration_frames) >= 50:  # ~1 second at 20ms frames
            ambient = float(np.mean(self._calibration_frames))
            # Threshold = 2.5× ambient energy, minimum 200 to avoid near-zero floors
            self._noise_floor = max(200.0, ambient * 2.5)
            self._calibrated = True
            logger.info("Noise floor calibrated: %.1f (ambient avg: %.1f)", self._noise_floor, ambient)

    def is_speech(self, frame: bytes) -> bool:
        """Return True if the frame contains speech."""
        samples = np.frombuffer(frame, dtype=np.int16)
        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        variance = float(np.var(samples))

        # Calibrate during initial silence before the user speaks
        if not self._calibrated and not self.speech_active:
            self._calibrate(energy)

        # Adaptive energy gate — rejects frames below the noise floor
        if energy < self._noise_floor and variance < self._noise_floor ** 2:
            return False

        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def is_sustained_speech(self, frame: bytes) -> bool:
        """Return True only after INTERRUPT_MIN_FRAMES consecutive speech frames.

        Used exclusively for barge-in detection to avoid triggering on brief
        noise bursts (coughs, clicks, mic pops) during agent speech.
        """
        if self.is_speech(frame):
            self._interrupt_frame_count += 1
            return self._interrupt_frame_count >= settings.INTERRUPT_MIN_FRAMES
        else:
            self._interrupt_frame_count = 0
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
        """Reset VAD state counters (call after interrupts and new turns)."""
        self.silence_counter = 0
        self.speech_active = False
        self._interrupt_frame_count = 0
        logger.debug("VAD state reset")
