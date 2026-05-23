"""
Audio format conversion utilities for Twilio Media Streams.

Twilio audio:   µ-law (mulaw) encoded, 8000 Hz, mono, 8-bit per sample
Azure STT:      PCM 16-bit, 16000 Hz, mono
Cartesia TTS:   PCM 16-bit, 16000 Hz, mono  (output)

Conversion chain (inbound):
    Twilio mulaw 8kHz  →  ulaw_to_pcm16()  →  PCM16 8kHz
    PCM16 8kHz         →  upsample_8k_to_16k()  →  PCM16 16kHz  →  VAD / Azure STT

Conversion chain (outbound):
    Cartesia PCM16 16kHz  →  downsample_16k_to_8k()  →  PCM16 8kHz
    PCM16 8kHz            →  pcm16_to_ulaw()  →  mulaw 8kHz  →  Twilio

All functions operate on raw bytes and return raw bytes.
No external dependencies beyond numpy (already in requirements.txt).
"""

import numpy as np

# ── G.711 µ-law decode lookup table ──────────────────────────────────────────
# Maps every possible 8-bit mulaw value (0–255) to its 16-bit linear PCM value.
# Source: ITU-T G.711 standard decoding table.
_ULAW_DECODE: np.ndarray = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
     -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
     -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
     -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
     -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
     -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
     -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
      -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
      -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
      -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
      -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
      -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
       -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
     32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
     23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
     15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
     11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
      7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
      5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
      3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
      2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
      1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
      1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
       876,    844,    812,    780,    748,    716,    684,    652,
       620,    588,    556,    524,    492,    460,    428,    396,
       372,    356,    340,    324,    308,    292,    276,    260,
       244,    228,    212,    196,    180,    164,    148,    132,
       120,    112,    104,     96,     88,     80,     72,     64,
        56,     48,     40,     32,     24,     16,      8,      0,
], dtype=np.int16)


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode µ-law bytes → 16-bit linear PCM bytes (same sample count).

    Input:  N bytes of 8-bit µ-law (8000 Hz from Twilio)
    Output: 2N bytes of 16-bit PCM (8000 Hz, same sample count)
    """
    indices = np.frombuffer(data, dtype=np.uint8)
    return _ULAW_DECODE[indices].tobytes()


def pcm16_to_ulaw(data: bytes) -> bytes:
    """Encode 16-bit linear PCM bytes → µ-law bytes.

    Input:  2N bytes of 16-bit PCM (8000 Hz)
    Output: N bytes of 8-bit µ-law (for Twilio)
    """
    BIAS = 0x84   # = 132
    CLIP = 32635

    samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
    # Sign mask: 0xFF for positive, 0x7F for negative
    mask = np.where(samples >= 0, np.int32(0xFF), np.int32(0x7F))
    magnitude = np.abs(samples)
    magnitude = np.minimum(magnitude, CLIP) + BIAS

    # Exponent: floor(log2(magnitude)) − 3, clamped to [0, 7].
    # magnitude ∈ [132, 32767] → log2 ∈ [7.04, 14.99] → exp ∈ [0, 7] after −3 and clamp
    exp = np.floor(np.log2(magnitude.astype(np.float32))).astype(np.int32) - 3
    exp = np.clip(exp, 0, 7)

    mantissa = ((magnitude >> (exp + 3)) & 0x0F).astype(np.int32)
    ulaw = (~(mask & ((exp << 4) | mantissa))).astype(np.uint8)
    return ulaw.tobytes()


def upsample_8k_to_16k(data: bytes) -> bytes:
    """Upsample 8 kHz PCM16 → 16 kHz PCM16 via linear interpolation (2× rate).

    Input:  N samples at 8000 Hz  (2N bytes)
    Output: 2N samples at 16000 Hz (4N bytes)
    """
    s = np.frombuffer(data, dtype=np.int16).astype(np.int32)
    n = len(s)
    out = np.empty(n * 2, dtype=np.int32)
    out[0::2] = s                                    # original samples at even positions
    out[1::2][:-1] = (s[:-1] + s[1:]) >> 1          # midpoint interpolation
    out[-1] = s[-1]                                  # last interpolated = last input
    return out.astype(np.int16).tobytes()


def downsample_16k_to_8k(data: bytes) -> bytes:
    """Downsample 16 kHz PCM16 → 8 kHz PCM16 by taking every other sample.

    Input:  2N samples at 16000 Hz (4N bytes)
    Output: N samples at 8000 Hz   (2N bytes)
    """
    s = np.frombuffer(data, dtype=np.int16)
    return s[::2].tobytes()
