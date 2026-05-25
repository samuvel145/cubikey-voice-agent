"""
Twilio Media Streams Handler — bridges phone calls to the voice agent.

Protocol:
  POST /twilio/voice  →  TwiML <Connect><Stream> redirects call audio to WS
  WS   /ws/twilio    →  Twilio sends/receives base64-encoded mulaw 8 kHz audio

Audio conversion chain:
  Inbound (caller → STT):
    base64 → mulaw 8kHz → ulaw_to_pcm16 → PCM16 8kHz
                        → upsample_8k_to_16k → PCM16 16kHz → VAD / Azure STT

  Outbound (TTS → caller):
    Cartesia PCM16 16kHz → downsample_16k_to_8k → PCM16 8kHz
                        → pcm16_to_ulaw → mulaw 8kHz → base64 → Twilio JSON

Twilio message types received:
  {"event": "connected", ...}
  {"event": "start", "start": {"streamSid": "MX...", ...}}
  {"event": "media",  "media": {"track": "inbound", "payload": "<b64>"}}
  {"event": "stop",   ...}

Twilio message types sent:
  {"event": "media",  "streamSid": "MX...", "media": {"payload": "<b64>"}}
  {"event": "clear",  "streamSid": "MX..."}   ← flush buffered audio (barge-in)
"""

import asyncio
import base64
import json
import logging
import random
from uuid import uuid4
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from config import settings
from audio.vad import VADProcessor
from audio.converter import (
    ulaw_to_pcm16,
    pcm16_to_ulaw,
    upsample_8k_to_16k,
    downsample_16k_to_8k,
)
from services.stt_azure import AzureSTTService
from services.llm_azure import AzureLLMService
from services.tts_cartesia import CartesiaTTSService
from session.session_manager import session_manager
from app.metrics import increment_connections

logger = logging.getLogger(__name__)
router = APIRouter()

# Scripted greeting played immediately when the call connects — no LLM latency.
TWILIO_GREETING = (
    "Hello, thank you for calling Cubikey. "
    "This is Maya, your AI assistant. "
    "How can I help you today?"
)

ACKNOWLEDGMENTS = ["Sure.", "Got it.", "Okay.", "Alright.", "Mm-hmm."]


# ── TwiML Webhook ─────────────────────────────────────────────────────────────

@router.post("/twilio/voice")
async def twilio_voice_webhook(request: Request) -> Response:
    """
    Twilio calls this URL when a call arrives on the configured phone number.
    We respond with TwiML that tells Twilio to stream bi-directional audio to
    our /ws/twilio WebSocket endpoint.
    """
    base = settings.PUBLIC_URL.rstrip("/")
    if base.startswith("https://"):
        ws_url = "wss://" + base[len("https://"):] + "/ws/twilio"
    else:
        ws_url = "ws://" + base[len("http://"):] + "/ws/twilio"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}"/>'
        "</Connect>"
        "</Response>"
    )
    logger.info("Twilio webhook hit — streaming to %s", ws_url)
    return Response(content=twiml, media_type="text/xml")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def async_gen_from_string(text: str) -> AsyncGenerator[str, None]:
    """Wrap a plain string as a single-yield async generator (for TTS)."""
    yield text


async def send_audio_to_twilio(
    websocket: WebSocket,
    pcm16_16k: bytes,
    stream_sid: str,
) -> None:
    """
    Convert Cartesia's PCM16 16kHz output → mulaw 8kHz → base64 → Twilio JSON.
    Skips silently if stream_sid is not yet known.
    """
    if not stream_sid:
        return
    pcm16_8k = downsample_16k_to_8k(pcm16_16k)
    mulaw = pcm16_to_ulaw(pcm16_8k)
    payload = base64.b64encode(mulaw).decode("ascii")
    await websocket.send_text(json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {"payload": payload},
    }))


async def clear_twilio_buffer(websocket: WebSocket, stream_sid: str) -> None:
    """Tell Twilio to discard any audio it has buffered but not yet played."""
    if not stream_sid:
        return
    try:
        await websocket.send_text(json.dumps({
            "event": "clear",
            "streamSid": stream_sid,
        }))
    except Exception:
        pass


async def handle_interrupt(session, websocket: WebSocket, stream_sid: str) -> None:
    """Cancel the running pipeline and flush Twilio's playback buffer."""
    session.is_interrupted = True
    session.is_agent_speaking = False
    await session.cancel_tasks()
    session.reset_speaking_state()
    await clear_twilio_buffer(websocket, stream_sid)
    logger.info("Barge-in interrupt completed for Twilio session %s", session.session_id)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_pipeline(
    transcript: str,
    session,
    websocket: WebSocket,
    stream_sid: str,
    llm: AzureLLMService,
    tts: CartesiaTTSService,
    stt: AzureSTTService,
    is_greeting: bool = False,
) -> None:
    """
    LLM → TTS pipeline.  Steps:
      1. Mark agent speaking; flush stale STT transcripts.
      2. Greeting: direct TTS, no LLM call.
      3. Regular turn: instant ack token → LLM streaming → TTS.
      4. Save assistant turn to history.
    """
    try:
        session.is_agent_speaking = True
        session.last_vocal_start = asyncio.get_event_loop().time()
        stt.clear()  # discard transcripts queued during previous agent speech

        if is_greeting:
            logger.info("Sending Twilio scripted greeting")
            async for chunk in tts.synthesize(async_gen_from_string(transcript)):
                if session.is_interrupted:
                    return
                await send_audio_to_twilio(websocket, chunk, stream_sid)
            return  # greeting not saved to history

        # ── Regular turn: instant ack → LLM → TTS ────────────────────────
        ack = random.choice(ACKNOWLEDGMENTS)
        async for chunk in tts.synthesize(async_gen_from_string(ack)):
            if session.is_interrupted:
                return
            await send_audio_to_twilio(websocket, chunk, stream_sid)

        if session.is_interrupted:
            return

        logger.info("Twilio pipeline started for: %s", transcript[:60])

        full_response: list[str] = []

        async def token_wrapper() -> AsyncGenerator[str, None]:
            async for token in llm.generate(transcript, session.history):
                if session.is_interrupted:
                    break
                full_response.append(token)
                yield token

        async for audio_chunk in tts.synthesize(token_wrapper()):
            if session.is_interrupted:
                break
            await send_audio_to_twilio(websocket, audio_chunk, stream_sid)

        final_text = "".join(full_response)
        if final_text.strip():
            session.add_turn("assistant", final_text)

    except asyncio.CancelledError:
        raise  # propagate so cancel_tasks() can await cleanly
    except Exception as exc:
        logger.error("Twilio pipeline error for session %s: %s", session.session_id, exc)
    finally:
        session.is_agent_speaking = False
        session.is_greeting = False
        session.last_vocal_start = None
        session.is_interrupted = False


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/twilio")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """
    Main Twilio Media Stream WebSocket handler — dual-loop architecture.

    audio_loop      — parses Twilio JSON events, converts mulaw→PCM16, runs
                      VAD, feeds STT, detects barge-in.
    transcript_loop — awaits Azure STT transcripts, fires LLM→TTS pipeline.
    """
    await websocket.accept()

    session_id = str(uuid4())
    session = session_manager.create_session(session_id)
    increment_connections()

    vad = VADProcessor()
    stt = AzureSTTService()
    llm = AzureLLMService()
    tts = CartesiaTTSService()

    # Mutable container: stream_sid arrives in the Twilio 'start' event.
    # Using a list so the inner functions (audio_loop, transcript_loop,
    # run_pipeline) all share the same reference without nonlocal gymnastics.
    stream_sid: list[str] = [""]

    try:
        logger.info("Twilio session %s accepted", session_id)
        await stt.start_stream()

        # ── audio_loop ────────────────────────────────────────────────────
        async def audio_loop() -> None:
            audio_buffer = bytearray()
            last_silence_send = 0.0
            greeting_started = False

            async for raw in websocket.iter_text():
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = data.get("event")

                # ── Twilio lifecycle events ───────────────────────────────
                if event == "connected":
                    logger.info("Twilio 'connected' for session %s", session_id)
                    continue

                if event == "start":
                    stream_sid[0] = data["start"]["streamSid"]
                    logger.info(
                        "Twilio stream started (streamSid=%s) for session %s",
                        stream_sid[0], session_id,
                    )
                    # Fire the greeting now that we know the stream_sid
                    if not greeting_started:
                        greeting_started = True
                        session.tts_task = asyncio.create_task(
                            run_pipeline(
                                TWILIO_GREETING,
                                session, websocket, stream_sid[0],
                                llm, tts, stt,
                                is_greeting=True,
                            )
                        )
                    continue

                if event == "stop":
                    logger.info("Twilio stream stopped for session %s", session_id)
                    break

                if event != "media":
                    continue

                # ── Audio frame processing ────────────────────────────────
                media = data.get("media", {})
                if media.get("track") != "inbound":
                    continue  # ignore outbound echo

                mulaw_bytes = base64.b64decode(media["payload"])

                # mulaw 8kHz → PCM16 8kHz → PCM16 16kHz
                pcm16_8k = ulaw_to_pcm16(mulaw_bytes)
                pcm16_16k = upsample_8k_to_16k(pcm16_8k)

                audio_buffer.extend(pcm16_16k)

                # Cap buffer to prevent stale frame buildup
                max_buf = settings.MAX_AUDIO_BUFFER_BYTES
                if len(audio_buffer) > max_buf:
                    audio_buffer = audio_buffer[-max_buf:]

                # Drain complete VAD frames (640 bytes = 20 ms at 16kHz)
                while len(audio_buffer) >= settings.FRAME_SIZE:
                    frame = bytes(audio_buffer[: settings.FRAME_SIZE])
                    audio_buffer = audio_buffer[settings.FRAME_SIZE:]

                    # ── Barge-in detection ────────────────────────────
                    if session.is_agent_speaking and stream_sid[0]:
                        vocal_start = (
                            session.last_vocal_start
                            or asyncio.get_event_loop().time()
                        )
                        elapsed = asyncio.get_event_loop().time() - vocal_start

                        if elapsed >= settings.INTERRUPT_COOLDOWN_S:
                            if vad.is_sustained_speech(frame):
                                logger.info(
                                    "Twilio barge-in detected for session %s",
                                    session_id,
                                )
                                await handle_interrupt(
                                    session, websocket, stream_sid[0]
                                )
                                stt.clear()
                                vad.reset()
                                await stt.send_audio(frame)
                                continue

                    # ── VAD → STT ─────────────────────────────────────
                    result = vad.process_frame(frame)

                    if not session.is_agent_speaking:
                        # Push every frame to Azure STT — silence frames are
                        # required for Azure's internal VAD to fire recognized_cb.
                        if result == "speech":
                            await stt.send_audio(frame)
                        else:
                            await stt.send_audio(b"\x00" * settings.FRAME_SIZE)
                    else:
                        # Agent speaking — minimal keepalive to hold STT connection
                        now = asyncio.get_event_loop().time()
                        if now - last_silence_send >= 0.5:
                            await stt.send_audio(b"\x00" * settings.FRAME_SIZE)
                            last_silence_send = now

        # ── transcript_loop ───────────────────────────────────────────────
        async def transcript_loop() -> None:
            try:
                while True:
                    transcript = await stt.get_transcript(
                        timeout=5.0,
                        max_age=settings.STT_TRANSCRIPT_MAX_AGE_S,
                    )

                    if not transcript:
                        logger.debug("Twilio STT timeout — still listening")
                        continue

                    logger.info("Twilio STT received: '%s'", transcript)

                    if session.is_agent_speaking:
                        logger.info(
                            "Discarding transcript (agent speaking): %.40s", transcript
                        )
                        continue

                    if len(transcript.strip()) < 2:
                        continue

                    logger.info("📞 Caller: %s", transcript)
                    session.add_turn("user", transcript)
                    stt.clear()

                    logger.info("🤖 AI responding to Twilio caller...")
                    session.tts_task = asyncio.create_task(
                        run_pipeline(
                            transcript,
                            session, websocket, stream_sid[0],
                            llm, tts, stt,
                        )
                    )
                    try:
                        await session.tts_task
                    except asyncio.CancelledError:
                        pass  # barge-in interrupted — return to listening

            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception(
                    "Twilio transcript loop error for session %s: %s",
                    session_id, exc,
                )

        # Run transcript_loop in background; foreground runs audio_loop.
        transcript_task = asyncio.create_task(transcript_loop())
        try:
            await audio_loop()
        except WebSocketDisconnect:
            logger.info("Twilio session %s disconnected", session_id)
        except Exception as exc:
            logger.exception(
                "Twilio audio loop crashed for session %s: %s", session_id, exc
            )
        finally:
            transcript_task.cancel()
            await asyncio.gather(transcript_task, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("Twilio session %s disconnected at setup", session_id)
    except Exception as exc:
        logger.exception("Twilio session %s crashed: %s", session_id, exc)
    finally:
        await session.cancel_tasks()
        await stt.close()
        await tts.close()
        session_manager.delete_session(session_id)
        logger.info("Twilio session %s cleaned up", session_id)
