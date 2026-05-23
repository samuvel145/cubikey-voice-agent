"""
WebSocket Handler — Pipeline orchestration for the voice agent.

Architecture: dual-loop
  audio_loop      — reads WebSocket frames, runs VAD, sends speech to Azure STT,
                    detects barge-in interrupts.
  transcript_loop — awaits Azure STT transcripts (push-based, zero polling),
                    gates on agent-speaking state, fires LLM→TTS pipeline.

Key fixes over the original design:
  - Transcripts are driven by Azure STT callbacks, not VAD end_of_speech polling.
    This eliminates the "answers previous question" bug where VAD fired before
    Azure finalized, causing the next utterance's end_of_speech to dequeue a
    stale transcript.
  - stt.clear() is called at the START of every pipeline so queued transcripts
    from agent-speech periods are discarded before they can be consumed.
  - Barge-in requires INTERRUPT_MIN_FRAMES consecutive speech frames and a
    INTERRUPT_COOLDOWN_S grace window — prevents false interrupts from noise.
  - Instant acknowledgment tokens ("Sure.", "Got it.") synthesized before LLM
    call so the user hears a response immediately while the LLM processes.
  - Greeting goes through LLM for natural, persona-consistent speech.
  - No artificial post-TTS sleep — is_agent_speaking tracks real audio flow.
  - Audio buffer is capped to MAX_AUDIO_BUFFER_BYTES to prevent stale frames.
  - Silence keep-alive uses a monotonic timer (not unreliable modulo-time).
"""

import asyncio
import logging
import random
from uuid import uuid4
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from audio.vad import VADProcessor
from services.stt_azure import AzureSTTService
from services.llm_azure import AzureLLMService
from services.tts_cartesia import CartesiaTTSService
from session.session_manager import session_manager
from app.metrics import increment_connections

logger = logging.getLogger(__name__)
router = APIRouter()

# Scripted greeting spoken as-is via TTS — no LLM call, instant playback.
# Commas create natural speech pauses between service names.
GREETING_TEXT = (
    "Hello, thank you for calling Cubikey. "
    "This is Maya, your AI assistant. "
    "We help businesses with digital growth, AI-powered marketing, "
    "Answer Engine Optimization, paid advertising, social media, websites, "
    "dashboards and analytics, AI automations, and enterprise ABM solutions. "
    "How may I assist you today?"
)

ACKNOWLEDGMENTS = ["Sure.", "Got it.", "Okay.", "Alright.", "Mm-hmm."]


# ── Helpers ──────────────────────────────────────────────────

async def async_gen_from_string(text: str) -> AsyncGenerator[str, None]:
    """Wrap a plain string as a single-yield async generator."""
    yield text


async def handle_interrupt(session, websocket: WebSocket) -> None:
    """Stop any in-progress pipeline immediately."""
    session.is_interrupted = True
    session.is_agent_speaking = False
    await session.cancel_tasks()
    session.reset_speaking_state()
    logger.info("Interruption completed for session: %s", session.session_id)


async def run_pipeline(
    transcript: str,
    session,
    websocket: WebSocket,
    llm: AzureLLMService,
    tts: CartesiaTTSService,
    stt: AzureSTTService,
    is_greeting: bool = False,
) -> None:
    """Run the LLM → TTS pipeline and stream audio to the client.

    Steps:
      1. Mark agent as speaking and flush the STT queue (discard stale transcripts).
      2. For non-greeting turns: synthesize an instant acknowledgment word
         so the user hears a response while the LLM is still processing.
      3. Stream LLM tokens → TTS sentence segments → WebSocket audio chunks.
      4. Record the assistant turn in history.
    """
    try:
        session.is_agent_speaking = True
        session.last_vocal_start = asyncio.get_event_loop().time()
        stt.clear()  # discard any transcripts queued while agent was last speaking

        if is_greeting:
            # ── Scripted greeting — direct TTS, no LLM call ───────
            # Gives instant playback with zero LLM latency.
            logger.info("Sending scripted greeting")
            async for chunk in tts.synthesize(async_gen_from_string(transcript)):
                if session.is_interrupted:
                    return
                await websocket.send_bytes(chunk)
            return  # greeting is not saved to conversation history

        # ── Regular turn: instant ack → LLM → TTS ─────────────
        ack = random.choice(ACKNOWLEDGMENTS)
        async for chunk in tts.synthesize(async_gen_from_string(ack)):
            if session.is_interrupted:
                return
            await websocket.send_bytes(chunk)

        if session.is_interrupted:
            return

        logger.info("Pipeline started: %s", transcript[:60])

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
            await websocket.send_bytes(audio_chunk)

        final_text = "".join(full_response)
        if final_text.strip():
            session.add_turn("assistant", final_text)

    except asyncio.CancelledError:
        raise  # propagate so cancel_tasks() can await cleanly
    except Exception as exc:
        logger.error("Pipeline error for session %s: %s", session.session_id, exc)
    finally:
        session.is_agent_speaking = False
        session.is_greeting = False
        session.last_vocal_start = None
        session.is_interrupted = False


# ── WebSocket Endpoint ───────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    """Main voice agent WebSocket endpoint — dual-loop architecture."""
    await websocket.accept()

    session_id = str(uuid4())
    session = session_manager.create_session(session_id)
    increment_connections()

    vad = VADProcessor()
    stt = AzureSTTService()
    llm = AzureLLMService()
    tts = CartesiaTTSService()

    try:
        await websocket.send_json({"type": "ready"})
        logger.info("Session %s connected and ready", session_id)

        await stt.start_stream()

        # Kick off the scripted greeting immediately (runs concurrently with audio_loop)
        session.tts_task = asyncio.create_task(
            run_pipeline(
                GREETING_TEXT, session, websocket, llm, tts, stt, is_greeting=True
            )
        )

        # ── audio_loop ───────────────────────────────────────
        # Reads raw PCM16 from the WebSocket, feeds VAD, sends speech to Azure STT.
        # Detects barge-in and cancels the running pipeline on confirmed interrupts.

        async def audio_loop() -> None:
            audio_buffer = bytearray()
            last_silence_send = 0.0  # monotonic time of last keep-alive silence frame

            async for message in websocket.iter_bytes():
                audio_buffer.extend(message)

                # Cap buffer — drop oldest audio if client sends faster than we process
                max_buf = settings.MAX_AUDIO_BUFFER_BYTES
                if len(audio_buffer) > max_buf:
                    audio_buffer = audio_buffer[-max_buf:]

                while len(audio_buffer) >= settings.FRAME_SIZE:
                    frame = bytes(audio_buffer[: settings.FRAME_SIZE])
                    audio_buffer = audio_buffer[settings.FRAME_SIZE :]

                    # ── Barge-in detection ───────────────────
                    if session.is_agent_speaking:
                        vocal_start = session.last_vocal_start or asyncio.get_event_loop().time()
                        elapsed = asyncio.get_event_loop().time() - vocal_start

                        if elapsed >= settings.INTERRUPT_COOLDOWN_S:
                            if vad.is_sustained_speech(frame):
                                logger.info("Barge-in detected for session %s", session_id)
                                await handle_interrupt(session, websocket)
                                try:
                                    await websocket.send_json({"type": "interrupt"})
                                except Exception:
                                    pass
                                stt.clear()
                                vad.reset()
                                # Prime STT with the frame that triggered the interrupt
                                await stt.send_audio(frame)
                                continue

                    # ── VAD → STT ────────────────────────────
                    result = vad.process_frame(frame)

                    if not session.is_agent_speaking:
                        # Always push to Azure STT when agent is silent:
                        # - Speech frames: carry actual voice data
                        # - Silence/end_of_speech frames: continuous silence is required
                        #   for Azure's internal VAD to detect utterance end and fire
                        #   recognized_cb. A sparse 0.5s keepalive is not enough.
                        if result == "speech":
                            await stt.send_audio(frame)
                        else:
                            await stt.send_audio(b"\x00" * settings.FRAME_SIZE)
                    else:
                        # Agent is speaking — send minimal keepalive to hold the Azure
                        # STT connection open without flooding it with silence.
                        now = asyncio.get_event_loop().time()
                        if now - last_silence_send >= 0.5:
                            await stt.send_audio(b"\x00" * settings.FRAME_SIZE)
                            last_silence_send = now

        # ── transcript_loop ──────────────────────────────────
        # Awaits Azure STT transcripts (blocking queue.get, zero CPU polling).
        # Gates on is_agent_speaking so transcripts during agent speech are discarded.
        # Runs one pipeline at a time; awaits completion (or interrupt/cancel) before next.

        async def transcript_loop() -> None:
            try:
                while True:
                    # Block until a transcript arrives or 30s timeout
                    transcript = await stt.get_transcript(
                        timeout=5.0,
                        max_age=settings.STT_TRANSCRIPT_MAX_AGE_S,
                    )

                    if not transcript:
                        logger.debug("STT timeout — still listening")
                        continue

                    logger.info("STT received: '%s'", transcript)

                    # Gate: discard if agent is currently speaking
                    if session.is_agent_speaking:
                        logger.info(
                            "Discarding transcript (agent speaking): %.40s", transcript
                        )
                        continue

                    if len(transcript.strip()) < 2:
                        continue

                    logger.info("📝 You: %s", transcript)
                    try:
                        await websocket.send_json({"type": "transcript", "text": transcript})
                    except Exception:
                        return  # WebSocket closed

                    session.add_turn("user", transcript)
                    stt.clear()

                    logger.info("🤖 AI is responding...")
                    session.tts_task = asyncio.create_task(
                        run_pipeline(transcript, session, websocket, llm, tts, stt)
                    )
                    try:
                        await session.tts_task
                    except asyncio.CancelledError:
                        pass  # interrupted — go back to listening

            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception("Transcript loop error for session %s: %s", session_id, exc)

        # Run transcript_loop as a background task; foreground runs audio_loop.
        # When audio_loop exits (WebSocket closed), we cancel transcript_loop.
        transcript_task = asyncio.create_task(transcript_loop())
        try:
            await audio_loop()
        except WebSocketDisconnect:
            logger.info("Session %s disconnected", session_id)
        except Exception as exc:
            logger.exception("Audio loop crashed for session %s: %s", session_id, exc)
        finally:
            transcript_task.cancel()
            await asyncio.gather(transcript_task, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("Session %s disconnected at setup", session_id)
    except Exception as exc:
        logger.exception("Session %s crashed: %s", session_id, exc)
    finally:
        await session.cancel_tasks()
        await stt.close()
        await tts.close()
        session_manager.delete_session(session_id)
        logger.info("Session %s cleaned up", session_id)
