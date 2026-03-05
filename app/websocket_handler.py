"""
WebSocket Handler — Pipeline orchestration for the voice agent.
Accepts WebSocket connections at /ws/voice, processes audio through
VAD → STT → LLM → TTS, and streams audio responses back to the client.
"""

import asyncio
import logging
from uuid import uuid4
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from audio.vad import VADProcessor
from services.stt_deepgram import DeepgramSTTService
from services.llm_groq import GroqLLMService
from services.tts_cartesia import CartesiaTTSService
from session.session_manager import session_manager
from app.main import increment_connections

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────

async def async_gen_from_string(text: str) -> AsyncGenerator[str, None]:
    """Wrap a plain string as a single-yield async generator."""
    yield text


async def handle_interrupt(session, websocket: WebSocket) -> None:
    """Handle user interruption while the agent is speaking."""
    session.is_interrupted = True
    session.is_agent_speaking = False
    await session.cancel_tasks()
    session.reset_speaking_state()
    logger.info("Interruption sequence completed for session: %s", session.session_id)


async def run_pipeline(
    transcript: str,
    session,
    websocket: WebSocket,
    llm: GroqLLMService,
    tts: CartesiaTTSService,
    is_greeting: bool = False,
) -> None:
    """Run the LLM → TTS pipeline and stream audio to the client."""
    try:
        session.is_agent_speaking = True
        session.is_greeting = is_greeting
        
        if is_greeting:
            logger.info("Sending direct greeting: %s", transcript)
            # Synthesize direct greeting
            first_chunk = True
            async for audio_chunk in tts.synthesize(async_gen_from_string(transcript)):
                if first_chunk:
                    # Update the official 'start' time to now, so interruption logic can see it
                    session.last_vocal_start = asyncio.get_event_loop().time()
                    first_chunk = False

                if session.is_interrupted:
                    break
                await websocket.send_bytes(audio_chunk)
        else:
            logger.info("Pipeline run started for transcript: %s", transcript)
            
            # Wrap LLM generator to capture full text for history
            full_response = []
            async def token_wrapper():
                async for token in llm.generate(transcript, session.history):
                    if session.is_interrupted:
                        break
                    full_response.append(token)
                    yield token

            # Feed LLM stream directly into TTS stream
            first_chunk = True
            async for audio_chunk in tts.synthesize(token_wrapper()):
                if first_chunk:
                    session.last_vocal_start = asyncio.get_event_loop().time()
                    first_chunk = False

                if session.is_interrupted:
                    break
                await websocket.send_bytes(audio_chunk)

            # Record turn in history
            final_text = "".join(full_response)
            if final_text.strip():
                session.add_turn("assistant", final_text)

        # Wait a moment after sending chunks to allow client buffer to play.
        # This keeps the interruption handler active while the user still hears the agent.
        if not session.is_interrupted:
            await asyncio.sleep(4.0 if is_greeting else 1.2)

    except Exception as exc:
        logger.error("Pipeline error for session %s: %s", session.session_id, exc)
    
    finally:
        session.is_agent_speaking = False
        session.is_greeting = False
        session.last_vocal_start = None

# ── WebSocket Endpoint ───────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    """Main voice agent WebSocket endpoint."""
    await websocket.accept()

    session_id = str(uuid4())
    session = session_manager.create_session(session_id)
    increment_connections()

    vad = VADProcessor()
    stt = DeepgramSTTService()
    llm = GroqLLMService()
    tts = CartesiaTTSService()

    try:
        # Signal readiness
        await websocket.send_json({"type": "ready"})
        logger.info("Session %s connected and ready", session_id)

        # ── Send initial greeting ────────────────────────────
        await asyncio.sleep(0.5)
        logger.info("Launching initial greeting for session %s", session_id)
        session.tts_task = asyncio.create_task(
            run_pipeline(
                "Hello! Please introduce yourself briefly and ask how you can help me today.",
                session, websocket, llm, tts, is_greeting=True
            )
        )

        await stt.start_stream()

        # ── Main audio loop ──────────────────────────────────
        audio_buffer = bytearray()
        
        async for message in websocket.iter_bytes():
            audio_buffer.extend(message)
            
            while len(audio_buffer) >= settings.FRAME_SIZE:
                frame = bytes(audio_buffer[:settings.FRAME_SIZE])
                audio_buffer = audio_buffer[settings.FRAME_SIZE:]

                # ── Interruption check ───────────────────────────
                # cooldown: 0.0s everywhere for maximum responsiveness
                vocal_start = session.last_vocal_start or (asyncio.get_event_loop().time())
                time_since_speech = asyncio.get_event_loop().time() - vocal_start
                
                if session.is_agent_speaking and (time_since_speech >= 0.0):
                    if vad.is_speech(frame):
                        logger.info("⚡ Interruption detected during %s", "greeting" if session.is_greeting else "response")
                        # 1. STOP SERVER-SIDE TASKS IMMEDIATELY
                        await handle_interrupt(session, websocket)
                        # 2. CLEAR CLIENT-SIDE AUDIO
                        try:
                            await websocket.send_json({"type": "interrupt"})
                        except: pass
                        # 3. PURGE STT
                        stt.clear()
                        # Exit early to prevent processing the trigger word as voice command
                        break 

                # ── VAD processing ───────────────────────────────
                result = vad.process_frame(frame)

                if result == "speech":
                    # If we are in the middle of thinking/speaking, ignore speech frames for STT
                    # we only send audio when the agent is silent
                    if not session.is_agent_speaking:
                        await stt.send_audio(frame)
                else:
                    if asyncio.get_event_loop().time() % 1.0 < 0.1:
                        await stt.send_audio(b"\x00" * 640)

                if result == "end_of_speech" and not session.is_agent_speaking:
                    transcript = await stt.get_transcript(timeout=2.0)
                    if not transcript or len(transcript.strip()) < 2:
                        continue

                    audio_buffer = bytearray()
                    logger.info("📝 You: %s", transcript)
                    await websocket.send_json({"type": "transcript", "text": transcript})
                    session.add_turn("user", transcript)
                    stt.clear()

                    logger.info("🤖 AI is responding...")
                    session.tts_task = asyncio.create_task(
                        run_pipeline(transcript, session, websocket, llm, tts)
                    )
                    break # Exit the frame-processing loop to wait for response synthesis

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception as exc:
        logger.exception("Session %s crashed: %s", session_id, exc)
    finally:
        await session.cancel_tasks()
        await stt.close()
        session_manager.delete_session(session_id)
        logger.info("Session %s cleaned up", session_id)
