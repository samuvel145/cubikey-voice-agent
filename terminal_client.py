import asyncio
import json
import logging
import queue
import sys

import numpy as np
import sounddevice as sd
import websockets

# Configuration — read PORT from .env so it stays in sync with run.py
def _read_port() -> int:
    try:
        with open(".env") as f:
            for line in f:
                if line.strip().startswith("PORT="):
                    return int(line.strip().split("=", 1)[1])
    except Exception:
        pass
    return 8000

WS_URL = f"ws://127.0.0.1:{_read_port()}/ws/voice"
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Size of each microphone read

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

async def main():
    """Terminal client for the AI Voice Agent."""
    print("\n🎤 AI Voice Agent Terminal Client")
    print(f"🔗 Connecting to {WS_URL}...")

    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected to backend!")

            # Low-latency playback state
            audio_out_queue = queue.Queue()
            pending_audio = []

            def sd_callback(outdata, frames, time, status):
                nonlocal pending_audio
                if status:
                    print(f"\r⚠️  {status}", end="")
                
                # Pull everything from queue into our pending list
                while not audio_out_queue.empty():
                    try:
                        pending_audio.append(audio_out_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # Flatten what we have and serve as much as possible
                if pending_audio:
                    data = np.concatenate(pending_audio)
                    if len(data) >= frames:
                        outdata[:] = data[:frames].reshape(-1, 1)
                        pending_audio = [data[frames:]]
                    else:
                        outdata[:len(data)] = data.reshape(-1, 1)
                        outdata[len(data):] = 0
                        pending_audio = []
                else:
                    outdata.fill(0)

            # High-priority output stream
            out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=512, # 32ms chunks = instant stop
                callback=sd_callback
            )
            out_stream.start()

            # ── Mic-mute state (shared between mic_sender & message_receiver) ──
            # agent_speaking:  True while server is sending TTS audio
            # mute_until:      monotonic time until which mic sends silence
            # playback_est_end: estimated time when client will finish playing queued audio
            #
            # WHY playback_est_end matters:
            # agent_stop is sent by the server when it finishes SENDING audio chunks,
            # not when the CLIENT finishes PLAYING them. The playback queue may hold
            # several seconds of audio still to be played. We must keep the mic muted
            # until that audio has actually played out, then add a short echo-settle.
            mic_state = {"agent_speaking": False, "mute_until": 0.0}
            playback_est_end = [0.0]  # list so inner closures can mutate it
            ECHO_SETTLE_S = 0.8

            async def mic_sender():
                """Reads from mic and sends to server.

                While the agent is speaking (agent_speaking=True) we still send
                real audio so the server can detect barge-in via VAD.  For the
                brief echo-settle window after the agent stops (mute_until) we
                send silence so the server STT doesn't transcribe speaker echo.
                """
                input_queue = asyncio.Queue()
                loop = asyncio.get_event_loop()
                def callback(indata, frames, time, status):
                    loop.call_soon_threadsafe(input_queue.put_nowait, indata.copy())

                silence_frame = np.zeros((CHUNK_SIZE, CHANNELS), dtype='int16')

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback):
                    print("🎙️  [Microphone Active] - Talk to the AI...")
                    while True:
                        data = await input_queue.get()
                        now = asyncio.get_event_loop().time()

                        if mic_state["agent_speaking"]:
                            # Terminal mode: no echo cancellation — send silence while
                            # agent TTS plays so the speaker echo cannot trigger false
                            # barge-in.  Barge-in works correctly on Twilio (phone AEC)
                            # and the browser client (WebRTC AEC).
                            s = np.zeros(len(data), dtype='int16')
                            await ws.send(s.tobytes())
                        elif now < mic_state["mute_until"]:
                            # Echo-settle window: send silence so server STT doesn't
                            # transcribe the dying speaker echo.
                            s = np.zeros(len(data), dtype='int16')
                            await ws.send(s.tobytes())
                        else:
                            await ws.send(data.tobytes())

            async def message_receiver():
                """Receives and processes server messages."""
                try:
                    async for message in ws:
                        if isinstance(message, bytes):
                            audio_chunk = np.frombuffer(message, dtype='int16')
                            audio_out_queue.put(audio_chunk)
                            # Track when this audio will finish playing so we can
                            # keep the mic muted until the speakers actually go quiet.
                            now = asyncio.get_event_loop().time()
                            duration = len(audio_chunk) / SAMPLE_RATE
                            if now >= playback_est_end[0]:
                                playback_est_end[0] = now + duration
                            else:
                                playback_est_end[0] += duration
                        else:
                            data = json.loads(message)
                            mtype = data.get("type")

                            if mtype == "ready":
                                print("\n🤖 Agent: [Ready!] ")

                            elif mtype == "agent_start":
                                # Agent starting — reset playback tracker for this turn
                                mic_state["agent_speaking"] = True
                                playback_est_end[0] = 0.0

                            elif mtype == "agent_stop":
                                # Agent done SENDING. Keep mic muted until:
                                #   • the playback buffer has finished playing  (playback_est_end)
                                #   • plus a short room-echo settle window      (ECHO_SETTLE_S)
                                mic_state["agent_speaking"] = False
                                settle_target = max(
                                    playback_est_end[0],
                                    asyncio.get_event_loop().time(),
                                ) + ECHO_SETTLE_S
                                mic_state["mute_until"] = settle_target
                                playback_est_end[0] = 0.0

                            elif mtype == "transcript":
                                print(f"\n📝 You: {data.get('text')}")

                            elif mtype == "interrupt":
                                nonlocal pending_audio
                                # Barge-in: playback stops immediately, short settle
                                mic_state["agent_speaking"] = False
                                playback_est_end[0] = 0.0
                                mic_state["mute_until"] = (
                                    asyncio.get_event_loop().time() + ECHO_SETTLE_S
                                )
                                print("\n⚡ [Interrupted]                             ", end="\r")
                                while not audio_out_queue.empty():
                                    try:
                                        audio_out_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                pending_audio.clear()
                                print("\n🎙️  Agent: [Stopped. Listening to you...]", end="\r")

                            elif mtype == "error":
                                print(f"\n❌ Server Error: {data.get('message')}")
                except Exception:
                    pass

            # Run sender and receiver concurrently
            try:
                await asyncio.gather(mic_sender(), message_receiver())
            except Exception as e:
                # If we are exiting due to Ctrl+C, it's handled in the main wrapper
                if not isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
                    print(f"\n⚠️  Stream error: {e}")
            finally:
                out_stream.stop()
                out_stream.close()

    except Exception as e:
        print(f"\n❌ FAILED to connect: {e}")
        print("Make sure the server is running with 'python run.py' first.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
