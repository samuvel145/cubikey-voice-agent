import asyncio
import json
import logging
import queue
import sys

import numpy as np
import sounddevice as sd
import websockets

# Configuration
WS_URL = "ws://127.0.0.1:8000/ws/voice"
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

            async def mic_sender():
                """Reads from mic and sends to server."""
                input_queue = asyncio.Queue()
                loop = asyncio.get_event_loop()
                def callback(indata, frames, time, status):
                    loop.call_soon_threadsafe(input_queue.put_nowait, indata.copy())

                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback):
                    print("🎙️  [Microphone Active] - Talk to the AI...")
                    while True:
                        data = await input_queue.get()
                        await ws.send(data.tobytes())

            async def message_receiver():
                """Receives and processes server messages."""
                try:
                    async for message in ws:
                        if isinstance(message, bytes):
                            audio_chunk = np.frombuffer(message, dtype='int16')
                            audio_out_queue.put(audio_chunk)
                        else:
                            data = json.loads(message)
                            mtype = data.get("type")
                            
                            if mtype == "ready":
                                print("\n🤖 Agent: [Ready!] ")
                            elif mtype == "transcript":
                                print(f"\n📝 You: {data.get('text')}")
                            elif mtype == "interrupt":
                                nonlocal pending_audio
                                print("\n⚡ [Interrupted]                             ", end="\r")
                                # 1. Flush the queue
                                while not audio_out_queue.empty():
                                    try: audio_out_queue.get_nowait()
                                    except queue.Empty: break
                                # 2. Clear the hardware-level pending buffer
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
