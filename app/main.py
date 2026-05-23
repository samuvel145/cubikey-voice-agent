"""
FastAPI application factory — startup/shutdown events, health check,
CORS middleware, and WebSocket router inclusion.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from session.session_manager import session_manager
from app.metrics import increment_connections, get_total_connections  # noqa: F401 (re-exported)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── App lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─ Startup ─
    logger.info("Voice Agent server starting...")

    missing = []
    if not settings.AZURE_SPEECH_KEY:
        missing.append("AZURE_SPEECH_KEY")
    if not settings.AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.CARTESIA_API_KEY:
        missing.append("CARTESIA_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing required API keys: {', '.join(missing)}. "
            "Please set them in your .env file."
        )

    logger.info("All API keys validated. Server ready.")
    yield
    # ─ Shutdown ─
    logger.info("Voice Agent server shutting down...")


# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Voice Agent",
    version="1.0.0",
    description="Real-time AI voice assistant backend",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ──────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": session_manager.active_count(),
        "total_connections": get_total_connections(),
    }


# ── Include WebSocket router ────────────────────────────────
from app.websocket_handler import router  # noqa: E402
from fastapi.responses import HTMLResponse

app.include_router(router)

# ── Built-in Web UI ──────────────────────────────────────────
@app.get("/")
async def root_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Real-Time AI Voice Agent</title>
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            h1 { font-size: 2.5rem; background: -webkit-linear-gradient(#38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            #status { font-size: 1.2rem; margin-bottom: 2rem; color: #94a3b8; }
            button { background: #3b82f6; border: none; color: white; padding: 1rem 3rem; font-size: 1.5rem; border-radius: 9999px; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39); }
            button:hover { background: #2563eb; transform: translateY(-2px); }
            button:active { transform: translateY(0); }
            button:disabled { background: #475569; pointer-events: none; opacity: 0.5; }
            .pulse { animation: pulse 2s infinite; }
            @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(59, 130, 246, 0); } 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); } }
        </style>
    </head>
    <body>
        <h1>Voice Agent</h1>
        <div id="status">Ready to connect...</div>
        <button id="connectBtn">Connect & Start Talking</button>

        <script>
            let ws;
            let audioContext;
            let mediaStream;
            let processor;
            const btn = document.getElementById('connectBtn');
            const status = document.getElementById('status');

            btn.onclick = async () => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                    return;
                }
                
                status.innerText = "Requesting microphone access...";
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 } });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                } catch (err) {
                    status.innerText = "Microphone access denied!";
                    console.error(err);
                    return;
                }

                status.innerText = "Connecting to Voice Agent...";
                ws = new WebSocket(`ws://${window.location.host}/ws/voice`);
                ws.binaryType = "arraybuffer";

                ws.onopen = () => {
                    status.innerText = "Connected! Waiting for agent to initialize...";
                    btn.innerText = "Disconnect";
                    btn.classList.add('pulse');
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'ready') {
                            status.innerText = "Listening... Speak now!";
                            startMicrophone();
                        } else if (msg.type === 'transcript') {
                            status.innerText = `You said: "${msg.text}"`;
                        } else if (msg.type === 'interrupt') {
                            status.innerText = "Listening...";
                            // Flush any buffered agent audio still scheduled to play.
                            // Closing and recreating the AudioContext is the only reliable
                            // way to stop Web Audio API's internal playback queue immediately.
                            if (audioContext) {
                                const old = audioContext;
                                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                                nextPlayTime = 0;
                                old.close().catch(() => {});
                            }
                        }
                    } else if (event.data instanceof ArrayBuffer) {
                        status.innerText = "Agent is speaking...";
                        playAudioChunk(event.data);
                    }
                };

                ws.onclose = () => {
                    status.innerText = "Disconnected.";
                    btn.innerText = "Connect & Start Talking";
                    btn.classList.remove('pulse');
                    stopMicrophone();
                };
            };

            function startMicrophone() {
                const source = audioContext.createMediaStreamSource(mediaStream);
                // ScriptProcessor is deprecated but easiest for exact 16kHz PCM16 downsampling natively in 1 file
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!ws || ws.readyState !== WebSocket.OPEN) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcm16Buffer = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        let s = Math.max(-1, Math.min(1, inputData[i]));
                        pcm16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    ws.send(pcm16Buffer.buffer);
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            }

            let nextPlayTime = 0;
            async function playAudioChunk(arrayBuffer) {
                // Cartesia sends raw PCM16 output. We must decode to float32.
                const pcm16 = new Int16Array(arrayBuffer);
                const float32 = new Float32Array(pcm16.length);
                for (let i = 0; i < pcm16.length; i++) {
                    float32[i] = pcm16[i] / 32768.0;
                }

                const audioBuffer = audioContext.createBuffer(1, float32.length, 16000);
                audioBuffer.getChannelData(0).set(float32);
                
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                if (nextPlayTime < audioContext.currentTime) {
                    nextPlayTime = audioContext.currentTime;
                }
                source.start(nextPlayTime);
                nextPlayTime += audioBuffer.duration;
            }

            function stopMicrophone() {
                if (processor) processor.disconnect();
                if (audioContext) audioContext.close();
                if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
