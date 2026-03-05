# 🎙️ Real-Time AI Voice Agent Backend

## Overview

A **Python/FastAPI WebSocket backend** that enables real-time voice conversations with an AI assistant. The system captures spoken audio, transcribes it, generates intelligent responses, and synthesizes natural speech — all with **≤ 700ms** end-to-end latency.

## Architecture

```
Microphone → WebSocket → VAD → STT (Deepgram) → LLM (Groq) → TTS (Cartesia) → WebSocket → Speaker
```

```
┌─────────────┐          ┌──────────────────────────────────┐          ┌─────────────┐
│   Client    │─── WS ──▶│  FastAPI WebSocket Server         │─── WS ──▶│   Client    │
│  (Audio In) │          │  VAD → STT → LLM → TTS           │          │ (Audio Out) │
└─────────────┘          └──────────────────────────────────┘          └─────────────┘
```

## Tech Stack

| Component       | Technology     | Model / Version          |
|-----------------|----------------|--------------------------|
| **Backend**     | FastAPI        | Async Python (3.11+)     |
| **Transport**   | WebSocket      | Full-duplex streaming    |
| **VAD**         | WebRTC VAD     | Aggressiveness level 2   |
| **STT**         | Deepgram       | nova-2                   |
| **LLM**         | Groq           | llama3-70b-8192          |
| **TTS**         | Cartesia       | sonic-2                  |

## Prerequisites

- Python 3.11+
- API keys for:
  - [Deepgram](https://deepgram.com) (Speech-to-Text)
  - [Groq](https://console.groq.com) (LLM)
  - [Cartesia](https://cartesia.ai) (Text-to-Speech)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/samuvel145/AI-Voice-agent.git
cd AI-Voice-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
# Create a .env file based on the Configuration section below
```

### Running the Agent

The easiest way to start both the server and the terminal client is using the provided startup script:

```bash
python start_agent.py
```

This will automatically:
1. Start the FastAPI backend server.
2. Wait for the server to be healthy.
3. Launch the terminal-based voice client.

---

## Alternative: Manual Start

If you prefer to run components separately:

**1. Start the Server:**
```bash
python run.py
```

**2. Start the Terminal Client (in a new terminal):**
```bash
python terminal_client.py
```

---

## Configuration (`.env`)

Create a `.env` file in the root directory with the following variables:

| Variable              | Description                      | Default           |
|-----------------------|----------------------------------|--------------------|
| `DEEPGRAM_API_KEY`    | Deepgram API key                 | *(required)*       |
| `GROQ_API_KEY`        | Groq API key                     | *(required)*       |
| `CARTESIA_API_KEY`    | Cartesia API key                 | *(required)*       |
| `CARTESIA_VOICE_ID`   | Cartesia voice identifier        | `default`          |
| `HOST`                | Server bind address              | `127.0.0.1`        |
| `PORT`                | Server port                      | `8000`             |
| `VAD_AGGRESSIVENESS`  | VAD sensitivity (0–3)            | `2`                |
| `SILENCE_THRESHOLD_MS`| Silence before end-of-speech     | `800`              |
| `MAX_HISTORY_TURNS`   | Conversation memory depth        | `5`                |

## Project Structure

```
voice-agent/
├── app/
│   ├── main.py                  # FastAPI app & health check
│   └── websocket_handler.py     # Pipeline orchestration
├── services/
│   ├── stt_deepgram.py          # STT streaming
│   ├── llm_groq.py              # LLM streaming
│   └── tts_cartesia.py          # TTS streaming
├── audio/
│   ├── vad.py                   # VAD logic
│   └── audio_buffer.py          # Frame buffering
├── session/
│   └── session_manager.py       # State & History
├── config.py                    # Environment settings
├── run.py                       # Server entry point
├── terminal_client.py           # Voice-enabled terminal client
├── start_agent.py               # Combined startup script
├── requirements.txt             # Dependencies
└── .env                         # Your local secrets
```

