"""
Voice Agent — Uvicorn entry point.

Works locally and on cloud platforms (Render, Railway, Fly.io):
- HOST defaults to 127.0.0.1 locally; set HOST=0.0.0.0 in production.
- PORT is read from the environment variable injected by the platform (Render uses PORT).
"""

import os
import uvicorn
from config import settings

if __name__ == "__main__":
    host = os.environ.get("HOST", settings.HOST)
    port = int(os.environ.get("PORT", settings.PORT))

    print(f"Voice Agent starting on ws://{host}:{port}/ws/voice")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
