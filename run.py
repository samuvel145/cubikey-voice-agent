"""
Voice Agent — Uvicorn entry point.
"""

import uvicorn
from config import settings

if __name__ == "__main__":
    print(
        f"🎙️  Voice Agent starting on "
        f"ws://{settings.HOST}:{settings.PORT}/ws/voice"
    )
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
