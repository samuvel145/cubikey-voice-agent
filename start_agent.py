import asyncio
import os
import subprocess
import sys
import time
import httpx

async def server_already_up(url="http://127.0.0.1:8000/health") -> bool:
    """True if something healthy is already listening (avoid duplicate bind on same port)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        return False

def start_server():
    """Start the FastAPI server in the background."""
    print("----------------------------------------")
    print("🚀 STEP 1: Starting Backend Server...")
    print("----------------------------------------")
    # Using the same interpreter to ensure venv consistency
    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "run.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    return server_process

async def wait_for_server(url="http://127.0.0.1:8000/health", timeout=10):
    """Wait for the server to respond with a healthy status."""
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    print("✅ Server is online!")
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False

def run_client():
    """Run the voice client logic."""
    print("🎤 Initializing Voice Interface...")
    python_exe = sys.executable
    # Running terminal_client.py as a separate process or just importing its main
    # For a "single terminal" feel, we'll just execute it and let it take over the TTY
    subprocess.run([python_exe, "terminal_client.py"])

if __name__ == "__main__":
    server = None
    try:
        # 1. Start server only if nothing is already listening on :8000
        if asyncio.run(server_already_up()):
            print("✅ Backend already running — skipping duplicate server start.")
        else:
            server = start_server()
            if not asyncio.run(wait_for_server()):
                print("❌ Server failed to start within timeout.")
                sys.exit(1)
        
        # 3. Start Client (this is blocking)
        run_client()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        if server:
            print("🛑 Stopping server...")
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
            print("✨ Done.")
