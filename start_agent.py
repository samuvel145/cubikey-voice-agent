import asyncio
import os
import socket
import subprocess
import sys
import time
import httpx

# Read PORT from .env so this file never hard-codes it
def _read_port() -> int:
    try:
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("PORT="):
                    return int(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return 8000

SERVER_PORT = _read_port()
HEALTH_URL = f"http://127.0.0.1:{SERVER_PORT}/health"


def free_port(port: int) -> None:
    """Kill any process occupying the given port (Windows + Linux)."""
    if os.name == "nt":
        # Windows: use netstat + taskkill
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr ":{port} "', shell=True, text=True
            )
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pid = int(parts[-1])
                    subprocess.call(
                        f"taskkill /PID {pid} /F",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except subprocess.CalledProcessError:
            pass  # nothing on that port
    else:
        # Linux/macOS: use fuser
        subprocess.call(
            f"fuser -k {port}/tcp",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


async def server_already_up(url: str = HEALTH_URL) -> bool:
    """True if something healthy is already listening (avoid duplicate bind)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False

def start_server():
    """Kill anything on the port, then start the FastAPI server."""
    print("----------------------------------------")
    print("🚀 STEP 1: Starting Backend Server...")
    print("----------------------------------------")
    free_port(SERVER_PORT)
    time.sleep(0.5)  # give OS a moment to release the socket
    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "run.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return server_process

async def wait_for_server(url: str = HEALTH_URL, timeout: int = 15):
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
        # 1. Start server only if nothing healthy is already on the port
        if asyncio.run(server_already_up()):
            print(f"✅ Backend already running on port {SERVER_PORT} — skipping duplicate start.")
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
