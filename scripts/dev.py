"""Start the API and the dashboard on the first free ports.

Defaults are API 8000 and dashboard 5173. If either port is already taken,
the next free port is used and the dashboard proxy follows the API.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _open(port: int, host: str) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def free_port(start: int) -> int:
    for port in range(start, start + 30):
        if _open(port, "127.0.0.1") or _open(port, "::1"):
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise SystemExit(f"No free port from {start} to {start + 29}.")


def main() -> None:
    api_port = free_port(8000)
    web_port = free_port(5173)
    env = os.environ.copy()
    env["LUNARMIND_API_PORT"] = str(api_port)
    env["LUNARMIND_WEB_PORT"] = str(web_port)
    npm = "npm.cmd" if os.name == "nt" else "npm"
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.app:app", "--host", "127.0.0.1", "--port", str(api_port)],
        cwd=ROOT,
        env=env,
    )
    web = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port), "--strictPort"],
        cwd=ROOT / "frontend",
        env=env,
    )
    print(f"API http://127.0.0.1:{api_port}", flush=True)
    print(f"Dashboard http://127.0.0.1:{web_port}", flush=True)
    try:
        while True:
            if api.poll() is not None:
                raise SystemExit(api.returncode or 1)
            if web.poll() is not None:
                raise SystemExit(web.returncode or 1)
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (web, api):
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()
