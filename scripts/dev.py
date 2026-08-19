"""
Local dev launcher: starts N workers + the Rust router, wires them together,
and tears everything down on Ctrl+C. Cross-platform.

Usage:
    python scripts/dev.py            # 2 workers + router
    python scripts/dev.py --workers 4
    python scripts/dev.py --no-router-build   # skip `cargo build`

Assumes a venv at .venv and that `cargo` is on PATH (or at ~/.cargo/bin).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
CARGO = os.environ.get("CARGO", str(Path.home() / ".cargo" / "bin" / ("cargo.exe" if IS_WIN else "cargo")))
if not Path(CARGO).exists():
    CARGO = "cargo"  # fall back to PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-port", type=int, default=9001)
    ap.add_argument("--router-port", type=int, default=8000)
    ap.add_argument("--no-router-build", action="store_true")
    args = ap.parse_args()

    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    procs: list[subprocess.Popen] = []

    def cleanup(*_):
        print("\n[dev] shutting down...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    # 1) start workers
    worker_specs = []
    for i in range(args.workers):
        port = args.base_port + i
        name = f"worker-{i}"
        print(f"[dev] starting {name} on :{port}")
        p = subprocess.Popen(
            [py, str(ROOT / "workers" / "worker.py"), "--port", str(port), "--name", name],
            cwd=str(ROOT / "workers"),
        )
        procs.append(p)
        worker_specs.append(f"{name}=http://127.0.0.1:{port}")

    # 2) build + start router
    if not args.no_router_build:
        print("[dev] building router (cargo build)...")
        subprocess.run([CARGO, "build"], cwd=str(ROOT / "router"), check=True)

    env = dict(os.environ)
    env["MD_WORKERS"] = ",".join(worker_specs)
    env["MD_LISTEN"] = f"0.0.0.0:{args.router_port}"
    env.setdefault("RUST_LOG", "info")

    time.sleep(1.5)  # give workers a moment to bind
    print(f"[dev] starting router on :{args.router_port} -> {env['MD_WORKERS']}")
    router = subprocess.Popen([CARGO, "run", "--quiet"], cwd=str(ROOT / "router"), env=env)
    procs.append(router)

    router.wait()
    cleanup()


if __name__ == "__main__":
    main()
