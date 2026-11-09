"""Thermal guard for the local AI stack (Mark LVI + Ollama).

Polls macOS's thermal pressure level every few seconds (no sudo needed):
  0 nominal  -> nothing
  1 fair     -> log a warning
  2 serious  -> STOP: quit Mark LVI, unload Ollama models, stop the Ollama service
  3 critical -> same as serious

Run:  .venv/bin/python thermal_guard.py
Restart afterwards:  brew services start ollama  &&  .venv/bin/python main.py
"""
import subprocess
import time
from datetime import datetime

from Foundation import NSProcessInfo

POLL_SECONDS = 5
STOP_AT = 2  # serious
NAMES = ["nominal", "fair", "serious", "critical"]


def now() -> str:
    return datetime.now().strftime("%I:%M:%S %p")


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # keep guarding even if one step fails
        return f"{cmd[0]} failed: {e}"


def stop_everything(state: int) -> None:
    print(f"{now()} STOP: thermal state {NAMES[state]} - shutting down the AI workload", flush=True)
    run(["pkill", "-f", ".venv/bin/python main.py"])                 # Mark LVI
    for line in run(["ollama", "ps"]).splitlines()[1:]:               # unload loaded models
        if line.split():
            run(["ollama", "stop", line.split()[0]])
    run(["brew", "services", "stop", "ollama"])                       # no new model loads
    run(["osascript", "-e",
         f'display notification "Mac is {NAMES[state]} - stopped Mark LVI and Ollama" '
         f'with title "Thermal guard"'])
    print(f"{now()} STOPPED: Mark LVI quit, models unloaded, Ollama service stopped", flush=True)


def main() -> None:
    info = NSProcessInfo.processInfo()
    last = None
    print(f"{now()} guard started (polling every {POLL_SECONDS}s, stops at {NAMES[STOP_AT]})", flush=True)
    while True:
        state = int(info.thermalState())
        if state != last:
            print(f"{now()} thermal state: {NAMES[state]}", flush=True)
            last = state
        if state >= STOP_AT:
            stop_everything(state)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
