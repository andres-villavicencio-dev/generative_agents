#!/usr/bin/env python3
"""
Supervisor for a long-running generative-agents simulation.

Launches the reverie backend (25,920 steps = 3 sim-days) and the headless
driver, keeps them alive for the whole run, logs progress, and restarts the
driver if it dies. The backend is NOT auto-restarted once steps have landed
(restarting it mid-run loses the in-memory persona state); a dead backend
with zero movement files means the run failed and the supervisor exits.

Usage:
    python run_live3d.py <sim_name> <total_steps>
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.expanduser("~/Projects/generative_agents")
REVERIE_DIR = os.path.join(BASE, "reverie/backend_server")
PYTHON = sys.executable
STORAGE = os.path.join(BASE, "environment/frontend_server/storage")


def movement_files(sim):
    d = os.path.join(STORAGE, sim, "movement")
    try:
        return sorted(
            int(f.split(".")[0])
            for f in os.listdir(d)
            if f.endswith(".json") and f.split(".")[0].isdigit()
        )
    except FileNotFoundError:
        return []


def alive(proc):
    return proc is not None and proc.poll() is None


def main():
    sim = sys.argv[1]
    total_steps = int(sys.argv[2])
    fork = sys.argv[3] if len(sys.argv) > 3 else "base_the_ville_isabella_maria_klaus"
    log_path = os.path.join(BASE, f"logs/{sim}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"=== supervisor start: sim={sim} total_steps={total_steps} fork={fork} ===")

    # -- backend ------------------------------------------------------------
    inputs = f"{fork}\n{sim}\nrun {total_steps}\n"
    backend_log = open(f"/tmp/{sim}_backend.log", "w")
    rev = subprocess.Popen(
        [PYTHON, "reverie.py"],
        cwd=REVERIE_DIR,
        stdin=subprocess.PIPE,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    rev.stdin.write(inputs)
    rev.stdin.flush()
    say(f"backend pid={rev.pid} (log: /tmp/{sim}_backend.log)")

    # -- wait for sim dirs, then driver ------------------------------------
    mov_dir = os.path.join(STORAGE, sim, "movement")
    waited = 0
    while not os.path.isdir(mov_dir) and waited < 300:
        time.sleep(2)
        waited += 2
    if not os.path.isdir(mov_dir):
        say("FATAL: movement dir never appeared — backend failed to fork sim")
        rev.terminate()
        sys.exit(1)
    say(f"movement dir ready after {waited}s")

    driver = None
    last_step = -1
    last_progress = time.time()
    stall_reported = False

    def start_driver():
        dl = open(f"/tmp/{sim}_driver.log", "a")
        p = subprocess.Popen(
            [PYTHON, "headless_driver.py", sim, "0"],
            cwd=BASE,
            stdout=dl,
            stderr=subprocess.STDOUT,
            text=True,
        )
        say(f"driver pid={p.pid} (log: /tmp/{sim}_driver.log)")
        return p

    driver = start_driver()

    # -- keep-alive loop ----------------------------------------------------
    while True:
        time.sleep(30)
        steps = movement_files(sim)
        cur = steps[-1] if steps else -1

        if cur >= 0 and cur != last_step:
            say(f"progress: step {cur}/{total_steps} "
                f"({cur * 100 / total_steps:.1f}%)")
            last_step = cur
            last_progress = time.time()
            stall_reported = False
        elif cur >= 0:
            if time.time() - last_progress > 1800 and not stall_reported:
                say(f"WARN: no new movement files for 30min (stuck at {cur})")
                stall_reported = False

        # completion check FIRST: backend exits cleanly at the target step
        if not alive(rev):
            say(f"backend exited (code {rev.returncode}) at step {cur}")
            if cur >= total_steps - 2:
                say("RUN COMPLETE")
                driver.terminate()
                break
            say("FATAL: backend died before target — aborting")
            driver.terminate()
            sys.exit(1)

        if not alive(driver):
            say(f"driver died (code {driver.returncode}) at step {cur} — restarting")
            driver = start_driver()

        if cur >= total_steps - 1:
            say("target step reached — wrapping up")
            rev.terminate()
            time.sleep(2)
            driver.terminate()
            break

    say("=== supervisor end ===")


if __name__ == "__main__":
    main()