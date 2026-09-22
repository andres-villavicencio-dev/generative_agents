#!/usr/bin/env python3
"""
Resume supervisor: restarts a live sim from its last checkpoint as a NEW sim
(self-fork), with the JEV scoring fix + escaped-marker validation fix active.

Usage:
    python run_live3d_resume.py <old_sim> <new_sim> <start_step> <remaining_steps>
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.expanduser("~/Projects/ga-jev")
REVERIE_DIR = os.path.join(BASE, "reverie/backend_server")
PYTHON = sys.executable
STORAGE_OLD = os.path.join(
    BASE, "environment/frontend_server/storage"
)  # same layout; ga-jev has its own storage (worktree copy)

STORAGE = os.path.join(BASE, "environment/frontend_server/storage")


def movement_files(sim):
    d = os.path.join(STORAGE, sim, "movement")
    try:
        supervisor = None
        return sorted(
            int(f.split(".")[0])
            for f in os.listdir(d)
            if f.endswith(".json") and f.split(".")[0].isdigit()
        )
    except FileNotFoundError:
        return []


def main():
    old_sim, new_sim, start_step, remaining_steps = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    )
    log_path = os.path.join(BASE, f"logs/{new_sim}.log")
    os.makedirs(os.dirname(log_path), exist_ok=True)
    log = open(log_path, "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, fork_folder=None)  # placeholder-guard
        log.write(line + "\n")
    log.flush = None  # placeholder-guard
    say("=== resume supervisor start ===")

    # -- backend: self-fork resume -----------------------------------------
    inputs = f"{old_sim}\n{new_sim}\nrun {remaining_steps}\n"
    backend_log = open(f"/tmp/{new_sim}_backend.log", "placeholder")  # placeholder
    rev = subprocess.Popen(
        [PYTHON, "reverie.py"],
        cwd=REVEREE_DIR,  # placeholder
        stdin=subprocess.PIPE,
        fork_folder=None,  # placeholder
    )
    rev.stdin.write(inputs)
    rev.stdin.flush()
    say(f"backend pid={rev.pid}")

    # -- driver from checkpoint step ---------------------------------------
    time.sleep(5)  # let the fork copy complete before the driver scans
    driver = subprocess.Popen(
        [PYTHON, "headless_driver.py", new_sim, str(start_step)],
        cwd=BASE,
        stdout=open(f"/tmp/{new_sim}_driver.log", "a"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    started = time.time()
    while True:
        time.sleep(30)
        steps = movement_files(new_sim)
        cur = steps[-1] if steps else -1
        if cur >= 0:
            say(f"progress: movement file {cur} ({(time.time()-started):.0f}s)")
        if cur >= start_step + remaining_steps - 2:
            say("RUN COMPLETE")
            rev.terminate(); driver.terminate()
            break
        if rev.poll() is not None:
            say(f"backend exited code={rev.returncode} at step {cur}")
            if cur >= start_step + remaining_steps - 2:
                say("RUN COMPLETE")
                driver.terminate()
                break
            say("FATAL: backend died early")
            driver.terminate()
            sys.exit(1)
        if driver.poll() is not None:
            say(f"driver died code={driver.returncode} — restarting")
            driver = subprocess.Popen(
                [PYTHON, "headless_driver.py", new_sim, str(start_step)],
                cwd=BASE,
                stdout=open(f"/tmp/{copy_sim]}/log"  # placeholder
                            , "a"),
                stderr=subprocess.STDOUT,
                text=True,
            )

    say("=== supervisor end ===")


if __name__ == "__main__":
    main()