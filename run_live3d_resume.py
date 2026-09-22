#!/usr/bin/env python3
"""
Resume supervisor: restarts a sim from a checkpoint as a NEW sim (self-fork),
so the backend reloads personas from the checkpointed memory state.

Usage:
    python run_live3d_resume.py <old_sim> <new_sim> <start_step> <total_target_step>

The backend receives "run N" where N = total_target_step - start_step
(run is RELATIVE: number of steps to advance).
The driver starts at start_step (movement files for it already exist).

The frontend (Django) is untouched: it reads whichever sim the browser
follows via curr_sim_code.json; update that if the name changes.
"""
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


def main():
    old_sim, new_sim = sys.argv[1], sys.argv[2]
    start_step, total_target = int(sys.argv[3]), int(sys.argv[4])
    remaining = total_target - start_step

    log_path = os.path.join(BASE, f"logs/{new_sim}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"=== resume: {old_sim} (step {start_step}) -> {new_sim} "
        f"(+{remaining} steps) ===")

    # -- backend: self-fork from old_sim; run is relative --------------------
    inputs = f"{old_sim}\n{new_sim}\nrun {remaining}\n"
    backend_log = open(f"/tmp/{new_sim}_backend.log", "a")
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
    say(f"backend pid={rev.pid} (log /tmp/{new_sim}_backend.log)")

    # -- driver: start at checkpoint step ------------------------------------
    mov_dir = os.path.join(STORAGE, new_sim, "movement")
    waited = 0
    while not os.path.isdir(mov_dir) and waited < 600:
        time.sleep(2)
        waited += 2
    say(f"movement dir ready after {waited}s")

    def start_driver():
        dl = open(f"/tmp/{new_sim}_driver.log", "a")
        p = subprocess.Popen(
            [PYTHON, "headless_driver.py", new_sim, str(start_step)],
            cwd=BASE,
            stdout=dl,
            stderr=subprocess.STDOUT,
            text=True,
        )
        say(f"driver pid={p.pid} (from step {start_step})")
        return p

    driver = start_driver()
    last_cur, last_progress = -1, time.time()
    stall_reported = False

    while True:
        time.sleep(30)
        steps = movement_files(new_sim)
        cur = steps[-1] if steps else -1

        if cur >= 0 and cur != last_cur:
            say(f"progress: step {cur}/{total_target} "
                f"({(cur - start_step) * 100 / remaining:.1f}% of resume)")
            last_cur = cur
            last_progress = time.time()
            stall_reported = False
        elif cur >= 0 and time.time() - last_progress > 1800 and not stall_reported:
            say(f"WARN: no new movement files for 30min (stuck at {cur})")
            stall_reported = True

        if cur >= total_target - 2:
            say("target reached — wrapping up")
            rev.terminate()
            time.sleep(2)
            driver.terminate()
            break

        if rev.poll() is not None:
            say(f"backend exited (code {rev.returncode}) at step {cur}")
            if cur >= total_target - 2:
                say("RUN COMPLETE")
                driver.terminate()
                break
            say("FATAL: backend died before target")
            driver.terminate()
            sys.exit(1)

        if driver.poll() is not None:
            say(f"driver died (code {driver.returncode}) — restarting")
            driver = start_driver()

    say("=== supervisor end ===")


if __name__ == "__main__":
    main()