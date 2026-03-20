#!/usr/bin/env python3
"""
Headless environment driver for Reverie simulation.

Mimics the browser frontend: reads movement/{step}.json from the backend,
extracts agent positions, then writes environment/{step+1}.json so the
backend can proceed to the next step. Fully autonomous — no browser needed.

Usage:
    python headless_driver.py <sim_code> [start_step]
"""

import json
import os
import sys
import time

STORAGE = os.path.join(
    os.path.dirname(__file__),
    "environment/frontend_server/storage"
)
MAZE_NAME = "the_ville"
SLEEP_INTERVAL = 0.5   # seconds between polls
MAX_WAIT = 120          # seconds to wait for a movement file before giving up


def read_movement(sim_code, step):
    path = os.path.join(STORAGE, sim_code, "movement", f"{step}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            content = f.read()
        if not content.strip():
            return None  # file exists but not written yet — race condition
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return None  # partial write — caller will retry


def write_environment(sim_code, step, personas):
    """Write environment/{step}.json from a dict of {name: [x, y]}."""
    env = {
        name: {"maze": MAZE_NAME, "x": pos[0], "y": pos[1]}
        for name, pos in personas.items()
    }
    path = os.path.join(STORAGE, sim_code, "environment", f"{step}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(env, f, indent=2)


def get_persona_names(sim_code):
    meta_path = os.path.join(STORAGE, sim_code, "reverie", "meta.json")
    with open(meta_path) as f:
        return json.load(f)["persona_names"]


def run(sim_code, start_step=0):
    print(f"[driver] Starting headless driver for '{sim_code}' from step {start_step}")
    persona_names = get_persona_names(sim_code)
    print(f"[driver] Personas: {persona_names}")

    step = start_step

    # Seed the first environment file if it doesn't exist yet
    first_env = os.path.join(STORAGE, sim_code, "environment", f"{step}.json")
    if not os.path.exists(first_env):
        # Use last known positions from the most recent movement file
        last_move = None
        for s in range(step - 1, -1, -1):
            m = read_movement(sim_code, s)
            if m:
                last_move = m
                break
        if last_move:
            positions = {
                name: last_move["persona"][name]["movement"]
                for name in persona_names
                if name in last_move["persona"]
            }
        else:
            # Default spawn positions
            positions = {name: [87, 20] for name in persona_names}
        write_environment(sim_code, step, positions)
        print(f"[driver] Seeded environment/{step}.json")

    while True:
        try:
            # Wait for movement/{step}.json to appear and be fully written
            waited = 0
            while True:
                movement = read_movement(sim_code, step)
                if movement:
                    break
                time.sleep(SLEEP_INTERVAL)
                waited += SLEEP_INTERVAL
                if waited >= MAX_WAIT:
                    print(f"[driver] WARNING: waited {MAX_WAIT}s for movement/{step}.json — still waiting...")
                    waited = 0  # reset and keep waiting

            # Extract positions from movement file
            positions = {}
            for name in persona_names:
                if name in movement.get("persona", {}):
                    positions[name] = movement["persona"][name]["movement"]
                else:
                    print(f"[driver] WARNING: {name} missing from movement/{step}.json")
                    positions[name] = [87, 20]

            curr_time = movement.get("meta", {}).get("curr_time", "?")
            print(f"[driver] step {step} → {step+1} | time={curr_time} | "
                  + " | ".join(f"{n}={positions[n]}" for n in persona_names))

            # Write environment/{step+1}.json for the backend
            write_environment(sim_code, step + 1, positions)
            step += 1

        except Exception as e:
            print(f"[driver] ERROR at step {step}: {e} — retrying in 2s...")
            time.sleep(2)
            # Don't advance step — retry the same step


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python headless_driver.py <sim_code> [start_step]")
        sys.exit(1)
    sim_code = sys.argv[1]
    start_step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    try:
        run(sim_code, start_step)
    except KeyboardInterrupt:
        print("\n[driver] Stopped.")
