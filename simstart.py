#!/usr/bin/env python3
"""
SimStart — One-command Generative Agents test runner

Replaces the manual 5-step ritual:
  1. cd reverie/backend_server
  2. python reverie.py
  3. Type fork name
  4. Type new sim name  
  5. run <steps>
  6. cd ../.. && python headless_driver.py <sim> 0

Usage:
    python simstart.py                    # Interactive prompts
    python simstart.py --fork needs_base_wakeup --steps 600 --auto
    python simstart.py -f needs_base_wakeup -s 600 -a

Auto-generates sim name from timestamp if not provided.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

BASE_DIR = os.path.expanduser("~/Projects/generative_agents")
REVERIE_DIR = os.path.join(BASE_DIR, "reverie/backend_server")
PYTHON = os.path.expanduser("~/.pyenv/versions/3.11.9/bin/python")


def run_reverie(fork_name: str, new_name: str, steps: int):
    """Start reverie backend with auto-answered prompts."""
    print(f"[simstart] Starting reverie: fork='{fork_name}' → new='{new_name}'")
    
    # Build input sequence
    inputs = f"{fork_name}\n{new_name}\nrun {steps}\n"
    
    proc = subprocess.Popen(
        [PYTHON, "reverie.py"],
        cwd=REVERIE_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Send inputs
    proc.stdin.write(inputs)
    proc.stdin.flush()
    
    # Stream output until we see "Enter option:" (meaning it's ready)
    ready = False
    buffer = ""
    start_time = time.time()
    
    while not ready and time.time() - start_time < 30:
        try:
            char = proc.stdout.read(1)
            if not char:
                break
            buffer += char
            sys.stdout.write(char)
            sys.stdout.flush()
            
            if "Enter option:" in buffer:
                ready = True
                break
        except:
            break
    
    if ready:
        print(f"\n[simstart] Reverie ready! Sim '{new_name}' running.")
    else:
        print(f"\n[simstart] Warning: Reverie may not be fully ready yet")
    
    return proc


def run_driver(sim_name: str, start_step: int = 0):
    """Start headless driver."""
    print(f"[simstart] Starting driver for '{sim_name}' from step {start_step}")
    
    proc = subprocess.Popen(
        [PYTHON, "headless_driver.py", sim_name, str(start_step)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream driver output
    for line in proc.stdout:
        line_str = line if isinstance(line, str) else line.decode()
        sys.stdout.write(line_str)
        sys.stdout.flush()
    
    return proc


def tail_resource_events(sim_name: str):
    """Tail logs filtering for resource events."""
    storage_dir = os.path.join(BASE_DIR, "environment/frontend_server/storage", sim_name)
    
    print(f"\n[simstart] Tailing resource events for '{sim_name}'...")
    print("=" * 60)
    
    # Use tmux to capture from running backend
    # This is a simplified version — full implementation would parse logs
    print("[simstart] Resource event tailing: implement with 'tail -f' on backend log")


def main():
    parser = argparse.ArgumentParser(
        description="One-command Generative Agents test runner"
    )
    parser.add_argument(
        "-f", "--fork",
        default="needs_base_wakeup",
        help="Fork base simulation name (default: needs_base_wakeup)"
    )
    parser.add_argument(
        "-n", "--name",
        help="New simulation name (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=600,
        help="Number of steps to run (default: 600)"
    )
    parser.add_argument(
        "-a", "--auto",
        action="store_true",
        help="Auto-start driver after reverie (no interactive wait)"
    )
    parser.add_argument(
        "--driver-only",
        action="store_true",
        help="Only start driver (reverie already running)"
    )
    parser.add_argument(
        "--sim",
        help="Sim name for --driver-only mode"
    )
    
    args = parser.parse_args()
    
    # Auto-generate sim name if not provided
    if not args.name:
        args.name = f"test_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    
    if args.driver_only:
        if not args.sim:
            print("[simstart] Error: --sim required with --driver-only")
            sys.exit(1)
        run_driver(args.sim)
        return
    
    # Start reverie
    reverie_proc = run_reverie(args.fork, args.name, args.steps)
    
    # Wait a bit for reverie to initialize
    print("[simstart] Waiting 5s for reverie to initialize...")
    time.sleep(5)
    
    # Start driver
    driver_proc = run_driver(args.name)
    
    # Wait for both (or until interrupted)
    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if reverie_proc.poll() is not None:
                print(f"\n[simstart] Reverie exited with code {reverie_proc.poll()}")
                break
            if driver_proc.poll() is not None:
                print(f"\n[simstart] Driver exited with code {driver_proc.poll()}")
                break
    except KeyboardInterrupt:
        print("\n[simstart] Interrupted — shutting down...")
        reverie_proc.terminate()
        driver_proc.terminate()


if __name__ == "__main__":
    main()
