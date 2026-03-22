#!/usr/bin/env python3
"""
Compresses a running simulation's individual movement files into the
compressed_storage format required by the /demo/ frontend view.

Usage:
    python compress_sim.py <sim_code>          # one-shot compress
    python compress_sim.py <sim_code> --watch  # live: re-compress as steps arrive
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path

BASE = Path(__file__).parent / "environment/frontend_server"
STORAGE = BASE / "storage"
COMPRESSED = BASE / "compressed_storage"


def compress(sim_code, verbose=True):
    src = STORAGE / sim_code
    dst = COMPRESSED / sim_code

    if not src.exists():
        print(f"ERROR: {src} not found")
        return 0

    dst.mkdir(parents=True, exist_ok=True)

    # 1. Copy meta.json
    meta_src = src / "reverie" / "meta.json"
    meta_dst = dst / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, meta_dst)

    # 2. Copy personas directory
    personas_src = src / "personas"
    personas_dst = dst / "personas"
    if personas_src.exists() and not personas_dst.exists():
        shutil.copytree(personas_src, personas_dst)

    # 3. Build master_movement.json from all movement/{step}.json files
    movement_dir = src / "movement"
    if not movement_dir.exists():
        print(f"No movement dir yet for {sim_code}")
        return 0

    step_files = sorted(
        [f for f in movement_dir.iterdir() if f.suffix == ".json"],
        key=lambda f: int(f.stem)
    )

    master = {}
    for step_file in step_files:
        step = step_file.stem  # string key e.g. "0", "1", ...
        try:
            with open(step_file) as f:
                data = json.load(f)
            # master_movement stores persona data directly (no "meta" key)
            master[step] = data.get("persona", data)
        except (json.JSONDecodeError, OSError):
            pass  # skip partial writes

    out_path = dst / "master_movement.json"
    with open(out_path, "w") as f:
        json.dump(master, f)

    n = len(master)
    if verbose:
        print(f"[compress] {sim_code}: {n} steps → compressed_storage/")
    return n


def watch(sim_code, interval=5):
    print(f"[compress] Watching {sim_code} (interval={interval}s)...")
    last_count = 0
    while True:
        try:
            n = compress(sim_code, verbose=False)
            if n != last_count:
                print(f"[compress] {sim_code}: {n} steps compressed")
                last_count = n
        except Exception as e:
            print(f"[compress] ERROR: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_sim.py <sim_code> [--watch]")
        sys.exit(1)

    sim_code = sys.argv[1]
    do_watch = "--watch" in sys.argv

    if do_watch:
        try:
            watch(sim_code)
        except KeyboardInterrupt:
            print("\n[compress] Stopped.")
    else:
        n = compress(sim_code)
        print(f"Done: {n} steps")
