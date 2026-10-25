#!/usr/bin/env python3
"""Pause-aware monitor for the boltchat arm (replaces model_ab.py's run_arm loop
after a deliberate freeze: its 4h cap / 20-min stall abort use wall clock and
would fire on resume). Reads logs/pause_state.json; counts progress only after
the resume time; writes model_ab_runs.json with paused_s so metrics can
subtract the gap."""
import glob, json, os, subprocess, sys, time

REPO = "/home/andus/Projects/generative_agents"
SIM, UNIT, END_STEP = "boltchat", "ga-boltchat", 19680
st = json.load(open(f"{REPO}/logs/pause_state.json"))


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


def frontier():
    f = [int(os.path.basename(p)[:-5]) for p in
         glob.glob(f"{REPO}/environment/frontend_server/storage/{SIM}/movement/*.json")
         if os.path.basename(p)[:-5].isdigit()]
    return max(f) if f else -1


while time.time() < st["resume_at"] + 120:
    time.sleep(60)
last = (frontier(), time.time())
while True:
    time.sleep(60)
    fr = frontier()
    if fr != last[0]:
        last = (fr, time.time())
    if fr >= END_STEP - 1:
        break
    if sh(f"systemctl --user is-active {UNIT}").stdout.strip() != "active":
        print("unit inactive early", flush=True); break
    if time.time() - last[1] > 20 * 60:
        print("no progress 20 min — abort arm", flush=True); break
stop = time.time()
fr = frontier()
sh(f"systemctl --user stop {UNIT}")
res = {"sim": SIM, "legacy": "0", "boot": st["boot"], "stop": stop, "frontier": fr,
       "paused_s": st["resume_at"] - st["pause_start"]}
json.dump(res, open(f"{REPO}/model_ab_runs.json", "w"), indent=1)
print("DONE", res, flush=True)
