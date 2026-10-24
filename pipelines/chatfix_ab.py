#!/usr/bin/env python3
"""Live A/B: chat->replan->reflection fixes (GA_CHAT_REPLAN_LEGACY).

Both arms fork the finished sim `dilate_1` at its final checkpoint (step
17280, Feb 15 05:52 — schedules for the day are still clean) and run the same
number of steps SEQUENTIALLY (shared model lane; see budget_ab.py).
  chatfixctl : GA_CHAT_REPLAN_LEGACY=1  (old clamp / 80-char insert / exact-tick reflect)
  chatfixexp : GA_CHAT_REPLAN_LEGACY=0  (span-preserving splice, richer prompt, robust reflect)
Metrics: pipelines/chatfix_ab_metrics.py
"""
import glob, json, os, shutil, subprocess, sys, time

REPO = "/home/andus/Projects/generative_agents"
STORAGE = f"{REPO}/environment/frontend_server/storage"
ORIGIN, CKPT, ORIGIN_LAST_MV = "dilate_1", 17280, 17279
STEPS = int(os.environ.get("AB_STEPS", "2400"))
END_STEP = CKPT + STEPS
MAX_WALL_S = 4 * 3600
UNIT_DIR = os.path.expanduser("~/.config/systemd/user")
ARMS = [("chatfixctl", "ga-chatfixctl", "1"), ("chatfixexp", "ga-chatfixexp", "0")]

ENV_BASE = """Environment=GA_OLLAMA_URL=http://127.0.0.1:11436
Environment=GA_JEV_SCORING=1
Environment=GA_CHAT_MODEL=qwen2b-ga-chat-v2:latest
Environment=GA_PLANNING_MODEL=qwen2b-ga-chat-v2:latest
Environment=GA_ARTIFACT_MODEL=qwen2b-ga-chat-v2:latest
Environment=GA_CONVO_MODEL=qwen2b-ga-chat-v2:latest
Environment=GA_INSIGHT_MODEL=qwen2b-ga-chat-v2:latest
Environment=GA_LAYA_POIGNANCY=1
Environment=GA_RIZZO_POIGNANCY=0
Environment=GA_RIZZO_REFLEX=0
Environment=GA_DRIVES=1
Environment=GA_ROUTINES=1
Environment=GA_CURRICULUM=1
Environment=GA_CURIOSITY=1
Environment=GA_MOVE_WORKERS=8
Environment=GA_ECONOMY2=1
Environment=GA_TALK_YES_BIAS=2.0
Environment=GA_RECHAT_COOLDOWN_STEPS=120
Environment=GA_POIG_IGNORE_FLOOR=0
Environment=GA_SOCIAL_DRIVE_MULT=1.6
Environment=GA_NUM_PREDICT=2048
Environment=GA_CHAT_TIME=1
Environment=GA_TRACE=1
Environment=GA_TRACE_FULL=1
Environment=GA_FAMILY_BUDGET=1
Environment=GA_FORK_SKIP_COPY=1
"""

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def frontier(sim):
    f = [int(os.path.basename(p)[:-5]) for p in glob.glob(f"{STORAGE}/{sim}/movement/*.json")
         if os.path.basename(p)[:-5].isdigit()]
    return max(f) if f else -1

def prep_fork(sim):
    dst = f"{STORAGE}/{sim}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    sh(f"cp -a {STORAGE}/{ORIGIN} {dst}")
    print(f"forked {ORIGIN} -> {sim}; movement max {frontier(sim)}", flush=True)

def run_arm(sim, unit, legacy):
    print(f"\n===== ARM {sim} legacy={legacy} -> step {END_STEP} =====", flush=True)
    prep_fork(sim)
    path = f"{UNIT_DIR}/{unit}.service"
    open(path, "w").write(f"""[Unit]
Description=GA chat-fix A/B arm {sim} (GA_CHAT_REPLAN_LEGACY={legacy})
After=ga-ollama.service

[Service]
Type=simple
WorkingDirectory={REPO}
{ENV_BASE}Environment=GA_CHAT_REPLAN_LEGACY={legacy}
Environment=GA_TRACE_DIR={REPO}/ga_traces_{sim}
ExecStart={REPO}/venv/bin/python {REPO}/run_live3d_resume.py {ORIGIN} {sim} {CKPT} {END_STEP}
Restart=no
""")
    sh("systemctl --user daemon-reload")
    sh(f"systemctl --user reset-failed {unit}")
    boot = time.time()
    sh(f"systemctl --user start {unit}")
    t0 = time.time()
    last = (-1, time.time())
    while time.time() - t0 < MAX_WALL_S:
        time.sleep(60)
        fr = frontier(sim)
        if fr != last[0]:
            last = (fr, time.time())
        if fr >= END_STEP - 1:
            break
        if sh(f"systemctl --user is-active {unit}").stdout.strip() != "active":
            print("unit inactive early", flush=True); break
        if time.time() - last[1] > 20 * 60:
            print("no progress 20 min — abort arm", flush=True); break
    stop = time.time()
    fr = frontier(sim)
    sh(f"systemctl --user stop {unit}")
    print(f"{sim}: frontier {fr} (fresh {fr - ORIGIN_LAST_MV}) in {(stop-boot)/60:.1f} min", flush=True)
    return {"sim": sim, "legacy": legacy, "boot": boot, "stop": stop, "frontier": fr}

if __name__ == "__main__":
    res = [run_arm(*a) for a in ARMS]
    json.dump(res, open(f"{REPO}/chatfix_ab_runs.json", "w"), indent=1)
    print("DONE", res, flush=True)
