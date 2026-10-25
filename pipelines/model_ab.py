#!/usr/bin/env python3
"""Live A/B: chat (convo) + artifact lanes on nemotron-3.5-bolt-e4b vs qwen2b-ga-chat-v2.

Treatment arm `boltchat` forks verify_1 @17280 (same fork + 2400 steps as the
adapt A/B, flags off) with ONLY GA_CONVO_MODEL and GA_ARTIFACT_MODEL swapped; the
fast lane, planning and insight stay on qwen2b. Control = `adaptctl` from
pipelines/adapt_ab (identical fork/steps/flags). Waits for the adapt A/B to
finish first (shared GPU/model lane — never run arms concurrently).
Metrics: pipelines/model_ab_metrics.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatfix_ab as c

REPO = c.REPO
MODEL = os.environ.get("AB_MODEL", "nemotron-3.5-bolt-e4b:latest")
WAIT_FOR = f"{REPO}/adapt_ab_runs.json"

if __name__ == "__main__":
    while not os.path.exists(WAIT_FOR):
        time.sleep(60)
    c.ORIGIN, c.CKPT, c.ORIGIN_LAST_MV = "verify_1", 17280, 17279
    c.END_STEP = c.CKPT + c.STEPS
    c.ENV_BASE += (f"Environment=GA_CONVO_MODEL={MODEL}\n"
                   f"Environment=GA_ARTIFACT_MODEL={MODEL}\n")
    r = c.run_arm("boltchat", "ga-boltchat", "0")
    json.dump(r, open(f"{REPO}/model_ab_runs.json", "w"), indent=1)
    print("DONE", r, flush=True)
