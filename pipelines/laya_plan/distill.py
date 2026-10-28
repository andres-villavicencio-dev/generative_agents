#!/usr/bin/env python3
"""Phase 4: continuous distillation. Every action_location answer the LLM gave since the last round
(shadow passes, or fallbacks when a gate declines) is already in ga_traces/calls, so the harvest is
'all action_location calls since the last cutoff'. One round:
  1. eval the incumbent on the NEW held-out set (served on :8019)
  2. build  data = previous data + loc items since the cutoff           (loc_build.py: LOC_BASE/LOC_SINCE)
  3. train  from the incumbent checkpoint (laya_v6_train.py, 1 epoch; GPU freed first)
  4. eval   the candidate on the same held-out set (temporary server :8020)
  5. promote only if every level's top-1 >= incumbent - 0.005 and the mean improves
Usage: distill.py [--dry]    state: pipelines/laya_plan/distill_state.json"""
import json, os, re, subprocess, sys, time
REPO = "/home/andus/Projects/generative_agents"; LT = os.path.expanduser("~/Projects/laya-test")
ST = f"{REPO}/pipelines/laya_plan/distill_state.json"
PY = f"{REPO}/venv/bin/python"
state = json.load(open(ST)) if os.path.exists(ST) else {
    "round": 0, "cutoff": os.path.getmtime(f"{LT}/v6_data/train.jsonl"),
    "ckpt": f"{LT}/laya-ga-finetuned-v6", "data": f"{LT}/v6_data", "history": []}
dry = "--dry" in sys.argv
rnd = state["round"] + 1
out_data, out_ckpt = f"{LT}/v6r{rnd}_data", f"{LT}/laya-ga-finetuned-v6r{rnd}"

def sh(c, tag):
    print(f"[{tag}] {c[:150]}", flush=True)
    if dry:
        return ""
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-600:], r.stderr[-600:]); raise SystemExit(f"{tag} failed")
    return r.stdout

def result(out):
    m = re.search(r"RESULT (\{.*\})", out)
    return json.loads(m.group(1)) if m else {}

sh(f"LOC_BASE={state['data']} LOC_OUT={out_data} LOC_SINCE={state['cutoff']} {PY} {REPO}/pipelines/laya_plan/loc_build.py 4000", "build")
inc = result(sh(f"LOC_TEST={out_data}/test.jsonl {PY} {REPO}/pipelines/laya_plan/loc_eval.py 8019", "eval-incumbent"))
sh("systemctl --user stop ga-laya-v6; for p in 11434 11436; do for m in qwen2b-ga-chat-v2:latest embeddinggemma:latest; do "
   "curl -s localhost:$p/api/generate -d \"{\\\"model\\\":\\\"$m\\\",\\\"keep_alive\\\":0}\" >/dev/null; done; done; sleep 5", "free-gpu")
sh(f"cd {LT} && .venv/bin/python {REPO}/pipelines/laya_v6_train.py {state['ckpt']} {out_data} {out_ckpt} 1 > v6r{rnd}_train.log 2>&1", "train")
sh(f"cd {LT} && LAYA_CKPT={out_ckpt} LAYA_PORT=8020 LAYA_DEVICE=cuda setsid nohup .venv/bin/python laya_v3_server.py > v6r{rnd}_server.log 2>&1 < /dev/null & echo $! > /tmp/v6r_server.pid; sleep 25", "serve-candidate")
cand = result(sh(f"LOC_TEST={out_data}/test.jsonl {PY} {REPO}/pipelines/laya_plan/loc_eval.py 8020", "eval-candidate"))
sh("kill $(cat /tmp/v6r_server.pid) 2>/dev/null; systemctl --user start ga-laya-v6; sleep 15", "restore")
promote = bool(inc and cand) and all(cand.get(k, 0) >= inc.get(k, 0) - 0.005 for k in inc) and sum(cand.values()) > sum(inc.values())
rec = {"round": rnd, "incumbent": inc, "candidate": cand, "promote": promote, "t": time.time()}
print(json.dumps(rec, indent=1))
if not dry:
    state["history"].append(rec); state["round"] = rnd; state["cutoff"] = time.time()
    state["data"] = out_data
    if promote:
        state["ckpt"] = out_ckpt
    json.dump(state, open(ST, "w"), indent=1)
